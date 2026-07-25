#!/usr/bin/env python3
"""
UNet grafi icin KARMA HASSASIYET kuantizasyon override'i uretir.

Neden: Referans binary (calisan model) sunu kullaniyor —
    sample         UFIXED_POINT_16
    timestamp      INT_32
    text_embedding UFIXED_POINT_16
    output         UFIXED_POINT_16
    dspArch        68
Ama v68/v69 donaniminda 16-bit MatMul DESTEKLENMEZ (QAIRT 2.39 dogrulayicisi
'expected >= 73' der). Yani ic hesaplar 8-bit olmali, YALNIZCA graf giris/
ciktilari 16-bit. Bunu qairt-converter --quantization_overrides ile yapariz;
QNN, 16-bit sinir ile 8-bit ic graf arasina otomatik Convert op'lari ekler.

Encoding araliklari kalibrasyon .raw dosyalarindan (gercek dagilim) hesaplanir.

Kullanim:
    python gen_io_encodings.py --calib work/calib/512x512 \
        --hidden 768 --latent-h 64 --latent-w 64 \
        --output work/onnx/unet_io_encodings.json
"""
import argparse
import glob
import json
import os

import numpy as np

BW = 16
QMAX = (1 << BW) - 1


def _enc(vmin: float, vmax: float) -> dict:
    """Asimetrik (unsigned) 16-bit encoding — UFIXED_POINT_16."""
    vmin = float(min(vmin, 0.0))
    vmax = float(max(vmax, 0.0))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-3
    scale = (vmax - vmin) / QMAX
    offset = -int(round(vmin / scale))          # QNN: gercek = scale*(q - offset)
    return {
        "bitwidth": BW,
        "dtype": "int",
        "is_symmetric": "False",
        "max": vmax,
        "min": vmin,
        "offset": -offset,
        "scale": scale,
    }


def _range_from_raws(calib_dir: str, prefix: str, pad: float = 1.15):
    """Kalibrasyon .raw dosyalarindan min/max (guvenlik payiyla)."""
    files = sorted(glob.glob(os.path.join(calib_dir, f"{prefix}_*.raw")))
    if not files:
        return None
    lo, hi = np.inf, -np.inf
    for f in files:
        a = np.fromfile(f, dtype=np.float32)
        if a.size:
            lo = min(lo, float(a.min()))
            hi = max(hi, float(a.max()))
    if not np.isfinite(lo):
        return None
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * pad
    return c - half, c + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="UNet kalibrasyon klasoru")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # sample (latent) ve text_embedding araliklari kalibrasyondan
    r_sample = _range_from_raws(args.calib, "sample") or (-6.0, 6.0)
    r_text = _range_from_raws(args.calib, "ehs") or (-30.0, 30.0)
    # output (noise_pred) latent ile benzer olcekte; biraz genis tut
    r_out = (r_sample[0] * 1.6, r_sample[1] * 1.6)

    enc = {
        "activation_encodings": {
            "sample": [_enc(*r_sample)],
            "text_embedding": [_enc(*r_text)],
            "output": [_enc(*r_out)],
        },
        "param_encodings": {},
    }
    # NOT: 'timestamp' INT_32 girdidir — kuantize edilmez, override verilmez.

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(enc, f, indent=2)

    print(f"[+] I/O encodings (16-bit) yazildi -> {args.output}")
    for k, v in enc["activation_encodings"].items():
        e = v[0]
        print(f"    {k:16s} bw={e['bitwidth']} min={e['min']:.4f} "
              f"max={e['max']:.4f} scale={e['scale']:.3e}")


if __name__ == "__main__":
    main()
