#!/usr/bin/env python3
"""
qairt-converter --config icin I/O yapilandirma YAML'i uretir.

NEDEN BU YOL:
  Motor UNet'e sabit tiple yazar — sample/text_embedding/output uint16,
  timestamp int32. Ama v68/v69'da 16-bit LayerNorm YOK, yani ic hesap 8-bit
  olmak zorunda.

  `--quantization_overrides` ile denedik: 16-bit etiketi cross-attention'in
  to_k/to_v MatMul'leri uzerinden ic grafa siziyor ve
  "8 bit activations with 16 bit weights are not supported" cikiyor.

  `--config` (bu dosya) SDK'nin bunun icin tasarlanmis mekanizmasi: yalnizca
  GRAF SINIRINDAKI tensorlerin istenen tipini/kuantizasyon parametrelerini
  belirtir, ic grafi hic etkilemez. Sema qairt-converter'in kendi
  --dump_config_template ciktisindan alindi.

Kuantizasyon: QNN sozlesmesi  gercek = Scale * (q + Offset)
  Scale  = (max - min) / 65535
  Offset = round(min / Scale)          (negatif)

Kullanim:
    python gen_io_config.py --calib work/X/calib/512x512 \
        --output work/X/onnx/unet_io_config.yaml
"""
import argparse
import glob
import os

import numpy as np

BW = 16
QMAX = (1 << BW) - 1


def _range_from_raws(calib_dir: str, prefix: str, pad: float = 1.15):
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


def _scale_offset(vmin: float, vmax: float):
    vmin = float(min(vmin, 0.0))
    vmax = float(max(vmax, 0.0))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-3
    scale = (vmax - vmin) / QMAX
    offset = int(round(vmin / scale))      # negatif
    return scale, offset


def _tensor_block(name: str, dtype: str, idx: int, kind: str,
                  quant=None) -> str:
    lines = [f"  # {kind} {idx}",
             f"  - Name: {name}",
             "    Src Model Parameters:",
             "        DataType: float32",
             "    Desired Model Parameters:",
             f"        DataType: {dtype}"]
    if quant is not None:
        scale, offset = quant
        lines += ["        QuantParams:",
                  f"          Scale: {scale:.10g}",
                  f"          Offset: {offset}"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="UNet kalibrasyon klasoru")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="uint16",
                    help="sinir tensorlerinin istenen tipi (uint16/int16)")
    args = ap.parse_args()

    r_sample = _range_from_raws(args.calib, "sample") or (-6.0, 6.0)
    r_text = _range_from_raws(args.calib, "ehs") or (-30.0, 30.0)
    r_out = (r_sample[0] * 1.6, r_sample[1] * 1.6)

    q_sample = _scale_offset(*r_sample)
    q_text = _scale_offset(*r_text)
    q_out = _scale_offset(*r_out)

    body = [
        "# qairt-converter --config  (sema: --dump_config_template)",
        "# 16-bit GRAF SINIRI + 8-bit ic hesap. Motorun bekledigi tipler:",
        "#   sample / text_embedding / output -> uint16 (UFIXED_POINT_16)",
        "#   timestamp                        -> int32",
        "Converted Graph:",
        "  - Input Tensors:",
        "  - Output Tensors:",
        "",
        "Input Tensor Configuration:",
        _tensor_block("sample", args.dtype, 1, "Input", q_sample),
        _tensor_block("timestamp", "int32", 2, "Input"),
        _tensor_block("text_embedding", args.dtype, 3, "Input", q_text),
        "",
        "Output Tensor Configuration:",
        _tensor_block("output", args.dtype, 1, "Output", q_out),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(body) + "\n")

    print(f"[+] I/O config yazildi -> {args.output}")
    for n, (s, o), r in (("sample", q_sample, r_sample),
                         ("text_embedding", q_text, r_text),
                         ("output", q_out, r_out)):
        print(f"    {n:16s} {args.dtype}  scale={s:.4e} offset={o:<8d} "
              f"[{r[0]:.3f}, {r[1]:.3f}]")
    print("    timestamp        int32 (kuantize edilmez)")


if __name__ == "__main__":
    main()
