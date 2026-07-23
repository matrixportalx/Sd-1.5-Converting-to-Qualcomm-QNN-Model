#!/usr/bin/env python3
"""
Adim 5 — Bilesenleri Local Dream (Ruya) klasor yapisinda topla ve
`<isim>_qnn<surum>_min.zip` (varsayilan surum 2.39) olarak paketle.

Beklenen icerik (Local Dream SD1.5 / NPU modeli):
    <isim>/
      unet_512x512.bin        <- QNN context binary (NPU)         [adim 3]
      unet_512x768.bin        (varsa)
      unet_768x512.bin        (varsa)
      text_encoder.mnn        <- CLIP (CPU/GPU)                   [adim 4]
      vae.mnn                 <- VAE decoder (CPU/GPU)            [adim 4]
      tokenizer/              <- diffusers tokenizer dosyalari     [adim 0]
      model_info.json         <- meta veri (isim, cozunurlukler, tier)

NOT: Ruya/Local Dream'in bekledigi kesin dosya adlari ve model_info semasi
uygulama surumune gore degisebilir. Ithal calismazsa, halihazirda calisan bir
resmi ".._qnn2.28_min.zip" (or. AbsoluteReality) icini acip dosya adlarini ve
model_info.json duzenini bununla birebir eslestirin.

Kullanim:
    python 05_package.py --name AbsoluteReality --tier min \
        --qnn work/qnn --mnn work/mnn --tokenizer work/pipeline/tokenizer \
        --resolutions 512x512,512x768,768x512 --output dist
"""
import argparse
import json
import os
import shutil
import zipfile

from soc_targets import get_tier

# tier -> ZIP ekindeki tier parcasi (sürüm ayri --qnn-version ile eklenir)
TIER_TAIL = {"min": "_min", "mid": "", "high": "_8gen3"}


def zip_suffix(qnn_version: str, tier: str) -> str:
    """Ornek: (2.39, min) -> '_qnn2.39_min'"""
    return f"_qnn{qnn_version}{TIER_TAIL[tier]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Model adi (or. AbsoluteReality)")
    ap.add_argument("--tier", default="min", choices=["min", "mid", "high"])
    ap.add_argument("--qnn-version", default="2.39",
                    help="ZIP/model_info sürüm etiketi (varsayilan: 2.39)")
    ap.add_argument("--qnn", default="work/qnn", help="unet_*.bin klasoru")
    ap.add_argument("--mnn", default="work/mnn", help="text_encoder.mnn / vae.mnn")
    ap.add_argument("--tokenizer", default="work/pipeline/tokenizer")
    ap.add_argument("--resolutions", default="512x512,512x768,768x512")
    ap.add_argument("--output", default="dist")
    args = ap.parse_args()

    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    tier = get_tier(args.tier)

    stage = os.path.join(args.output, args.name)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)

    # UNet context binary'leri
    included_res = []
    for tag in resolutions:
        src = os.path.join(args.qnn, f"unet_{tag}.bin")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage, f"unet_{tag}.bin"))
            included_res.append(tag)
        else:
            print(f"[!] Atlaniyor (bulunamadi): {src}")
    if not included_res:
        raise SystemExit("HATA: Hic UNet .bin bulunamadi. Once adim 3'u calistirin.")

    # MNN bilesenleri
    for fn in ("text_encoder.mnn", "vae.mnn"):
        src = os.path.join(args.mnn, fn)
        if not os.path.exists(src):
            raise SystemExit(f"HATA: {src} yok. Once adim 4'u calistirin.")
        shutil.copy2(src, os.path.join(stage, fn))

    # Tokenizer
    if os.path.isdir(args.tokenizer):
        shutil.copytree(args.tokenizer, os.path.join(stage, "tokenizer"))
    else:
        print(f"[!] Tokenizer klasoru yok: {args.tokenizer} (atlaniyor)")

    # Meta veri
    model_info = {
        "name": args.name,
        "base": "sd1.5",
        "runtime": f"qnn{args.qnn_version}",
        "tier": args.tier,
        "dsp_arch": tier["dsp_arch"],
        "resolutions": included_res,
        "unet": {r: f"unet_{r}.bin" for r in included_res},
        "text_encoder": "text_encoder.mnn",
        "vae": "vae.mnn",
    }
    with open(os.path.join(stage, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    # ZIP
    zip_name = f"{args.name}{zip_suffix(args.qnn_version, args.tier)}.zip"
    zip_path = os.path.join(args.output, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, args.output)
                zf.write(full, arc)

    print(f"[+] Paket hazir -> {zip_path}")
    print(f"    Cozunurlukler: {', '.join(included_res)} | tier={args.tier} "
          f"(dsp_arch={tier['dsp_arch']})")


if __name__ == "__main__":
    main()
