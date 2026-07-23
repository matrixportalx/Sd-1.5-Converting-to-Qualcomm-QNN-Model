#!/usr/bin/env python3
"""
Adim 0 — safetensors (tek dosya) -> diffusers klasor yapisi.

civitai.red / Hugging Face'ten indirdiginiz tek parca SD1.5 .safetensors
dosyasini, diffusers'in bilesenlere ayrilmis (text_encoder / unet / vae)
klasor duzenine cevirir. Sonraki adimlar bu klasoru kullanir.

Kullanim:
    python 00_load_safetensors.py \
        --checkpoint /yol/AbsoluteReality.safetensors \
        --output work/pipeline

Not: Model SD1.5 mimarisi olmalidir (SDXL/SD2.x degil). SDXL icin Local Dream
yalnizca Snapdragon 8 Gen 3+ destekler ve akis farklidir.
"""
import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="safetensors -> diffusers pipeline")
    ap.add_argument("--checkpoint", required=True,
                    help="Tek parca SD1.5 .safetensors dosyasi")
    ap.add_argument("--output", default="work/pipeline",
                    help="Cikti diffusers klasoru")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="Yukleme hassasiyeti (ONNX export icin fp32 onerilir)")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"[*] Checkpoint yukleniyor: {args.checkpoint}")
    pipe = StableDiffusionPipeline.from_single_file(
        args.checkpoint,
        torch_dtype=dtype,
        safety_checker=None,
        load_safety_checker=False,
    )

    os.makedirs(args.output, exist_ok=True)
    print(f"[*] diffusers klasoru yaziliyor: {args.output}")
    pipe.save_pretrained(args.output)
    print("[+] Tamam. Bilesenler: text_encoder/ unet/ vae/ tokenizer/")


if __name__ == "__main__":
    main()
