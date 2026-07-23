#!/usr/bin/env python3
"""
Adim 1 — diffusers pipeline -> ONNX (sabit / static sekilli).

Uc bileseni ayri ayri ONNX'e verir:
  * text_encoder.onnx  (CLIP)      -> sonra MNN'e cevrilir (CPU/GPU)
  * vae_decoder.onnx   (VAE decode)-> sonra MNN'e cevrilir (CPU/GPU)
  * unet_<WxH>.onnx     (UNet)     -> sonra QNN context binary'ye (NPU)

NEDEN sabit sekil? NPU (HTP) dinamik sekil sevmez; her cozunurluk icin ayri
bir UNet grafi export edilir. Local Dream varsayilan olarak 512x512, 512x768
ve 768x512 paketler.

Kullanim:
    python 01_export_onnx.py --pipeline work/pipeline --output work/onnx \
        --resolutions 512x512,512x768,768x512 --opset 17
"""
import argparse
import os

import torch

from common import (LATENT_CHANNELS, TEXT_SEQ_LEN, parse_resolutions,
                    DEFAULT_RESOLUTIONS)


def export_text_encoder(pipe, out_dir, opset):
    te = pipe.text_encoder.eval()
    path = os.path.join(out_dir, "text_encoder.onnx")
    dummy = torch.randint(0, 1000, (1, TEXT_SEQ_LEN), dtype=torch.int32)
    print(f"[*] text_encoder -> {path}")
    torch.onnx.export(
        te, (dummy,), path,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=opset,
        do_constant_folding=True,
    )


def export_vae_decoder(pipe, out_dir, opset, res0):
    vae = pipe.vae.eval()

    class Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            # diffusers latent olcegi: 1/0.18215
            latent = latent / self.vae.config.scaling_factor
            return self.vae.decode(latent).sample

    path = os.path.join(out_dir, "vae_decoder.onnx")
    dummy = torch.randn(1, LATENT_CHANNELS, res0.latent_h, res0.latent_w)
    print(f"[*] vae_decoder -> {path}")
    torch.onnx.export(
        Decoder(vae), (dummy,), path,
        input_names=["latent"],
        output_names=["image"],
        opset_version=opset,
        do_constant_folding=True,
    )


def export_unet(pipe, out_dir, opset, res):
    unet = pipe.unet.eval()
    hidden = pipe.text_encoder.config.hidden_size

    class UNetWrap(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.unet(sample, timestep,
                             encoder_hidden_states=encoder_hidden_states).sample

    path = os.path.join(out_dir, f"unet_{res.tag}.onnx")
    sample = torch.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w)
    timestep = torch.tensor(1, dtype=torch.int64)
    ehs = torch.randn(1, TEXT_SEQ_LEN, hidden)
    print(f"[*] unet {res.tag} -> {path}")
    torch.onnx.export(
        UNetWrap(unet), (sample, timestep, ehs), path,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        opset_version=opset,
        do_constant_folding=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, help="diffusers klasoru (adim 0)")
    ap.add_argument("--output", default="work/onnx")
    ap.add_argument("--resolutions", default="",
                    help="or. 512x512,512x768,768x512 (bos = varsayilan uclu)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--unet-only", action="store_true",
                    help="Sadece UNet export et (text_encoder/vae zaten hazirsa)")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    os.makedirs(args.output, exist_ok=True)

    resolutions = (parse_resolutions(args.resolutions)
                   if args.resolutions else DEFAULT_RESOLUTIONS)

    print(f"[*] Pipeline yukleniyor: {args.pipeline}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pipeline, torch_dtype=torch.float32,
        safety_checker=None, load_safety_checker=False)

    with torch.no_grad():
        if not args.unet_only:
            export_text_encoder(pipe, args.output, args.opset)
            export_vae_decoder(pipe, args.output, args.opset, resolutions[0])
        for res in resolutions:
            export_unet(pipe, args.output, args.opset, res)

    print("[+] ONNX export tamam ->", args.output)


if __name__ == "__main__":
    main()
