#!/usr/bin/env python3
"""
Adim 2 — UNet kuantizasyonu icin kalibrasyon verisi uretir.

QNN, UNet'i INT8/INT16'ya kuantize ederken tipik girdi dagilimini gormek ister.
Bu script her UNet girdisi (sample, timestep, encoder_hidden_states) icin ham
(raw, little-endian float32) tensor dosyalari ve bunlari listeleyen bir
`input_list.txt` uretir. `qnn-onnx-converter --input_list` bunu kullanir.

Iki mod:
  * --mode random : hizli; gercek dagilimi tam yansitmaz (kaba kuantizasyon).
  * --mode real   : diffusers pipeline'i birkac prompt icin gercekten calistirip
                    ara latent/timestep/embedding'leri toplar (daha iyi kalite).

Kullanim:
    python 02_gen_quant_data.py --pipeline work/pipeline \
        --resolution 512x512 --output work/calib/512x512 \
        --mode real --num-samples 8 --steps 20
"""
import argparse
import os

import numpy as np
import torch

from common import LATENT_CHANNELS, TEXT_SEQ_LEN, Resolution


def _save_raw(arr: np.ndarray, path: str) -> None:
    arr.astype(np.float32).tofile(path)


def _save_timestep(t, path: str) -> None:
    """ONNX'te 'timestamp' girisi INT_32'dir (referans binary de INT_32).
    Kalibrasyonu float32 yazarsak quantizer bit desenini tamsayi olarak okur
    (1.0 -> 1065353216) ve zaman-gomme yolunun tum encoding'leri bozulur."""
    np.array([int(round(float(t)))], dtype=np.int32).tofile(path)


def gen_random(res: Resolution, hidden: int, n: int, out: str):
    os.makedirs(out, exist_ok=True)
    lines = []
    for i in range(n):
        s = np.random.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w)
        t = np.random.randint(0, 1000)
        e = np.random.randn(1, TEXT_SEQ_LEN, hidden)
        sp = os.path.join(out, f"sample_{i:03d}.raw")
        tp = os.path.join(out, f"timestep_{i:03d}.raw")
        ep = os.path.join(out, f"ehs_{i:03d}.raw")
        _save_raw(s, sp); _save_timestep(t, tp); _save_raw(e, ep)
        lines.append(f"sample:={sp} timestamp:={tp} text_embedding:={ep}")
    _write_list(out, lines)


def gen_real(pipeline_dir, res: Resolution, n: int, steps: int, out: str):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    os.makedirs(out, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        pipeline_dir, torch_dtype=torch.float32,
        safety_checker=None, load_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    hidden = pipe.text_encoder.config.hidden_size

    prompts = [
        "a portrait of a woman, detailed, photorealistic",
        "a landscape with mountains and a lake at sunset",
        "a cat sitting on a wooden table, studio light",
        "a futuristic city street at night, neon signs",
        "a bowl of fruit on a kitchen counter",
        "a fantasy castle on a hill, dramatic clouds",
        "a close up of a flower with dew drops",
        "an astronaut riding a horse on mars",
    ]

    captured = []  # (sample, timestep, ehs)

    def hook(module, inp):
        # UNet.forward(sample, timestep, encoder_hidden_states=...)
        sample = inp[0].detach().cpu().numpy()
        timestep = inp[1]
        if torch.is_tensor(timestep):
            timestep = timestep.detach().cpu().float().numpy().reshape(-1)[:1]
        else:
            timestep = np.array([float(timestep)], dtype=np.float32)
        return None

    # UNet cagrilarini toplamak icin forward_pre_hook yerine dogrudan
    # scheduler dongusunu kullaniyoruz (encoder_hidden_states kwarg oldugundan).
    lines = []
    idx = 0
    generator = torch.Generator().manual_seed(0)
    for p in prompts[:n]:
        # metin gomme
        ids = pipe.tokenizer(p, padding="max_length",
                             max_length=TEXT_SEQ_LEN, truncation=True,
                             return_tensors="pt").input_ids
        with torch.no_grad():
            ehs = pipe.text_encoder(ids)[0]
        latent = torch.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w,
                             generator=generator)
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            with torch.no_grad():
                noise = pipe.unet(latent, t, encoder_hidden_states=ehs).sample
            # bu adimin girdilerini kaydet
            sp = os.path.join(out, f"sample_{idx:04d}.raw")
            tp = os.path.join(out, f"timestep_{idx:04d}.raw")
            ep = os.path.join(out, f"ehs_{idx:04d}.raw")
            _save_raw(latent.cpu().numpy(), sp)
            _save_timestep(t, tp)
            _save_raw(ehs.cpu().numpy(), ep)
            lines.append(
                f"sample:={sp} timestamp:={tp} text_embedding:={ep}")
            latent = pipe.scheduler.step(noise, t, latent).prev_sample
            idx += 1
    _write_list(out, lines)
    print(f"[+] {idx} kalibrasyon ornegi uretildi.")


def gen_vae_decoder_calib(pipeline_dir, res, n, steps, out):
    """VAE decoder kalibrasyonu: birkac prompt icin denoising calistirip NIHAI
    latent'i alir ve OLCEKSIZ (latent/scaling_factor) haliyle 'latent' girisi
    olarak kaydeder — decoder'in cihazda gordugu dagilim budur."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    os.makedirs(out, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        pipeline_dir, torch_dtype=torch.float32,
        safety_checker=None, load_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    sf = pipe.vae.config.scaling_factor

    prompts = [
        "a portrait of a woman, detailed, photorealistic",
        "a landscape with mountains and a lake at sunset",
        "a cat sitting on a wooden table, studio light",
        "a futuristic city street at night, neon signs",
        "a bowl of fruit on a kitchen counter",
        "a fantasy castle on a hill, dramatic clouds",
    ]
    lines = []
    gen = torch.Generator().manual_seed(0)
    for i, p in enumerate(prompts[:n]):
        ids = pipe.tokenizer(p, padding="max_length", max_length=TEXT_SEQ_LEN,
                             truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            ehs = pipe.text_encoder(ids)[0]
        latent = torch.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w,
                             generator=gen)
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            with torch.no_grad():
                noise = pipe.unet(latent, t, encoder_hidden_states=ehs).sample
            latent = pipe.scheduler.step(noise, t, latent).prev_sample
        dec_in = (latent / sf).cpu().numpy().astype(np.float32)  # olceksiz
        pth = os.path.join(out, f"latent_{i:03d}.raw")
        dec_in.tofile(pth)
        lines.append(f"latent:={pth}")
    _write_list(out, lines)
    print(f"[+] {len(lines)} VAE decoder kalibrasyon ornegi uretildi.")


def gen_vae_encoder_calib(pipeline_dir, res, n, steps, out):
    """VAE encoder kalibrasyonu: nihai latent'i fp32 VAE ile decode edip
    goruntu ([-1,1]) uretir ve 'image' girisi olarak kaydeder."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    os.makedirs(out, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        pipeline_dir, torch_dtype=torch.float32,
        safety_checker=None, load_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    sf = pipe.vae.config.scaling_factor
    prompts = [
        "a portrait of a woman, detailed, photorealistic",
        "a landscape with mountains and a lake at sunset",
        "a cat sitting on a wooden table, studio light",
        "a futuristic city street at night, neon signs",
        "a bowl of fruit on a kitchen counter",
        "a fantasy castle on a hill, dramatic clouds",
    ]
    lines = []
    gen = torch.Generator().manual_seed(1)
    for i, p in enumerate(prompts[:n]):
        ids = pipe.tokenizer(p, padding="max_length", max_length=TEXT_SEQ_LEN,
                             truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            ehs = pipe.text_encoder(ids)[0]
        latent = torch.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w,
                             generator=gen)
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            with torch.no_grad():
                noise = pipe.unet(latent, t, encoder_hidden_states=ehs).sample
            latent = pipe.scheduler.step(noise, t, latent).prev_sample
        with torch.no_grad():
            img = pipe.vae.decode(latent / sf).sample  # [1,3,H,W] ~[-1,1]
        img = img.clamp(-1, 1).cpu().numpy().astype(np.float32)
        pth = os.path.join(out, f"image_{i:03d}.raw")
        img.tofile(pth)
        lines.append(f"image:={pth}")
    _write_list(out, lines)
    print(f"[+] {len(lines)} VAE encoder kalibrasyon ornegi uretildi.")


def _abs_token(tok: str) -> str:
    """'name:=path' veya 'path' icindeki yolu mutlak yapar. qairt-quantizer
    input_list'i farkli bir CWD'den okudugu icin goreli yollar bulunamaz."""
    if ":=" in tok:
        name, p = tok.split(":=", 1)
        return f"{name}:={os.path.abspath(p)}"
    return os.path.abspath(tok) if tok else tok


def _write_list(out: str, lines) -> None:
    abs_lines = [" ".join(_abs_token(t) for t in line.split(" ") if t)
                 for line in lines]
    with open(os.path.join(out, "input_list.txt"), "w") as f:
        f.write("\n".join(abs_lines) + "\n")
    print(f"[+] input_list.txt yazildi ({len(abs_lines)} satir, mutlak yol) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--resolution", default="512x512")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["random", "real"], default="real")
    ap.add_argument("--target", choices=["unet", "vae_decoder", "vae_encoder"],
                    default="unet")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    w, h = args.resolution.lower().split("x")
    res = Resolution(int(w), int(h))

    if args.target == "vae_decoder":
        gen_vae_decoder_calib(args.pipeline, res, args.num_samples,
                              args.steps, args.output)
    elif args.target == "vae_encoder":
        gen_vae_encoder_calib(args.pipeline, res, args.num_samples,
                              args.steps, args.output)
    elif args.mode == "random":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.pipeline, torch_dtype=torch.float32,
            safety_checker=None, load_safety_checker=False)
        gen_random(res, pipe.text_encoder.config.hidden_size,
                   args.num_samples, args.output)
    else:
        gen_real(args.pipeline, res, args.num_samples, args.steps, args.output)


if __name__ == "__main__":
    main()
