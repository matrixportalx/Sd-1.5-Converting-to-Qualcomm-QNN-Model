#!/usr/bin/env python3
"""
LoRA'lari SD1.5 checkpoint'ine KAYNASTIRIR ve yeni bir .safetensors yazar.

Neden bu asamada:
    NPU'da calisma zamani LoRA MUMKUN DEGIL. QNN grafigi derlenmis ve
    agirliklari icine gomulmus; disaridan agirlik delta'si eklenecek bir yer
    yok. Tek yol, LoRA'yi daha ONCE — ONNX'e cikmadan once — temel agirliklara
    islemek. Ag YAPISI degismiyor, yalnizca sayilar kayiyor: yeni op yok,
    niceleme riski yok. (IP-Adapter'dan farki tam olarak bu; o yeni katman
    ekliyor, bu eklemiyor.)

Neden CHECKPOINT'e, diffusers dizinine degil:
    Hat iki yerde model okuyor ve IKISI DE ayni agirliklari gormeli:

        prepare_data.py  --model_path <ckpt>   -> kalibrasyon verisi
        export_onnx.py   --model_path ./model  -> ONNX

    LoRA yalnizca export tarafina islenirse kalibrasyon TEMEL modelden
    cikar; niceleme araliklari kaymis agirliklarla uyusmaz ve sonuc sessizce
    bozulur. Checkpoint'e kaynastirmak ikisini de tek hamlede duzeltiyor ve
    resmi scriptlerin hicbirine dokunmuyor.

Kullanim:
    python fuse_lora.py --ckpt work/input.safetensors \
                        --out  work/input_lora.safetensors \
                        --lora style.safetensors:0.8 detail.safetensors:0.5

Agirlik ':' ile veriliyor; verilmezse 0.8 (Ruya'nin CPU yolundaki varsayilanla
ayni — iki yol arasinda ayni LoRA'nin farkli davranmasi kafa karistirici olur).
"""
import argparse
import os
import sys


def parse_lora_arg(raw: str):
    """'dosya.safetensors:0.7' -> ('dosya.safetensors', 0.7)"""
    # rsplit: Windows yollarindaki 'C:' iki nokta ustusunu yutmasin.
    if ":" in raw:
        path, _, weight = raw.rpartition(":")
        try:
            return path, float(weight)
        except ValueError:
            # 'C:/yol/lora.safetensors' gibi bir durumda son parca sayi degil;
            # o zaman tamami yoldur.
            return raw, 0.8
    return raw, 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="temel SD1.5 .safetensors")
    ap.add_argument("--out", required=True, help="yazilacak kaynastirilmis .safetensors")
    ap.add_argument("--lora", nargs="+", required=True,
                    help="lora.safetensors[:agirlik] (birden fazla verilebilir)")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"HATA: checkpoint yok -> {args.ckpt}", file=sys.stderr)
        return 1

    loras = [parse_lora_arg(x) for x in args.lora]
    for path, _ in loras:
        if not os.path.isfile(path):
            print(f"HATA: LoRA dosyasi yok -> {path}", file=sys.stderr)
            return 1

    import torch
    from diffusers import StableDiffusionPipeline

    print(f"[fuse] temel model yukleniyor: {args.ckpt}")
    # safety_checker=None: bu adimda uretim yapilmiyor, yalnizca agirlik
    # islenip geri yaziliyor. Yuklemesi bosuna ~1 GB RAM ve dakika.
    pipe = StableDiffusionPipeline.from_single_file(
        args.ckpt, torch_dtype=torch.float32, safety_checker=None
    )

    for path, weight in loras:
        name = os.path.basename(path)
        print(f"[fuse] {name} (agirlik {weight}) kaynastiriliyor")
        # adapter_name: ayni pipeline'a birden fazla LoRA yuklenince
        # diffusers her birini ADIYLA ayiriyor; ad verilmezse ikincisi
        # birincinin uzerine yaziliyor ve sessizce tek LoRA uygulanmis oluyor.
        adapter = os.path.splitext(name)[0].replace(".", "_")
        pipe.load_lora_weights(path, adapter_name=adapter)
        pipe.fuse_lora(lora_scale=weight, adapter_names=[adapter])
        # Kaynastirildiktan sonra adapter'i BOSALTMAK sart: yuklu kalirsa
        # bir sonraki fuse_lora onu da ikinci kez isler.
        pipe.unload_lora_weights()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # Tek dosya .safetensors olarak geri yazmak diffusers'ta dogrudan yok;
    # resmi hat da zaten diffusers DIZINI kabul ediyor (model_path_arg).
    # Dizin yazip yolunu bildirmek, tek dosyaya geri donusturmekten hem hizli
    # hem kayipsiz.
    out_dir = os.path.splitext(args.out)[0] + "_diffusers"
    print(f"[fuse] yaziliyor: {out_dir}")
    pipe.save_pretrained(out_dir, safe_serialization=True)
    print(f"[fuse] BITTI -> {out_dir}")
    print("[fuse] Hat bu dizini --model_path olarak kullanmali.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
