#!/usr/bin/env python3
"""
Adim 5 — Local Dream (Ruya) SD1.5-NPU formatinda paketle.

ZIP KOKUNDE (fazladan klasor YOK) su dosyalar bulunur:
    token_emb.bin      (fp16 gomme tablosu)        [adim 1]
    pos_emb.bin        (fp32 pozisyon gomme)       [adim 1]
    clip_v2.mnn        (CLIP transformer, MNN)     [adim 4]
    unet.bin           (QNN int8, graf 'unet')     [adim 3]
    vae_decoder.bin    (QNN fp16, 'vae_decoder')   [adim 3]
    vae_encoder.bin    (QNN fp16, 'vae_encoder')   [adim 3, opsiyonel]
    tokenizer.json     (HF tokenizer)              [adim 0]

Cikti: <isim>_qnn<surum>_<tier>.zip

Kullanim:
    python 05_package.py --name CyberRealisticLCM --tier min --qnn-version 2.39 \
        --onnx work/onnx --mnn work/mnn --qnn work/qnn \
        --tokenizer work/pipeline/tokenizer --output dist
"""
import argparse
import os
import shutil
import zipfile

TIER_TAIL = {"min": "_min", "mid": "", "high": "_8gen3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tier", default="min", choices=["min", "mid", "high"])
    ap.add_argument("--qnn-version", default="2.39")
    ap.add_argument("--onnx", default="work/onnx", help="token_emb/pos_emb burada")
    ap.add_argument("--mnn", default="work/mnn", help="clip_v2.mnn burada")
    ap.add_argument("--qnn", default="work/qnn", help="unet/vae_*.bin burada")
    ap.add_argument("--tokenizer", default="work/pipeline/tokenizer")
    ap.add_argument("--output", default="dist")
    args = ap.parse_args()

    stage = os.path.join(args.output, args.name)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)

    def copy(src, dst_name, required=True):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage, dst_name))
            print(f"    + {dst_name} ({os.path.getsize(src)>>20} MB)")
            return True
        if required:
            raise SystemExit(f"HATA: gerekli dosya yok: {src}")
        print(f"    - {dst_name} atlandi (yok: {src})")
        return False

    # Zorunlu bilesenler
    copy(os.path.join(args.onnx, "token_emb.bin"), "token_emb.bin")
    copy(os.path.join(args.onnx, "pos_emb.bin"), "pos_emb.bin")
    copy(os.path.join(args.mnn, "clip_v2.mnn"), "clip_v2.mnn")
    copy(os.path.join(args.qnn, "unet.bin"), "unet.bin")
    copy(os.path.join(args.qnn, "vae_decoder.bin"), "vae_decoder.bin")
    # Opsiyonel (img2img)
    copy(os.path.join(args.qnn, "vae_encoder.bin"), "vae_encoder.bin", required=False)

    # clip.mnn — referans paketlerde clip_v2.mnn ile birlikte bulunur
    copy(os.path.join(args.mnn, "clip.mnn"), "clip.mnn", required=False)

    # --- QNN runtime .so'lari (KRITIK) --------------------------------------
    # Motor (QnnRuntime.hpp): g_backendPath = <lib_dir>/libQnnHtp.so
    # Referans model zip'leri bu kutuphaneleri KENDI ICINDE tasir. Eksikse
    # motor backend'i yukleyemez ve coker.
    sdk = os.environ.get("QNN_SDK_ROOT", "")
    if sdk:
        aarch = os.path.join(sdk, "lib", "aarch64-android")
        n = 0
        for fn in ("libQnnHtp.so", "libQnnSystem.so"):
            if copy(os.path.join(aarch, fn), fn, required=False):
                n += 1
        for v in ("68", "69", "73", "75", "79", "81"):
            # Stub: aarch64-android ; Skel: hexagon-vXX/unsigned
            if copy(os.path.join(aarch, f"libQnnHtpV{v}Stub.so"),
                    f"libQnnHtpV{v}Stub.so", required=False):
                n += 1
            skel = os.path.join(sdk, "lib", f"hexagon-v{v}", "unsigned",
                                f"libQnnHtpV{v}Skel.so")
            if copy(skel, f"libQnnHtpV{v}Skel.so", required=False):
                n += 1
        print(f"    [QNN runtime] {n} kutuphane eklendi (SDK: {sdk})")
        if n == 0:
            print("    [!] UYARI: hic QNN .so eklenemedi — motor coksebilir!")
    else:
        print("    [!] UYARI: QNN_SDK_ROOT yok -> QNN runtime .so'lari "
              "EKLENMEDI. Motor backend'i bulamayip cokebilir.")

    # tokenizer.json (diffusers CLIPTokenizerFast bunu uretir)
    tok_json = os.path.join(args.tokenizer, "tokenizer.json")
    if not copy(tok_json, "tokenizer.json", required=False):
        raise SystemExit(
            "HATA: tokenizer.json bulunamadi. diffusers hizli tokenizer uretmemis "
            "olabilir; 'pip install tokenizers' ve pipeline'i yeniden kaydedin.")

    # ZIP (dosyalar KOKte)
    tail = TIER_TAIL[args.tier]
    zip_name = f"{args.name}_qnn{args.qnn_version}{tail}.zip"
    zip_path = os.path.join(args.output, zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(stage)):
            zf.write(os.path.join(stage, fn), fn)

    print(f"\n[+] Paket hazir -> {zip_path}")
    print(f"    Icerik: {', '.join(sorted(os.listdir(stage)))}")


if __name__ == "__main__":
    main()
