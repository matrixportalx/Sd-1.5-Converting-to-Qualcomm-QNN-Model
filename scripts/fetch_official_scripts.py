#!/usr/bin/env python3
"""
Local Dream'in RESMI SD1.5 NPU donusturme scriptlerini (npuconvertv2.zip)
indirir, acar ve kritik dosyalari LOGA DOKER.

Neden: rehber (ld-guide.chino.icu/conversion/sd15) donusumun QNN SDK 2.28 ile
ve kendi scriptleriyle yapilmasini soyluyor. Bizim tersine muhendislikle
buldugumuz her sey (girdi sirasi, tip sinirlari, graf adi) bu scriptlerde
zaten yaziyor olmali. Ozellikle merak ettigimiz:

    scripts/convert_all.sh   -> unet.bin hangi arac zinciriyle uretiliyor?
                                (qnn-onnx-converter -> qnn-model-lib-generator
                                 -> qnn-context-binary-generator --model ?)
    export_onnx.py           -> ONNX girdi adlari/sirasi, timestamp tipi
    gen_quant_data.py        -> kalibrasyon (bizde FAST_TRIAL ile kisaltilmis)

Kullanim:
    python fetch_official_scripts.py --dest work/_official
    OFFICIAL_SCRIPTS_URL=<dogrudan link> python fetch_official_scripts.py ...
"""
import argparse
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

GUIDE = "https://ld-guide.chino.icu/conversion/sd15"
UA = {"User-Agent": "Mozilla/5.0 (curl-like)"}

# Loga dokulecek dosyalar: adi eslesirse basilir (en degerliler basta)
# EN KRITIK OLANLAR BASTA. convert_all.sh yalnizca envsetup.sh'i source edip
# alt scriptleri cagiriyor; asil arac zinciri convert_unet.sh icinde.
DUMP = [
    "scripts/convert_unet.sh",          # <-- unet.bin'i ureten gercek zincir
    "scripts/convert_clip.sh",
    "scripts/convert_vae_decoder.sh",
    "scripts/convert_vae_encoder.sh",
    "htp_config_min.json",
    "htp_backend_min.json",
    # ONNX'i HTP dostu yapan degistirilmis diffusers modulleri
    "redefined_modules/diffusers/models/embeddings.py",   # timestamp yolu
    "redefined_modules/diffusers/models/attention.py",    # MHA -> SHA (conv)
    "export.sh",
]
DUMP_MAX_LINES = 400


def _get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def discover_url() -> str:
    """Rehber sayfasindan npuconvert*.zip linkini cikar."""
    print(f"[*] rehber taraniyor: {GUIDE}")
    html = _get(GUIDE).decode("utf-8", "replace")
    cands = re.findall(r'href="([^"]+\.zip)"', html)
    cands += re.findall(r'https?://[^\s"\'<>]+\.zip', html)
    seen, out = set(), []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    if not out:
        raise SystemExit(
            "Rehberde .zip linki bulunamadi. Linki elle verin:\n"
            "  config.env -> OVERRIDE_OFFICIAL_SCRIPTS_URL=<link>")
    print(f"[*] bulunan .zip linkleri: {out}")
    pref = [c for c in out if "npuconvert" in c.lower()]
    pick = (pref or out)[0]
    return urllib.parse.urljoin(GUIDE, pick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="work/_official")
    ap.add_argument("--url", default=os.environ.get("OFFICIAL_SCRIPTS_URL", ""))
    ap.add_argument("--no-dump", action="store_true")
    args = ap.parse_args()

    url = args.url or discover_url()
    print(f"[*] indiriliyor: {url}")
    os.makedirs(args.dest, exist_ok=True)
    zpath = os.path.join(args.dest, "npuconvert.zip")
    if not os.path.exists(zpath):
        data = _get(url)
        with open(zpath, "wb") as f:
            f.write(data)
    print(f"[+] {os.path.getsize(zpath)/1e6:.1f} MB -> {zpath}")

    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        zf.extractall(args.dest)
    print(f"[+] {len(names)} dosya acildi")

    # ZIP calistirma bitlerini korumuyor: pakette gelen MNNConvert ikilisi ve
    # .sh scriptleri aksi halde "Permission denied" veriyor.
    n_exec = 0
    for n in names:
        base = os.path.basename(n)
        if base == "MNNConvert" or base.endswith(".sh"):
            f = os.path.join(args.dest, n)
            if os.path.isfile(f):
                os.chmod(f, os.stat(f).st_mode | 0o111)
                n_exec += 1
    if n_exec:
        print(f"[+] {n_exec} dosyaya calistirma izni verildi")
    print("\n--- ICERIK ---")
    for n in sorted(names):
        print("   ", n)

    if args.no_dump:
        return

    # Kritik dosyalari bas — bir kosuda hepsini gormek icin.
    for want in DUMP:
        hit = None
        for n in names:
            if n.replace("\\", "/").endswith(want):
                hit = n
                break
        if not hit:
            continue
        path = os.path.join(args.dest, hit)
        if not os.path.isfile(path):
            continue
        print(f"\n{'='*70}\n=== {hit}\n{'='*70}")
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= DUMP_MAX_LINES:
                    print(f"    ... ({want} kirpildi, {DUMP_MAX_LINES} satir)")
                    break
                print("   ", line.rstrip())


if __name__ == "__main__":
    main()
