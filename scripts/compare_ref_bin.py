#!/usr/bin/env python3
"""
Referans (calisan) unet.bin ile bizimkini ALAN ALAN karsilastirir.

Neden: cihaz "Motor sureci beklenmedik sekilde kapandi (kod 1) / Could not free
context" diyor ama bu logun SON satiri; gercek hata gorunmuyor. Tipler dogru
(UFIXED_POINT_16 / INT_32), SDK surumunu 2.40 -> 2.39 cekmek de degistirmedi.
Tahmin etmeyi birakip calisan bir binary ile bizimkinin metadata'sini
karsilastiriyoruz: dsp_arch, VTCM, optimizasyon seviyesi, graf adi, spill-fill
boyutu, tensor sirasi/tipleri/kuantizasyon parametreleri...

Kullanim:
    python compare_ref_bin.py --ours work/X/qnn/unet.bin
    python compare_ref_bin.py --ours ... --ref-zip https://.../foo_min.zip
    python compare_ref_bin.py --ours ... --ref /yol/unet.bin
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HF_API = "https://huggingface.co/api/models/xororz/sd-qnn"
HF_RESOLVE = "https://huggingface.co/xororz/sd-qnn/resolve/main/{name}"


# --------------------------------------------------------------------------
# referans paketi bul / indir
# --------------------------------------------------------------------------
def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find_ref_zip() -> str:
    """HF deposundaki SD1.5 '_min' paketlerinden birini secer."""
    print(f"[*] referans paket araniyor: {HF_API}")
    data = json.loads(_get(HF_API).decode())
    files = [s["rfilename"] for s in data.get("siblings", [])]
    mins = [f for f in files
            if f.lower().endswith(".zip")
            and "_min" in f.lower()
            and "sdxl" not in f.lower()]
    if not mins:
        raise SystemExit(
            "HF deposunda '_min' zip bulunamadi. Elle verin:\n"
            "  --ref-zip <url>   ya da   --ref <unet.bin yolu>\n"
            f"  mevcut dosyalar: {files[:20]}")
    mins.sort(key=len)
    print(f"    {len(mins)} aday, secilen: {mins[0]}")
    return HF_RESOLVE.format(name=mins[0])


def download_unet(url: str, dest_dir: str) -> str:
    """Zip'i indirir, SADECE unet.bin'i cikarir, zip'i siler (disk dar)."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ref.zip")
    if not os.path.exists(zip_path):
        print(f"[*] indiriliyor: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(zip_path, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {done/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({pct:.0f}%)", end="", flush=True)
            print()
    out = os.path.join(dest_dir, "unet_ref.bin")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("unet.bin")]
        if not names:
            raise SystemExit(f"zip icinde unet.bin yok: {zf.namelist()[:20]}")
        with zf.open(names[0]) as src, open(out, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    os.remove(zip_path)
    print(f"[+] referans unet.bin -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")
    return out


# --------------------------------------------------------------------------
# metadata dokumu
# --------------------------------------------------------------------------
def dump_meta(bin_path: str) -> dict:
    root = os.environ.get("QNN_SDK_ROOT")
    if not root:
        raise SystemExit("QNN_SDK_ROOT ayarli degil")
    tool = os.path.join(root, "bin", "x86_64-linux-clang",
                        "qnn-context-binary-utility")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.join(root, "lib", "x86_64-linux-clang")
                              + ":" + env.get("LD_LIBRARY_PATH", ""))
    out = os.path.join(tempfile.mkdtemp(), "meta.json")
    subprocess.run([tool, "--context_binary", bin_path, "--json_file", out],
                   env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(out):
        raise SystemExit(f"metadata uretilemedi: {bin_path}")
    with open(out) as f:
        return json.load(f)


def flatten(obj, prefix="", out=None):
    """Ic ice JSON'u 'a.b.c' -> deger duz sozlugune cevirir. Listeler indeksli."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = obj
    return out


# Karsilastirmada anlamsiz olan alanlar (her binary'de zaten farkli).
NOISE = ("binaryhash", "timestamp", "size", "offset", "id", "buffersize")


def is_noise(key: str) -> bool:
    k = key.lower()
    return any(n in k for n in NOISE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="bizim unet.bin")
    ap.add_argument("--ref", help="referans unet.bin (yerel)")
    ap.add_argument("--ref-zip", help="referans paket .zip URL'si")
    ap.add_argument("--cache", default="work/_refcmp",
                    help="indirilen referansin saklanacagi klasor")
    ap.add_argument("--keep", action="store_true",
                    help="referans unet.bin'i silme (tekrar indirmemek icin)")
    args = ap.parse_args()

    ref_path = args.ref
    if not ref_path:
        cached = os.path.join(args.cache, "unet_ref.bin")
        if os.path.exists(cached):
            print(f"[*] onbellekten: {cached}")
            ref_path = cached
        else:
            ref_path = download_unet(args.ref_zip or find_ref_zip(), args.cache)

    print()
    print("=" * 74)
    print(" REFERANS (calisan)  <->  BIZIMKI")
    print("=" * 74)
    for label, p in (("referans", ref_path), ("bizim", args.ours)):
        print(f"  {label:9s} {p}  ({os.path.getsize(p)/1e6:.0f} MB)")
    print()

    a = flatten(dump_meta(ref_path))
    b = flatten(dump_meta(args.ours))

    keys = sorted(set(a) | set(b))
    same, diff, only_a, only_b = [], [], [], []
    for k in keys:
        if is_noise(k):
            continue
        if k in a and k in b:
            (same if a[k] == b[k] else diff).append(k)
        elif k in a:
            only_a.append(k)
        else:
            only_b.append(k)

    if diff:
        print(f"--- FARKLI ({len(diff)}) " + "-" * 50)
        for k in diff:
            print(f"  {k}")
            print(f"      referans: {a[k]!r}")
            print(f"      bizim   : {b[k]!r}")
        print()
    if only_a:
        print(f"--- SADECE REFERANSTA ({len(only_a)}) " + "-" * 38)
        for k in only_a:
            print(f"  {k} = {a[k]!r}")
        print()
    if only_b:
        print(f"--- SADECE BIZDE ({len(only_b)}) " + "-" * 42)
        for k in only_b:
            print(f"  {k} = {b[k]!r}")
        print()
    print(f"[=] ayni olan alan sayisi: {len(same)}")
    if not diff and not only_a and not only_b:
        print("[+] metadata birebir ayni — sorun binary disinda.")

    if not args.keep and not args.ref and os.path.exists(ref_path):
        os.remove(ref_path)
        print(f"[disk] {ref_path} silindi (--keep ile saklanir)")


if __name__ == "__main__":
    main()
