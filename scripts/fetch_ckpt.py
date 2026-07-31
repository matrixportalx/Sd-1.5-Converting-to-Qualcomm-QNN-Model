#!/usr/bin/env python3
"""
.safetensors kontrol noktasini indirir — civitai / Hugging Face / duz link.

Neden ayri script: Colab hucresine uzun kod yapistirmak mobilde bozuluyor.
Hucre yalnizca token'lari ortam degiskenine koyup bu scripti cagirir.

Token:
    CIVITAI_TOKEN=...   civitai /api/download/... uclari tokensiz 401 verir
    HF_TOKEN=hf_...     gated/private HF repolari icin

Kullanim:
    CIVITAI_TOKEN=xxx python scripts/fetch_ckpt.py \
        --url "https://civitai.com/api/download/models/123?fileId=456" \
        --out work/input.safetensors
"""
import argparse
import os
import sys
import urllib.parse

import requests

CHUNK = 1 << 20


def _is_html(head: bytes) -> bool:
    h = head[:400].lower()
    return b"<html" in h or b"<!doctype" in h


def fetch(url: str, out: str, min_size: int) -> None:
    if os.path.exists(out) and os.path.getsize(out) >= min_size:
        print(f"[ATLANDI] {out} zaten var "
              f"({os.path.getsize(out)/1e9:.2f} GB)")
        return

    host = urllib.parse.urlparse(url).netloc.lower()
    headers = {"User-Agent": "sd-qnn/1.0"}

    if "civitai" in host:
        tok = os.environ.get("CIVITAI_TOKEN", "").strip()
        if not tok:
            sys.exit(
                "HATA: civitai token yok — tokensiz 401 Unauthorized doner.\n"
                "  civitai.com -> Account settings -> API Keys -> Add API key\n"
                "  Colab: Secrets (anahtar simgesi) -> CIVITAI_TOKEN")
        url += ("&" if "?" in url else "?") + "token=" + tok
    elif "huggingface" in host:
        tok = os.environ.get("HF_TOKEN", "").strip()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"

    # Yarim dosyadan devam (Range). Sunucu desteklemezse bastan.
    have = os.path.getsize(out) if os.path.exists(out) else 0
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"[*] {have/1e6:.0f} MB var, kaldigi yerden devam")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # URL'yi token'siz bas — defter ciktisina sizmasin
    print(f"[*] indiriliyor: {url.split('?')[0]}")

    with requests.get(url, headers=headers, stream=True, timeout=120,
                      allow_redirects=True) as r:
        if r.status_code in (401, 403):
            sys.exit(f"HATA {r.status_code}: yetki reddedildi. Token yanlis, "
                     f"suresi dolmus ya da model giris istiyor.")
        if r.status_code == 404:
            sys.exit("HATA 404: link bulunamadi. civitai'de 'dogrudan indirme' "
                     "linki /api/download/models/<id> seklinde olmali.")
        r.raise_for_status()

        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            sys.exit("HATA: sunucu HTML dondurdu (model degil). Link muhtemelen "
                     "model SAYFASI; 'Download' butonunun linkini kullanin.")

        resumed = r.status_code == 206
        if have and not resumed:
            have = 0                      # sunucu Range desteklemedi
        mode = "ab" if resumed else "wb"
        total = int(r.headers.get("content-length", 0)) + (have if resumed else 0)

        done = have if resumed else 0
        last = -1
        with open(out, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                if done == 0 and _is_html(chunk):
                    f.close()
                    os.remove(out)
                    sys.exit("HATA: indirilen icerik HTML — model degil.")
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    if pct != last:       # her %1'de bir satir (log sismesin)
                        last = pct
                        print(f"\r    {done>>20} / {total>>20} MB ({pct}%)",
                              end="", flush=True)
        print()

    size = os.path.getsize(out)
    if size < min_size:
        os.remove(out)
        sys.exit(f"HATA: dosya cok kucuk ({size} bayt) — indirme eksik/bozuk, "
                 f"silindi. Tekrar deneyin.")
    with open(out, "rb") as f:
        if _is_html(f.read(400)):
            os.remove(out)
            sys.exit("HATA: HTML indi, model degil — silindi.")
    print(f"[+] {size/1e9:.2f} GB -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SAFETENSORS_URL", ""))
    ap.add_argument("--out", default="work/input.safetensors")
    ap.add_argument("--min-size", type=int, default=100_000_000)
    a = ap.parse_args()
    if not a.url.strip():
        sys.exit("HATA: --url (veya SAFETENSORS_URL) bos — 1. hucreyi doldurun.")
    fetch(a.url.strip(), a.out, a.min_size)


if __name__ == "__main__":
    main()
