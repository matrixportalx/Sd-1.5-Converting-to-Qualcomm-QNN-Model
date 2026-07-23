#!/usr/bin/env python3
"""
Bir .safetensors modelini URL'den indirir (civitai / Hugging Face / düz link).

civitai indirme linkleri genelde token ister:
    export CIVITAI_TOKEN=xxxx
Hugging Face gated/private repo için:
    export HF_TOKEN=hf_xxxx

Kullanim:
    python fetch_model.py --url "https://civitai.com/api/download/models/128713" \
        --output work/model.safetensors
"""
import argparse
import os
import re
import sys
import urllib.parse

import requests


def _filename_from_headers(resp, fallback):
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    if m:
        return m.group(1)
    return fallback


def download(url: str, output: str) -> str:
    headers = {"User-Agent": "sd-qnn-converter/1.0"}
    host = urllib.parse.urlparse(url).netloc.lower()

    # Token ekleme
    if "civitai" in host:
        tok = os.environ.get("CIVITAI_TOKEN")
        if tok:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={tok}"
    elif "huggingface" in host:
        tok = os.environ.get("HF_TOKEN")
        if tok:
            headers["Authorization"] = f"Bearer {tok}"

    print(f"[*] İndiriliyor: {url.split('?')[0]}")
    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 401 or r.status_code == 403:
            sys.exit(f"HATA {r.status_code}: Yetki gerekli. "
                     f"CIVITAI_TOKEN / HF_TOKEN ayarladınız mı?")
        r.raise_for_status()

        # Cikti bir klasorse dosya adini header/URL'den turet
        if output.endswith("/") or os.path.isdir(output):
            base = _filename_from_headers(
                r, os.path.basename(urllib.parse.urlparse(url).path)
                or "model.safetensors")
            os.makedirs(output, exist_ok=True)
            output = os.path.join(output, base)
        else:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(output, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r    {done>>20} / {total>>20} MB ({pct}%)",
                          end="", flush=True)
        print()

    if not output.lower().endswith(".safetensors"):
        print(f"[!] Uyarı: dosya .safetensors uzantılı değil ({output}). "
              f"Yine de devam ediliyor.")
    print(f"[+] Kaydedildi: {output}  ({os.path.getsize(output)>>20} MB)")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--output", default="work/model.safetensors")
    args = ap.parse_args()
    path = download(args.url, args.output)
    # convert_all.sh'in kullanabilmesi icin yolu stdout'a da yaz
    print(f"MODEL_PATH={path}")


if __name__ == "__main__":
    main()
