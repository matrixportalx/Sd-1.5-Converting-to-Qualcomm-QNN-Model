#!/usr/bin/env python3
"""
QAIRT / QNN SDK'yi bir GitHub release'inden otomatik indirir ve acar,
ardindan QNN_SDK_ROOT yolunu bulur.

Varsayilan: matrixportalx/qairt-sdk  v2.39.0.250926 (public release).
Public ise token gerekmez. Private ise --token (GitHub PAT) verin.

Kullanim:
    python setup_qnn_sdk.py --dest /content/qairt
    # veya farkli surum/asset:
    python setup_qnn_sdk.py --repo matrixportalx/qairt-sdk \
        --tag v2.39.0.250926 --dest /content/qairt

Cikti: son satirda  QNN_SDK_ROOT=<yol>  yazar (kabuk/CI yakalayabilir).
"""
import argparse
import os
import sys
import tarfile
import urllib.parse
import zipfile

import requests

ARCHIVE_EXTS = (".zip", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar")


def _api_get(url, token):
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "sd-qnn-setup/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def resolve_asset(repo, tag, token):
    """Release'teki arsiv asset'inin (name, download_url) ciftini dondurur."""
    rel = _api_get(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}", token)
    assets = rel.get("assets", [])
    if not assets:
        sys.exit(f"HATA: release'te asset yok: {repo}@{tag}")
    for a in assets:
        name = a["name"].lower()
        if name.endswith(ARCHIVE_EXTS):
            return a["name"], a["browser_download_url"]
    # Arsiv bulunamazsa ilk asset'i dene
    return assets[0]["name"], assets[0]["browser_download_url"]


def download(url, out_path, token):
    headers = {"User-Agent": "sd-qnn-setup/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"[*] SDK indiriliyor: {url}")
    with requests.get(url, headers=headers, stream=True, timeout=120,
                      allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done>>20}/{total>>20} MB "
                          f"({done*100//total}%)", end="", flush=True)
        print()


def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"[*] Aciliyor -> {dest}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def find_sdk_root(base):
    """Icinde bin/x86_64-linux-clang/{qairt-converter|qnn-onnx-converter}
    olan klasoru QNN_SDK_ROOT olarak dondurur."""
    for d, _, _ in os.walk(base):
        bindir = os.path.join(d, "bin", "x86_64-linux-clang")
        if os.path.isdir(bindir):
            for tool in ("qairt-converter", "qnn-onnx-converter"):
                if os.path.exists(os.path.join(bindir, tool)):
                    return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="matrixportalx/qairt-sdk")
    ap.add_argument("--tag", default="v2.39.0.250926")
    ap.add_argument("--dest", default="qairt")
    ap.add_argument("--asset-url", default=None,
                    help="Dogrudan indirme linki (API'yi atlar)")
    ap.add_argument("--token", default=os.environ.get("GH_TOKEN")
                    or os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--github-env", action="store_true",
                    help="QNN_SDK_ROOT'u $GITHUB_ENV'e de yaz (Actions)")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    if args.asset_url:
        name = os.path.basename(urllib.parse.urlparse(args.asset_url).path)
        url = args.asset_url
    else:
        name, url = resolve_asset(args.repo, args.tag, args.token)

    archive = os.path.join(args.dest, name)
    if not os.path.exists(archive):
        download(url, archive, args.token)
    else:
        print(f"[*] Arsiv zaten var: {archive}")

    extract(archive, args.dest)
    root = find_sdk_root(args.dest)
    if not root:
        sys.exit("HATA: QNN_SDK_ROOT bulunamadi "
                 "(bin/x86_64-linux-clang/ icinde converter yok).")

    print(f"[+] QNN_SDK_ROOT = {root}")
    if args.github_env and os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"QNN_SDK_ROOT={root}\n")
    # Kabuk yakalayabilsin diye son satir:
    print(f"QNN_SDK_ROOT={root}")


if __name__ == "__main__":
    main()
