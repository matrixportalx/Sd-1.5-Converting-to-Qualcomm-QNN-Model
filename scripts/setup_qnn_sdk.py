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
import time
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


def download(url, out_path, token, attempts=5):
    """SDK'yi indirir. KALDIGI YERDEN DEVAM eder ve koparsa tekrar dener.

    Neden: Colab'da her yeni calisma zamani SDK'yi bastan indirmek demek
    (1.3-2 GB) ve baglanti ortasinda kopabiliyor ("%53'te" gibi). Tek bir
    kopma tum adimi bastan aldirmasin diye HTTP Range ile devam ediyoruz.
    """
    print(f"[*] SDK indiriliyor: {url}")
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        headers = {"User-Agent": "sd-qnn-setup/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120,
                              allow_redirects=True) as r:
                if have and r.status_code == 200:
                    # Sunucu Range'i yok saydi -> bastan yaz
                    have = 0
                elif have and r.status_code != 206:
                    r.raise_for_status()
                else:
                    r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + have
                done = have
                mode = "ab" if have else "wb"
                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r    {done>>20}/{total>>20} MB "
                                  f"({done*100//total}%)", end="", flush=True)
                print()
                if total and done < total:
                    raise IOError(f"eksik indirme: {done}/{total}")
                return
        except Exception as e:
            got = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            print(f"\n  [!] indirme koptu ({type(e).__name__}: {e}) — "
                  f"{got>>20} MB alindi, deneme {attempt}/{attempts}")
            if attempt == attempts:
                raise
            time.sleep(min(2 ** attempt, 16))


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


def make_bins_executable(sdk_root):
    """ZIP acilinca kaybolan calistirma bitlerini geri ver ('bin/' altindaki
    tum dosyalara +x). QAIRT/QNN araclari aksi halde 'Permission denied' verir."""
    import stat
    count = 0
    for d, _, files in os.walk(sdk_root):
        if "bin" not in d.split(os.sep):
            continue
        for fn in files:
            p = os.path.join(d, fn)
            try:
                st = os.stat(p)
                os.chmod(p, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                count += 1
            except OSError:
                pass
    print(f"[*] {count} dosyaya calistirma izni verildi (bin/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="matrixportalx/qairt-sdk")
    ap.add_argument("--tag", default="v2.39.0.250926")
    ap.add_argument("--dest", default="qairt")
    ap.add_argument("--cache-dir", default=os.environ.get("QAIRT_CACHE"),
                    help="Arsivin saklanacagi kalici klasor (or. Drive). "
                         "Boylece her oturumda yeniden indirilmez.")
    ap.add_argument("--asset-url", default=None,
                    help="Dogrudan indirme linki (API'yi atlar)")
    ap.add_argument("--token", default=os.environ.get("GH_TOKEN")
                    or os.environ.get("GITHUB_TOKEN"))
    args = ap.parse_args()

    # config.env -> OVERRIDE_QAIRT_ASSET_URL / OVERRIDE_QAIRT_TAG
    # Not defterine dokunmadan SDK surumu degistirebilmek icin (bkz. config.env).
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config.env")
    if os.path.exists(cfg):
        for line in open(cfg):
            line = line.strip()
            if not line.startswith("OVERRIDE_QAIRT_"):
                continue
            key, _, val = line.partition("=")
            val = val.split("#")[0].strip()
            if not val:
                continue
            if key == "OVERRIDE_QAIRT_ASSET_URL":
                args.asset_url = val
                print(f"[config.env] asset_url = {val}")
            elif key == "OVERRIDE_QAIRT_TAG":
                args.tag = val
                print(f"[config.env] tag = {val}")
            elif key == "OVERRIDE_QAIRT_REPO":
                args.repo = val
                print(f"[config.env] repo = {val}")

    # Surumler AYRI dizinlere acilir; yoksa 2.39 uzerine yazip karisir ve
    # "zaten acik" kontrolu eski surumu dondurur.
    if args.asset_url:
        key = os.path.basename(urllib.parse.urlparse(args.asset_url).path)
    else:
        key = args.tag or "default"
    for ext in ARCHIVE_EXTS:
        if key.endswith(ext):
            key = key[: -len(ext)]
            break
    args.dest = os.path.join(args.dest, key.lstrip("v"))
    os.makedirs(args.dest, exist_ok=True)
    print(f"[*] SDK dizini: {args.dest}")

    # SDK zaten acilmissa hic dokunma (yeniden acmak dakikalar suruyor)
    existing = find_sdk_root(args.dest)
    if existing:
        print(f"[*] SDK zaten acik: {existing}")
        make_bins_executable(existing)
        print(f"[+] QNN_SDK_ROOT = {existing}")
        print(f"QNN_SDK_ROOT={existing}")
        return

    if args.asset_url:
        name = os.path.basename(urllib.parse.urlparse(args.asset_url).path)
        url = args.asset_url
    else:
        name, url = resolve_asset(args.repo, args.tag, args.token)

    # Arsivi kalici cache'te tut (Drive); acma her zaman YEREL diske yapilir —
    # Drive uzerinden .so/binary calistirmak izin ve hiz sorunlari cikariyor.
    cache = args.cache_dir or args.dest
    os.makedirs(cache, exist_ok=True)
    archive = os.path.join(cache, name)
    if not os.path.exists(archive):
        try:
            download(url, archive, args.token)
        except Exception as e:
            # Indirme basarisizsa (or. 404) sessizce olme: hangi surumlerin
            # indirilebildigini BURADA goster ki config.env tek satirla
            # duzeltilebilsin.
            if os.path.exists(archive):
                os.remove(archive)
            print(f"\n[!] Indirilemedi: {url}\n    ({type(e).__name__}: {e})\n")
            sys.exit("HATA: SDK indirilemedi. Calisan bir surum URL'sini "
                     "config.env icindeki OVERRIDE_QAIRT_ASSET_URL satirina "
                     "yazip 4. adimi tekrar calistirin.")
    else:
        print(f"[*] Arsiv onbellekten: {archive} "
              f"({os.path.getsize(archive)>>20} MB)")

    extract(archive, args.dest)
    root = find_sdk_root(args.dest)
    if not root:
        sys.exit("HATA: QNN_SDK_ROOT bulunamadi "
                 "(bin/x86_64-linux-clang/ icinde converter yok).")

    make_bins_executable(root)
    print(f"[+] QNN_SDK_ROOT = {root}")
    # Kabuk yakalayabilsin diye son satir:
    print(f"QNN_SDK_ROOT={root}")


if __name__ == "__main__":
    main()
