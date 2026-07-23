#!/usr/bin/env python3
"""
Üretilen ZIP'i (ve istenirse klasörü) Hugging Face repoya yükler.

Gerekli: yazma (write) izinli HF token.
    export HF_TOKEN=hf_xxxx

Kullanim:
    python upload_hf.py --repo kullanici/AbsoluteReality-SD1.5-qnn2.28 \
        --file dist/AbsoluteReality_qnn2.28_min.zip
"""
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="Hedef repo: kullanici/repo-adi")
    ap.add_argument("--file", required=True, help="Yüklenecek dosya (zip)")
    ap.add_argument("--path-in-repo", default=None,
                    help="Repo içindeki hedef yol (varsayılan: dosya adı)")
    ap.add_argument("--private", action="store_true",
                    help="Repo yoksa private olarak oluştur")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if not args.token:
        sys.exit("HATA: HF_TOKEN yok. Yazma izinli token gerekli "
                 "(export HF_TOKEN=hf_...).")
    if not os.path.exists(args.file):
        sys.exit(f"HATA: dosya bulunamadı: {args.file}")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    print(f"[*] Repo hazırlanıyor: {args.repo}")
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    path_in_repo = args.path_in_repo or os.path.basename(args.file)
    print(f"[*] Yükleniyor: {args.file} -> {args.repo}/{path_in_repo} "
          f"({os.path.getsize(args.file)>>20} MB)")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add QNN converted SD1.5 model",
    )
    url = f"https://huggingface.co/{args.repo}/blob/main/{path_in_repo}"
    print(f"[+] Yüklendi: {url}")


if __name__ == "__main__":
    main()
