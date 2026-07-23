#!/usr/bin/env python3
"""
Üretilen ZIP'i Hugging Face'e yükler.

İki mod:
  * --repo kullanici/repo   : bu repoya yükler.
  * --name ModelAdi         : repo adini model isminden turetir ->
                              <senin_kullanici_adin>/ModelAdi (yoksa olusturur).

Gerekli: yazma (write) izinli HF token -> export HF_TOKEN=hf_xxxx

Kullanim:
    python upload_hf.py --name CyberRealisticLCM \
        --file dist/CyberRealisticLCM_qnn2.39_min.zip
"""
import argparse
import os
import sys

# soc_targets'i CWD'den bagimsiz import edebilmek icin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


MODEL_CARD = """---
license: creativeml-openrail-m
tags:
  - stable-diffusion
  - sd1.5
  - qualcomm
  - qnn
  - qairt
  - snapdragon
  - local-dream
  - text-to-image
---

# {name} — SD1.5 → Qualcomm QNN ({runtime})

Snapdragon NPU üzerinde **Ruya / Local Dream** uygulamasında çalışacak şekilde
dönüştürülmüş Stable Diffusion 1.5 modeli.

- **Taban:** SD 1.5
- **Runtime:** {runtime}  (tier: `{tier}`, HTP `{dsp_arch}`, {act}-bit aktivasyon)
- **Çözünürlük(ler):** {resolutions}
- **UNet:** QNN context binary (NPU) · **text_encoder/VAE:** MNN (CPU/GPU)

## Kurulum
1. `{zip_name}` dosyasını telefona indirin.
2. **Ruya / Local Dream → Settings → Import Custom Model** ile içe aktarın.

> `{tier}` (HTP `{dsp_arch}`) varyanti: {tier_desc}

---
_[Sd-1.5-Converting-to-Qualcomm-QNN-Model](https://github.com/matrixportalx/Sd-1.5-Converting-to-Qualcomm-QNN-Model) ile üretildi._
"""


def _read_model_info(zip_path):
    """ZIP icindeki model_info.json'dan meta veri okumaya calisir."""
    import json
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for n in zf.namelist():
                if n.endswith("model_info.json"):
                    return json.loads(zf.read(n))
    except Exception:
        pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="Hedef repo: kullanici/repo-adi")
    ap.add_argument("--name", default=None,
                    help="Repo adini model isminden turet: <kullanici>/<name>")
    ap.add_argument("--file", required=True, help="Yüklenecek dosya (zip)")
    ap.add_argument("--path-in-repo", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-card", action="store_true",
                    help="Model karti (README.md) olusturma/yukleme")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if not args.token:
        sys.exit("HATA: HF_TOKEN yok. Yazma izinli token gerekli "
                 "(Colab Secrets -> HF_TOKEN).")
    if not os.path.exists(args.file):
        sys.exit(f"HATA: dosya bulunamadı: {args.file}")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    # Repo adini belirle
    repo = args.repo
    if not repo:
        if not args.name:
            sys.exit("HATA: --repo veya --name verilmeli.")
        who = api.whoami()
        user = who.get("name") or who.get("email", "").split("@")[0]
        if not user:
            sys.exit("HATA: HF kullanici adi alinamadi (token gecerli mi?).")
        repo = f"{user}/{args.name}"
        print(f"[*] Repo adi model isminden turetildi: {repo}")

    print(f"[*] Repo hazırlanıyor: {repo}  (private={args.private})")
    api.create_repo(repo_id=repo, repo_type="model",
                    private=args.private, exist_ok=True)

    # Model karti (README.md)
    if not args.no_card:
        info = _read_model_info(args.file)
        try:
            from soc_targets import TIERS
        except Exception:
            TIERS = {}
        tier = info.get("tier", "min")
        tinfo = TIERS.get(tier, {})
        card = MODEL_CARD.format(
            name=args.name or repo.split("/")[-1],
            runtime=info.get("runtime", "qnn2.39"),
            tier=tier,
            dsp_arch=info.get("dsp_arch", tinfo.get("dsp_arch", "v69")),
            act="8" if tier == "min" else "16",
            resolutions=", ".join(info.get("resolutions", ["512x512"])),
            zip_name=os.path.basename(args.file),
            tier_desc=tinfo.get("desc", ""),
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo, repo_type="model",
            commit_message="Add model card",
        )
        print("[*] Model karti (README.md) yuklendi")

    path_in_repo = args.path_in_repo or os.path.basename(args.file)
    print(f"[*] Yükleniyor: {args.file} -> {repo}/{path_in_repo} "
          f"({os.path.getsize(args.file)>>20} MB)")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
        commit_message="Add QNN converted SD1.5 model",
    )
    url = f"https://huggingface.co/{repo}/blob/main/{path_in_repo}"
    print(f"[+] Yüklendi: {url}")
    print(f"[+] Repo: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
