#!/usr/bin/env python3
"""
Üretilen ZIP'i Hugging Face'e yükler. Üç kullanım:

  MOD A — her model AYRI repo (dosyalar kökte):
    python upload_hf.py --name CyberRealisticLCM --file dist/..._min.zip
    -> <kullanici>/CyberRealisticLCM

  MOD B — hepsi TEK koleksiyon reposunda (her model alt klasörde):
    python upload_hf.py --collection sd_qnn --name CyberRealisticLCM --file ...
    -> <kullanici>/sd_qnn/CyberRealisticLCM/...

  Dogrudan tam repo adi:
    python upload_hf.py --repo kullanici/istedigin-repo --file ...

Gerekli: yazma (write) izinli HF token -> export HF_TOKEN=hf_xxxx
(--name verilmezse model adi zip dosya isminden turetilir.)
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


def _model_name_from_zip(zip_path: str) -> str:
    b = os.path.basename(zip_path)
    if "_qnn" in b:
        return b.split("_qnn")[0]
    return os.path.splitext(b)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="ZIP'i Hugging Face'e yukler")
    ap.add_argument("--file", required=True, help="Yüklenecek dosya (zip)")
    # --- Yukleme modu (birini secin) ---
    ap.add_argument("--repo", default=None,
                    help="MOD A/B: Tam repo adi 'kullanici/repo' (dogrudan bu repoya)")
    ap.add_argument("--name", default=None,
                    help="MOD A (ayri repo): repo = <kullanici>/<name>, dosyalar kokte")
    ap.add_argument("--collection", default=None,
                    help="MOD B (koleksiyon): tek repo (or. 'sd_qnn'); her model "
                         "kendi alt klasorunde. '/' yoksa <kullanici>/<collection>")
    # ---
    ap.add_argument("--subdir", default=None,
                    help="Repo icindeki alt klasor (koleksiyonda otomatik = model adi)")
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

    model_name = args.name or _model_name_from_zip(args.file)

    def _user():
        who = api.whoami()
        u = who.get("name") or who.get("email", "").split("@")[0]
        if not u:
            sys.exit("HATA: HF kullanici adi alinamadi (token gecerli mi?).")
        return u

    # --- Repo ve alt klasoru belirle ---
    subdir = (args.subdir or "").strip("/")
    if args.repo:
        repo = args.repo                      # dogrudan verilen repo
    elif args.collection:
        # MOD B: koleksiyon reposu; her model alt klasorde
        coll = args.collection
        repo = coll if "/" in coll else f"{_user()}/{coll}"
        if not subdir:
            subdir = model_name
        print(f"[*] Koleksiyon modu: {repo}  (alt klasor: {subdir})")
    elif args.name:
        # MOD A: model adindan ayri repo
        repo = f"{_user()}/{args.name}"
        print(f"[*] Ayri repo modu: {repo}")
    else:
        sys.exit("HATA: --repo, --name veya --collection verilmeli.")

    prefix = f"{subdir}/" if subdir else ""

    print(f"[*] Repo hazırlanıyor: {repo}  (private={args.private})")
    api.create_repo(repo_id=repo, repo_type="model",
                    private=args.private, exist_ok=True)

    # Model karti: kokte (ayri repo) veya alt klasorde (koleksiyon) —
    # koleksiyonda ana README'yi ezmeyiz.
    if not args.no_card:
        info = _read_model_info(args.file)
        try:
            from soc_targets import TIERS
        except Exception:
            TIERS = {}
        tier = info.get("tier", "min")
        tinfo = TIERS.get(tier, {})
        card = MODEL_CARD.format(
            name=model_name,
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
            path_in_repo=f"{prefix}README.md",
            repo_id=repo, repo_type="model",
            commit_message=f"Add model card ({model_name})",
        )
        print(f"[*] Model karti yuklendi -> {prefix}README.md")

    path_in_repo = f"{prefix}{os.path.basename(args.file)}"
    print(f"[*] Yükleniyor: {args.file} -> {repo}/{path_in_repo} "
          f"({os.path.getsize(args.file)>>20} MB)")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
        commit_message=f"Add {model_name} (QNN SD1.5)",
    )
    print(f"[+] Yüklendi: https://huggingface.co/{repo}/blob/main/{path_in_repo}")
    print(f"[+] Repo: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
