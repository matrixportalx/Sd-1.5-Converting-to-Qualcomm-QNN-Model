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
import re
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


# Resmi hattin (npuconvertv2) SOC adlari ile HTP mimarileri. Degerler paketle
# gelen htp_config_<soc>.json dosyalarindan alinmistir — TIERS tablosu eski
# hattin (min/mid/high) adlandirmasidir ve bunlarla ortusmez.
OFFICIAL_SOCS = {
    "min":   {"dsp_arch": "v68",
              "desc": "SD1.5 destekleyen tum cihazlar (Snapdragon 7 serisi dahil)"},
    "8gen1": {"dsp_arch": "v69",
              "desc": "Snapdragon 8 Gen 1, 7 Gen 1, 7s Gen 2"},
    "8gen2": {"dsp_arch": "v73",
              "desc": "Snapdragon 8 Gen 2, 8s Gen 3, 7+ Gen 2, 7 Gen 3"},
    "8gen3": {"dsp_arch": "v75",
              "desc": "Snapdragon 8 Gen 3 ve uzeri"},
}


def _meta_from_filename(zip_path):
    """'<Ad>_qnn2.28_8gen2.zip' -> {'runtime': 'qnn2.28', 'tier': '8gen2', ...}

    RESMI hat (npuconvertv2) paketi model_info.json icermez — dosyalar duz
    durur. Bu yuzden meta veriyi dosya adindan cikariyoruz. Ekteki SOC adi
    paketin DERLENDIGI HTP mimarisini belirler; eski surum 8gen1/8gen2'yi de
    'min' (v68) sayip model kartina yanlis mimari yaziyordu.
    """
    b = os.path.basename(zip_path)
    m = re.search(r"_(qnn[0-9.]+)(?:_(min|8gen1|8gen2|8gen3))?\.zip$", b)
    if not m:
        return {}
    tail = m.group(2) or "mid"          # eksik ek = eski hattin 'mid' tier'i
    info = {"runtime": m.group(1), "tier": tail}
    info.update(OFFICIAL_SOCS.get(tail, {}))
    return info


def _resolutions_from_zip(zip_path):
    """Pakette hangi cozunurluklerin secilebilecegini YAMA DOSYALARINDAN cikarir.

    QNN'de cozunurluk derleme zamani ozelligidir: unet.bin 512x512 icindir,
    diger boyutlar yaninda duran zstd yamalaridir. Uygulama (Ruya / Local
    Dream) da listeyi tam olarak boyle tariyor:
        768.patch      -> 768x768   (kare yamalar tek sayiyla)
        512x768.patch  -> 512x768   (dikdortgen yamalar WxH)
    """
    import zipfile
    square = re.compile(r"^(\d+)\.patch$")
    rect = re.compile(r"^(\d+)x(\d+)\.patch$")
    found = [(512, 512)]
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for n in zf.namelist():
                base = os.path.basename(n)
                m = square.match(base)
                if m:
                    found.append((int(m.group(1)), int(m.group(1))))
                    continue
                m = rect.match(base)
                if m:
                    found.append((int(m.group(1)), int(m.group(2))))
    except Exception:
        pass
    seen, out = set(), []
    for w, h in sorted(found, key=lambda t: t[0] * t[1]):
        if (w, h) not in seen:
            seen.add((w, h))
            out.append(f"{w}x{h}")
    return out


def _read_model_info(zip_path):
    """ZIP icindeki model_info.json'dan meta veri okumaya calisir;
    yoksa dosya adindan cikarir. Cozunurluk listesi her iki durumda da
    paketin kendi yama dosyalarindan gelir (model_info.json'da yok)."""
    import json
    import zipfile
    info = None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for n in zf.namelist():
                if n.endswith("model_info.json"):
                    info = json.loads(zf.read(n))
                    break
    except Exception:
        pass
    if info is None:
        info = _meta_from_filename(zip_path)
    info.setdefault("resolutions", _resolutions_from_zip(zip_path))
    return info


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
        tinfo = OFFICIAL_SOCS.get(tier) or TIERS.get(tier, {})
        card = MODEL_CARD.format(
            name=model_name,
            runtime=info.get("runtime", "qnn2.28"),
            tier=tier,
            dsp_arch=info.get("dsp_arch", tinfo.get("dsp_arch", "v69")),
            # Resmi tarif (npuconvertv2) her tier icin --act_bitwidth 16
            act=str(info.get("act_bitwidth", 16)),
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
