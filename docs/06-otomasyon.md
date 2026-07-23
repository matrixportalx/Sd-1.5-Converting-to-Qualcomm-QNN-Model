# 06 — Otomasyon: Colab Notebook ve GitHub Actions

"Link gir → dönüştür → Hugging Face'e yükle" akışı için iki yol var.

---

## A) Google Colab (önerilen) ✅

**Dosya:** [`notebooks/SD15_to_QNN_Colab.ipynb`](../notebooks/SD15_to_QNN_Colab.ipynb)

Colab'da açmak için depoyu GitHub'a yükledikten sonra:
`https://colab.research.google.com/github/matrixportalx/Sd-1.5-Converting-to-Qualcomm-QNN-Model/blob/claude/qnn-model-conversion-snapdragon7-rsk8og/notebooks/SD15_to_QNN_Colab.ipynb`

### Akış
1. **Runtime → Change runtime type → High-RAM** seç.
2. 1. hücrede formu doldur: `SAFETENSORS_URL`, `MODEL_NAME`, `TIER=min`, `RESOLUTIONS`, `HF_REPO`.
3. Colab **Secrets** (🔑) içine ekle: `HF_TOKEN` (write), gerekiyorsa `CIVITAI_TOKEN`.
4. Hücreleri sırayla çalıştır.

### SDK otomatik — manuel adım yok
QAIRT SDK **`matrixportalx/qairt-sdk` v2.39.0.250926** release'inden otomatik
indirilir (`scripts/setup_qnn_sdk.py`). Release **public** olduğu için token bile
gerekmez; Qualcomm'dan indirme veya Drive'a yükleme **yok**. (Release'i private
yaparsan Colab Secrets'a `GH_TOKEN` ekle.)

### Kısıtlar
- **RAM:** 512px için ~20 GB gerekir. Ücretsiz Colab (12 GB) OOM olabilir; notebook
  16 GB swap ekleyerek yardımcı olur ama **Colab Pro / High-RAM** çok daha güvenli.
- **Süre:** Çözünürlük başına saatler. Sekmeyi kapatma; Colab boşta kalırsa oturum düşer.
- Birden fazla çözünürlük istiyorsan (512x768, 768x512) süre ve RAM ihtiyacı artar.

---

## B) GitHub Actions — yalnızca **self-hosted runner** ⚠️

**Dosya:** [`.github/workflows/convert.yml`](../.github/workflows/convert.yml)

> **Ücretsiz (github-hosted) runner'lar ÇALIŞMAZ.** Tek sebep artık **RAM**:
> ~7 GB var, 20 GB+ gerekli. (SDK otomatik indirildiği için lisans sorunu yok.)

SDK'yı workflow zaten release'ten otomatik indirir; sadece yeterli RAM'li bir
runner gerekir:

1. **Depo → Settings → Actions → Runners → New self-hosted runner** (kendi 32 GB+
   Linux makineniz). *Alternatif:* GitHub "larger runner" (ücretli) — `runs-on`'u
   değiştirin.
2. **Depo Secrets:** `HF_TOKEN` (write), gerekiyorsa `CIVITAI_TOKEN`.
   (qairt-sdk release public ise `GH_TOKEN` gerekmez.)
3. **Actions → "Convert SD1.5 -> QNN" → Run workflow** → linki, model adını,
   `qnn_version` (varsayılan 2.39) girin.

Çıktı hem **artifact** olarak indirilebilir hem de `hf_repo` doldurulduysa HF'ye yüklenir.

---

## Hangisini seçmeliyim?

| Durum | Öneri |
|---|---|
| Kendi güçlü PC'niz yok | **Colab (Pro / High-RAM)** |
| Elinizde 32 GB+ RAM'li Linux makine var | **Self-hosted Actions** (tekrarlı işler için pratik) |
| Sadece tek seferlik deneme | Colab |

Her iki yol da aynı `convert_all.sh` hattını ve aynı `fetch_model.py` / `upload_hf.py`
yardımcılarını kullanır; sonuç birebir aynıdır.
