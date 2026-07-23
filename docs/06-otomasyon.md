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

### Zorunlu tek manuel adım: QNN SDK
QNN SDK 2.28 **Qualcomm lisansı** yüzünden otomatik indirilemez. Bir kez:
1. Qualcomm AI Hub / QPM'den `v2.28.0.241029` indir.
2. Google Drive'a `.zip` olarak yükle (ör. `MyDrive/qnn/v2.28.0.241029.zip`).
3. Notebook'ta yolunu `QNN_SDK_ZIP_ON_DRIVE` alanına yaz — notebook mount edip açar.

### Kısıtlar
- **RAM:** 512px için ~20 GB gerekir. Ücretsiz Colab (12 GB) OOM olabilir; notebook
  16 GB swap ekleyerek yardımcı olur ama **Colab Pro / High-RAM** çok daha güvenli.
- **Süre:** Çözünürlük başına saatler. Sekmeyi kapatma; Colab boşta kalırsa oturum düşer.
- Birden fazla çözünürlük istiyorsan (512x768, 768x512) süre ve RAM ihtiyacı artar.

---

## B) GitHub Actions — yalnızca **self-hosted runner** ⚠️

**Dosya:** [`.github/workflows/convert.yml`](../.github/workflows/convert.yml)

> **Ücretsiz (github-hosted) runner'lar ÇALIŞMAZ.** Sebep: ~7 GB RAM (20 GB+ gerekli)
> ve QNN SDK'nın lisanslı olup otomatik indirilememesi.

Kendi güçlü Linux makineniz varsa self-hosted runner olarak bağlayın:

1. **Depo → Settings → Actions → Runners → New self-hosted runner** adımlarını izleyin.
2. Makinede QNN SDK 2.28'i açın ve runner'a kalıcı env verin:
   ```
   QNN_SDK_ROOT=/opt/qairt/2.28.0.241029
   ```
3. `pip install MNN` (mnnconvert komutu).
4. **Depo Secrets:** `HF_TOKEN` (write), gerekiyorsa `CIVITAI_TOKEN`.
5. **Actions → "Convert SD1.5 -> QNN" → Run workflow** → linki ve model adını girin.

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
