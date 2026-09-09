# 06 — Otomasyon: Colab Notebook ve GitHub Actions

"Link gir → dönüştür → Hugging Face'e yükle" akışı için iki yol var.

---

## A) Google Colab (önerilen) ✅

**Dosya:** [`notebook/SD15_NPU.ipynb`](../notebook/SD15_NPU.ipynb)
(kökteki `SD15_NPU_Official_Colab.ipynb` aynı defterin kopyasıdır.)

Colab bağlantısı:
`https://colab.research.google.com/github/matrixportalx/Sd-1.5-Converting-to-Qualcomm-QNN-Model/blob/dev/notebook/SD15_NPU.ipynb`

### Akış
1. **Runtime → Change runtime type → High-RAM** seç.
2. 1. hücrede formu doldur: `SAFETENSORS_URL`, `MODEL_NAME`, `TIER=min`, `RESOLUTIONS`, `HF_REPO`.
3. Colab **Secrets** (🔑) içine ekle: `HF_TOKEN` (write), gerekiyorsa `CIVITAI_TOKEN`.
4. Hücreleri sırayla çalıştır.

### SDK otomatik — manuel adım yok
QNN SDK **2.28** `scripts/setup_qnn_sdk.py` ile otomatik indirilir; Qualcomm'dan
elle indirme veya Drive'a yükleme **yok**. Sürüm şarttır (bkz.
[`02-gereksinimler.md`](02-gereksinimler.md)); indirme linki `config.env`
içindeki `OVERRIDE_QAIRT_ASSET_URL` satırından gelir.

### Kısıtlar
- **RAM:** 512px için ~20 GB gerekir. Ücretsiz Colab (12 GB) OOM olabilir; notebook
  16 GB swap ekleyerek yardımcı olur ama **Colab Pro / High-RAM** çok daha güvenli.
- **Süre:** Çözünürlük başına saatler. Sekmeyi kapatma; Colab boşta kalırsa oturum düşer.
- **Çözünürlük:** 512×512 tabandır; ek boyutlar 1. adımdaki kutulardan seçilir ve
  her biri **tam bir dönüştürme turu** ekler. Ayrıntı:
  [`07-cozunurlukler.md`](07-cozunurlukler.md)

---

## GitHub Actions neden yok?

Bir zamanlar `.github/workflows/convert.yml` vardı; **silindi.** Kuantizasyon
512px için bile ~20 GB RAM istiyor, GitHub'ın ücretsiz runner'ı ~7 GB veriyor —
yani workflow yalnızca self-hosted bir runner'la anlam taşıyordu. Depoya hiç
runner kaydedilmedi, iki deneme koşusu da runner beklerken iptal edildi.
Üstelik workflow terk edilmiş elle boru hattını çağırıyordu; kalsaydı çalışmayan
bir yolu doğruymuş gibi gösterirdi.

Elinizde 32 GB+ RAM'li bir Linux makine varsa Actions'a gerek yok, scripti
doğrudan koşabilirsiniz:

```bash
export QNN_SDK_ROOT=...        # 2.28
scripts/06_official_pipeline.sh model.safetensors CyberRealistic work/ min
```
