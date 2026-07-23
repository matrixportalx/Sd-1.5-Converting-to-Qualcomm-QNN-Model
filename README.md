# SD 1.5 → Qualcomm QNN (qnn2.28_min) Dönüştürme

civitai / Hugging Face üzerindeki **SD 1.5 `.safetensors`** modellerini,
telefonunuzdaki **Ruya / Local Dream** uygulamasının NPU'da çalıştırdığı
**`<isim>_qnn2.28_min.zip`** formatına dönüştürmek için uçtan uca bir toolkit.

> **Neden `_min`?** Birçok geliştirici artık sadece Snapdragon 8 Gen 2/3 için
> dönüştürüyor. `_min` varyantı en düşük Hexagon mimarisini (V68+) hedefler ve
> bu sayede **Snapdragon 7 serisi dahil** tüm uyumlu cihazlarda çalışır. Bu depo
> tam olarak bu varyantı üretir.

---

## Sorunun özü: neden SD7 ≠ SD8?

Local Dream'de görüntü üretiminin ağır parçası olan **UNet**, Qualcomm QNN
**"context binary"** olarak NPU'da (Hexagon HTP/DSP) çalışır. Bir context binary
**her zaman belirli bir HTP mimari sürümüne** göre derlenir:

| Snapdragon | HTP (DSP) mimarisi |
|---|---|
| 7 Gen 1 / 7s Gen 2 / 8 Gen 1 | v69 |
| 7+ Gen 2 / 7 Gen 3 / 8 Gen 2 / 8s Gen 3 | v73 |
| 8 Gen 3 | v75 |
| 8 Elite | v79 |

Düşük mimariye (ör. **v68/v69**) derlenmiş bir binary, o sürümden **yukarı**
olan tüm cihazlarda geriye-dönük uyumlu çalışır. Yüksek mimariye (v75) derlenmiş
bir binary ise Snapdragon 7'de **çalışmaz**. İşte "sadece SD8 için dönüştürülmüş"
modellerin telefonunuzda açılmama sebebi budur.

**Çözüm:** UNet'i en düşük mimariye (`tier = min → v68`) hedefleyerek derlemek.

---

## Mimari: ZIP'in içinde ne var?

Local Dream SD1.5 modelini **iki motora** böler:

```
AbsoluteReality_qnn2.28_min.zip
└── AbsoluteReality/
    ├── unet_512x512.bin      ← QNN context binary   → NPU (Hexagon)   [QNN'e çevrilir]
    ├── unet_512x768.bin
    ├── unet_768x512.bin
    ├── text_encoder.mnn      ← CLIP metin kodlayıcı  → CPU/GPU (MNN)   [MNN'e çevrilir]
    ├── vae.mnn               ← VAE çözücü            → CPU/GPU (MNN)   [MNN'e çevrilir]
    ├── tokenizer/            ← diffusers tokenizer
    └── model_info.json       ← meta veri
```

- **Sadece UNet** NPU'da çalışır → QNN'e dönüştürülür (mimariye duyarlı olan kısım).
- **text_encoder** ve **vae** MNN motorunda (CPU/GPU) çalışır → MNN'e dönüştürülür.
- NPU sabit (static) şekil istediğinden **her çözünürlük için ayrı UNet** derlenir
  (varsayılan: 512×512, 512×768, 768×512).

---

## Gereksinimler

| Gereksinim | Not |
|---|---|
| **İşletim sistemi** | Linux veya Windows'ta **WSL2** (konvertörler yalnızca x86-64 Linux) |
| **RAM** | 512px için **20 GB+**, yüksek çözünürlük için **64 GB+** (+ swap) |
| **Qualcomm AI Engine Direct SDK** | **2.28** (`v2.28.0.241029`) — Qualcomm AI Hub / QPM'den indirin |
| **MNN (MNNConvert)** | `-DMNN_BUILD_CONVERTER=ON` ile derlenmiş |
| **Python paketleri** | `pip install -r requirements.txt` |
| **Süre** | Her çözünürlük × her tier **saatler** sürebilir (CPU kuantizasyonu normaldir) |

> `qnn2.28` ismindeki **2.28**, kullanmanız gereken SDK sürümüdür. Farklı bir
> sürüm kullanırsanız binary çalışmayabilir — **tam olarak 2.28** kullanın.

Kurulum ayrıntıları için: [`docs/02-gereksinimler.md`](docs/02-gereksinimler.md)

---

## Hızlı başlangıç

```bash
# 0) Bir defaya mahsus kurulum
pip install -r requirements.txt
export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029      # SDK'yı açtığınız yol
export MNNCONVERT=/opt/MNN/build/MNNConvert        # derlenmiş MNNConvert

# 1) Uçtan uca dönüştür (Snapdragon 7 için tier = min)
./convert_all.sh /indirilenler/AbsoluteReality.safetensors AbsoluteReality min

# Çıktı:
#   dist/AbsoluteReality_qnn2.28_min.zip
```

Bu ZIP'i telefona kopyalayıp **Ruya / Local Dream → Settings → Import Custom
Model** ile içe aktarın.

Adımları tek tek çalıştırmak isterseniz: [`docs/03-donusum-adimlari.md`](docs/03-donusum-adimlari.md)

### 🚀 Otomatik: Colab veya GitHub Actions

Kendi Linux'unuz yoksa **link gir → dönüştür → HF reponuza yükle** akışını
hazır bir **Colab notebook** ile yapabilirsiniz:

- **Colab (önerilen):** [`notebooks/SD15_to_QNN_Colab.ipynb`](notebooks/SD15_to_QNN_Colab.ipynb)
  — safetensors linki + HF token girin, gerisini yapar. (High-RAM runtime + QNN SDK'yı
  bir kez Drive'a yüklemeniz gerekir.)
- **GitHub Actions:** [`.github/workflows/convert.yml`](.github/workflows/convert.yml)
  — **yalnızca self-hosted runner'da** çalışır (ücretsiz runner'lar RAM ve SDK
  lisansı nedeniyle yetersiz).

Ayrıntı ve kısıtlar: [`docs/06-otomasyon.md`](docs/06-otomasyon.md)

---

## Adımlar (özet)

| Adım | Script | Ne yapar |
|---|---|---|
| 0 | `scripts/00_load_safetensors.py` | `.safetensors` → diffusers klasörü |
| 1 | `scripts/01_export_onnx.py` | text_encoder / unet / vae → ONNX (sabit şekil) |
| 2 | `scripts/02_gen_quant_data.py` | UNet kuantizasyonu için kalibrasyon verisi |
| 3 | `scripts/03_convert_unet_qnn.sh` | UNet ONNX → **QNN context binary** (`unet_*.bin`) |
| 4 | `scripts/04_convert_mnn.sh` | text_encoder + vae ONNX → **MNN** (`*.mnn`) |
| 5 | `scripts/05_package.py` | Hepsini topla → `*_qnn2.28_min.zip` |

SoC/tier tablosunu görmek için: `python scripts/soc_targets.py`
Ayrıntı: [`docs/04-soc-htp-tablosu.md`](docs/04-soc-htp-tablosu.md)

---

## Önemli notlar ve dürüstlük payı

- Bu toolkit, **Local Dream'in belgelenmiş dönüştürme mantığını** ve QNN 2.28
  araç zincirini yeniden üretir. ONNX export ve paketleme adımları burada tam
  olarak çalışır; **QNN/MNN adımları** ise sizin kurduğunuz SDK'lara bağlıdır.
- QNN kuantizasyon bayrakları (`ACT_BW`/`WEIGHT_BW`) ve HTP config şeması, SDK
  sürümüne göre küçük farklılıklar gösterebilir. Bir şey oynamazsa, **çalışan bir
  resmi `*_qnn2.28_min.zip`** (ör. AbsoluteReality) dosyasının içini açıp dosya
  adlarını ve `model_info.json` düzenini birebir eşleştirin.
- **En yetkili kaynak:** Local Dream resmi dönüştürme kılavuzu →
  <https://ld-guide.chino.icu/zh/conversion/sd15> (Çince; tarayıcı çevirisi yeterli).

Sorun giderme: [`docs/05-sorun-giderme.md`](docs/05-sorun-giderme.md)

---

## Yasal / lisans

Bu depo yalnızca **dönüştürme araçlarını** içerir; herhangi bir model ağırlığı
içermez. Dönüştürdüğünüz her modelin kendi lisansına (civitai model sayfasındaki
kullanım koşulları) uymak sizin sorumluluğunuzdadır. Local Dream, MNN ve
Qualcomm QNN SDK kendi lisanslarına tabidir.

## Kaynaklar

- Local Dream (uygulama): <https://github.com/xororz/local-dream>
- Dönüştürme kılavuzu: <https://ld-guide.chino.icu/zh/conversion/sd15>
- Örnek dönüştürülmüş modeller: Hugging Face `Mr-J-369/*-SD1.5-qnn2.28`
- Qualcomm AI Engine Direct (QNN) SDK: Qualcomm AI Hub / QPM
