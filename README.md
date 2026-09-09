# SD 1.5 → Qualcomm QNN (Local Dream / Ruya) Dönüştürme

civitai / Hugging Face üzerindeki **SD 1.5 `.safetensors`** modellerini,
telefonunuzdaki **Ruya / Local Dream** uygulamasının NPU'da çalıştırdığı
**`<isim>_qnn2.28_<soc>.zip`** formatına dönüştürmek için uçtan uca bir toolkit.

Dönüşümü **Local Dream'in kendi resmi scriptleri** (`npuconvertv2`) yapar; bu
depo onları indirir, QNN SDK 2.28 ile doğru sırada koşar ve çıktıyı uygulamanın
beklediği ZIP düzenine paketler.

---

## Hızlı başlangıç — Colab (önerilen)

Kendi Linux makineniz yoksa: **link gir → dönüştür → indir / HF'e yükle**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matrixportalx/Sd-1.5-Converting-to-Qualcomm-QNN-Model/blob/dev/notebook/SD15_NPU.ipynb)

→ [`notebook/SD15_NPU.ipynb`](notebook/SD15_NPU.ipynb)
(kökteki [`SD15_NPU_Official_Colab.ipynb`](SD15_NPU_Official_Colab.ipynb) aynı defterin kopyasıdır — eski bağlantılar bozulmasın diye duruyor.)

Gereken tek şey **yüksek bellekli (High-RAM) çalışma zamanı**. SDK ve resmi
scriptler otomatik iner.

> **civitai token şart.** civitai indirme uçları tokensiz `401` döner.
> civitai.com → *Account settings → API Keys → Add API key*, sonra Colab
> **🔑 Secrets → `CIVITAI_TOKEN`** (*Notebook access* açık olmalı). Gated **HF**
> linkleri için aynı şekilde `HF_TOKEN`.

## Hızlı başlangıç — kendi Linux'unuzda

```bash
pip install -r requirements.txt

# SDK 2.28'i indir (sürüm önemli — aşağıya bakın)
python scripts/setup_qnn_sdk.py --dest ./qairt \
  --asset-url "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip"
export QNN_SDK_ROOT=/yol/qairt/2.28.0.241029/qairt/2.28.0.241029

# Uçtan uca dönüştür:  <ckpt> <isim> <work_dir> [min|8gen1|8gen2]
bash scripts/06_official_pipeline.sh \
    /indirilenler/CyberRealistic.safetensors CyberRealistic work/cyber 8gen2

# 512×512 dışında boyutlar da isteniyorsa (her biri ayrı bir dönüştürme turu):
RESOLUTIONS="512x768,768x512,768x768" bash scripts/06_official_pipeline.sh \
    /indirilenler/CyberRealistic.safetensors CyberRealistic work/cyber 8gen2

# Çıktı:
#   dist/CyberRealistic_qnn2.28_8gen2.zip
```

ZIP'i telefona kopyalayıp **Ruya / Local Dream → Settings → Import Custom
Model** ile içe aktarın.

> Model adını **sade** verin (`CyberRealistic`). Sürüm ve SOC eki paket adına
> zaten eklenir; `CyberRealistic_qnn2.28_min` yazarsanız ek iki kez görünür.

---

## Hangi SOC'u seçmeliyim?

Bu, dönüşümün **en önemli tek kararıdır** ve kaliteyi değil **hızı** belirler.

| SOC değeri | HTP mimarisi | Hangi cihazlar |
|---|---|---|
| `min` | v68 | SD1.5 destekleyen **tüm** cihazlar (Snapdragon 7 serisi dahil) |
| `8gen1` | v69 | Snapdragon 8 Gen 1, 7 Gen 1, 7s Gen 2 |
| `8gen2` | v73 | Snapdragon 8 Gen 2, 8s Gen 3, 7+ Gen 2, 7 Gen 3 |

**Kural: kendi cihazınızın mimarisini seçin.** Düşük mimariye derlenmiş bir
binary yukarı cihazlarda *çalışır* ama **native derlenmişten belirgin biçimde
yavaştır** — donanımın yeni yeteneklerini kullanamaz.

Ölçüm (OnePlus 12R / Snapdragon 8 Gen 2, aynı model, aynı prompt ve seed,
20 adım · CFG 7 · 512×512):

| | `min` paketi | `8gen2` paketi |
|---|---|---|
| Görsel üretimi | 13,7 sn | **5,7 sn** (2,4× hızlı) |
| Model yükleme + graf hazırlığı | ~27,6 sn | **~4,2 sn** (6,5× hızlı) |
| Görsel kalitesi | — | **aynı** |

`min`'i yalnızca paketi **başkalarıyla paylaşacaksanız** ya da farklı nesil
cihazlarda kullanacaksanız seçin.

### Kalite SOC'tan bağımsızdır

Kaliteyi belirleyen ayarlar resmi hatta **sabittir** ve SOC'a göre değişmez:
`--act_bitwidth 16`, `--use_per_channel_quantization`, `redefined_modules/`
(MHA→SHA, Linear→Conv) ve `vtcm_mb: 2`. SOC yalnızca hangi
`htp_config_<soc>.json` ile derleneceğini seçer.

Dolayısıyla "8gen2 paketleri daha kalitesiz" gibi bir kural **yoktur**; öyle
görünen hazır paketler, farklı kuantizasyon ayarlarıyla dönüştürülmüş
olduklarından öyledir.

Cihaz ↔ mimari tam tablosu: [`docs/04-soc-htp-tablosu.md`](docs/04-soc-htp-tablosu.md)

---

## Çözünürlükler

NPU'da çözünürlük bir istek parametresi **değil**, derleme zamanı özelliğidir:
`unet.bin` sabit tensör şekilleriyle derlenir. Bu yüzden uygulama taban
512×512 binary'sini yükler ve başka bir boyut istendiğinde model klasöründeki
**zstd yamasını** açılışta `unet.bin`'e uygular. Yaması olmayan bir boyut
seçilemez (Ruya bu durumda üretime hiç başlamaz — yamasız 768 istenirse çıktı
renkli gürültü olurdu).

```bash
RESOLUTIONS="512x768,768x512,768x768,768x1024,1024x768,1024x1024" \
  bash scripts/06_official_pipeline.sh <ckpt> <isim> <work> 8gen2
```

512×512 **her zaman** üretilir (yamaların tabanı odur); listeye yazmaya gerek
yoktur. Yamalar paketin köküne, `unet.bin`'in yanına konur:

| Boyut | Dosya |
|---|---|
| 768×768 | `768.patch` (kare boyutlar tek sayıyla) |
| 1024×1024 | `1024.patch` |
| 512×768 | `512x768.patch` (dikdörtgen boyutlar `WxH`) |
| 768×512 | `768x512.patch` |

Kenarlar **64'ün katı** olmalıdır (SD1.5 UNet latent'i 3 kez yarılar).

### Bedeli

Her ek çözünürlük **tam bir dönüştürme turudur**: o boyutta kalibrasyon verisi
(`prepare_data`) + ONNX + kuantizasyon baştan koşar. Yalnızca UNet yeniden
üretilir — CLIP ve VAE taban koşudan gelir.

| | süre | RAM |
|---|---|---|
| taban 512×512 | 1× | ~20 GB |
| her ek boyut | +1× | 768 için ~2×, 1024 için ~4× |

Bu yüzden hat **kaldığı yerden devam eder**: biten bir çözünürlüğün yaması
pakette varsa o tur atlanır, `CACHE_REPO` doluysa hem kalibrasyon verisi hem
üretilmiş binary + yamalar Hugging Face'e yedeklenir. Kopan Colab oturumu
tekrar başlatıldığında yalnızca eksik boyutlar koşar.

### `min` uyarısı

Resmi tarif ek çözünürlükleri yalnızca `8gen1`/`8gen2` için üretir
(*"Non-flagship SOC versions can't run higher resolutions"*). `min` ile de
yama üretilir ama v68 sınıfı cihazlarda yüklenmeyebilir. 1024 kenarlı
boyutlar cihaz tarafında VTCM'ye sığmayabilir; önce 768 ile doğrulayın.

---

## Neden resmi hat?

Bu deponun ilk hattı (`convert_all.sh`, `qairt-converter` → DLC → context
binary; artık depoda **yok**) biçimsel olarak doğru paketler üretiyordu ama
**cihazda yüklenmiyorlardı.**
Resmi scriptler eline geçince sebep anlaşıldı — üçü de bizim tarafta
çözülemeyecek cinsten:

| bizim (terk edilen) | resmi (`npuconvertv2`) |
|---|---|
| `qairt-converter` → DLC | `qnn-onnx-converter` → `model.cpp` → `.so` → context binary |
| `--act_bitwidth 8` + io16 hilesi | **`--act_bitwidth 16`** |
| per-channel kapalı | `--use_per_channel_quantization` |
| stok diffusers | `redefined_modules/` — HTP dostu yeniden yazılmış model |
| VTCM ayarsız | `"vtcm_mb": 2` |

En kritik fark **girdi sırası**: `qnn-onnx-converter` yolunda sıra
`model.cpp`'deki bildirim sırasıdır; DLC yolunda "grafta ilk tüketim" kuralına
göre oluşuyordu ve değiştirilemiyordu. İkincisi, ONNX'i HTP'ye uygun yapan şey
bayraklar değil **modelin kendisidir** (`redefined_modules/`).

O hattan kalan dosyalar (`convert_all.sh`, `scripts/0[0-5]_*`, `probe_*`,
`gen_io_*` ve GitHub Actions workflow'u) **silindi**: çalışmayan bir yolu
doğruymuş gibi gösteriyorlardı. Depoda artık yalnızca gerçekten koşan yol var.
Gerekirse git geçmişinden okunabilirler.

---

## Gereksinimler

| Gereksinim | Not |
|---|---|
| **İşletim sistemi** | Linux veya Windows'ta **WSL2** (konvertörler yalnızca x86-64 Linux) |
| **RAM** | **20 GB+** (rehberin verdiği alt sınır) |
| **QNN SDK** | **2.28 — sürüm şart.** Pipeline başında doğrulanır; 2.39 ile koşmak saatleri boşa harcar |
| **Süre** | `prepare_data` ~35 dk (GPU'da ~3 dk), kuantizasyon **saatler** — **her ek çözünürlük bunu bir kez daha koşar** |
| **GPU** | Yalnızca `prepare_data.py`'yi hızlandırır (kuantizasyon/model-lib/context-binary **tamamen CPU**). Hat CUDA torch'u kilitli CPU sürümünün yerine kurar ve sonucu **ölçer**: `[cuda] torch … CUDA ETKIN` |

Kurulum ayrıntıları: [`docs/02-gereksinimler.md`](docs/02-gereksinimler.md)

### Uzun koşuyu kurtarmak: `CACHE_REPO`

En pahalı adımlar bir HF deposuna yedeklenebilir. Colab oturumu koparsa
(mobilde sekme arka plana atılınca oluyor) sonraki çalıştırma kaldığı yerden
devam eder. Yedeklenenler:

- `<slug>/res_<WxH>/` — her çözünürlüğün `prepare_data` çıktısı (`data.pkl`, `images/`)
- `<slug>/out_<soc>/` — birikmiş çıktı: taban binary + o ana kadarki yamalar
  (`CACHE_OUTPUT=0` ile kapatılabilir)

```bash
CACHE_REPO=sd-qnn-cache HF_TOKEN=hf_... bash scripts/06_official_pipeline.sh ...
```

`HF_TOKEN` yoksa önbellek sessizce kapanır.

### Kalibrasyon örneği sayısı: `CALIB_LIMIT`

Resmi tarif 400 örnek kullanır (saatler). `CALIB_LIMIT` ile kısaltılabilir:
`24` = boru hattını doğrula, `150` = iyi denge, `0` = tam. Kırpılmamış listeler
saklandığı için değeri sonradan **büyütmek de** mümkündür — `prepare_data`
tekrar koşmaz, yalnızca kuantizasyon yenilenir.

---

## Hat ne yapıyor?

`scripts/06_official_pipeline.sh` sırasıyla:

| Aşama | Ne yapar |
|---|---|
| — | `uv` ortamı (kilitli sürümler), libc++ / clang / zstd, CUDA torch |
| **taban 512×512** | |
| 0 | `CACHE_REPO` varsa önbelleği çeker (kalibrasyon verisi + birikmiş çıktı) |
| 1 | `prepare_data.py` — 20 prompt × difüzyon, kalibrasyon verisi (**en uzun adım**) |
| 2 | `gen_quant_data.py` — kuantizasyon girdi listeleri |
| 2b | Önbelleğe yazar |
| 3 | `export_onnx.py` — `redefined_modules` ile ONNX (sabit şekil) |
| 4 | `qnn-onnx-converter` → `qnn-model-lib-generator` → `qnn-context-binary-generator` |
| **her ek çözünürlük** | (yalnızca `RESOLUTIONS` verilmişse) |
| 1–2 | aynı adımlar, o boyutta |
| 3 | `export_onnx_unet_only.py --width W --height H` |
| 4 | `convert_all_unet_only.sh` — yalnızca UNet |
| 5 | `zstd --patch-from <taban unet.bin>` → `768.patch` / `512x768.patch` |
| **son** | |
| 6 | Çıktıyı `dist/<isim>_qnn2.28_<soc>.zip` olarak paketler |

Resmi scriptler (`npuconvertv2`) çalışma anında indirilir; depoda tutulmaz.

### ZIP'in içinde ne var?

Dosyalar **ZIP kökünde**, fazladan klasör olmadan (uygulamanın beklediği düzen):

```
CyberRealistic_qnn2.28_8gen2.zip
├── token_emb.bin      ← CLIP token gömme (ham fp16)
├── pos_emb.bin        ← pozisyon gömme (ham fp32)
├── clip_v2.mnn        ← metin kodlayıcı  → CPU/GPU (MNN)
├── unet.bin           ← QNN context binary → NPU (Hexagon)  [mimariye duyarlı kısım]
├── vae_decoder.bin    ← QNN
├── vae_encoder.bin    ← QNN (motor `--no_img2img` verilmedikçe yükler)
├── tokenizer.json     ← tüm SD1.5 için aynı CLIP tokenizer
├── 768.patch          ← (varsa) 768×768 için unet.bin zstd yaması
└── 512x768.patch      ← (varsa) 512×768 için …
```

`*.patch` dosyaları yalnızca `RESOLUTIONS` verildiğinde oluşur; uygulamanın
çözünürlük listesi tam olarak bu dosyalardan çıkarılır.

Formatın uygulama kaynağından doğrulanmış hâli ve teknik bulgular:
[`docs/DURUM.md`](docs/DURUM.md)

Sorun giderme: [`docs/05-sorun-giderme.md`](docs/05-sorun-giderme.md)

---

## Yasal / lisans

Bu depo yalnızca **dönüştürme araçlarını** içerir; herhangi bir model ağırlığı
içermez. Dönüştürdüğünüz her modelin kendi lisansına (civitai model sayfasındaki
kullanım koşulları) uymak sizin sorumluluğunuzdadır. Local Dream, MNN ve
Qualcomm QNN SDK kendi lisanslarına tabidir.

## Kaynaklar

- Local Dream (uygulama): <https://github.com/xororz/local-dream>
- **Resmi dönüştürme kılavuzu** (en yetkili kaynak):
  <https://ld-guide.chino.icu/zh/conversion/sd15> (Çince; tarayıcı çevirisi yeterli)
- Örnek dönüştürülmüş modeller: Hugging Face `Mr-J-369/*-SD1.5-qnn2.28`
- Qualcomm AI Engine Direct (QNN) SDK: Qualcomm AI Hub / QPM
