# Durum ve Devam Notları (beklemede)

Bu belge, dönüşüm çalışmasının nerede kaldığını ve ileride nasıl devam
edileceğini özetler. Çok fazla teknik detay burada — sıfırdan başlamamak için.

## Nereye geldik

Toolkit, `safetensors` → **eksiksiz ve doğru formatlı** bir Local Dream (Ruya)
SD1.5-NPU modeli üretiyor. Üretilen model telefonda **gerçek Hexagon NPU'ya
yükleniyor** (`QnnDsp` mesajları görülüyor). Kalan tek engel: NPU context
yüklemesinde `Could not free context` hatası — büyük olasılıkla **QNN sürüm
uyumsuzluğu** (aşağıda).

Pipeline'ın tamamı çalışıyor: export → CLIP böl → MNN → QNN kuantize →
context binary → paketle. Cihazda **çökme yerine yükleme** aşamasındayız.

## Doğrulanmış hedef format (uygulama kaynağından)

Local Dream SD1.5-NPU zip'i (KÖK dizinde, fazladan klasör YOK):

| Dosya | Format | Not |
|---|---|---|
| `token_emb.bin` | ham **fp16** `[49408,768]` | CLIP token gömme; uygulama lazy lookup yapar |
| `pos_emb.bin` | ham **fp32** `[77,768]` | pozisyon gömme; uygulama ekler |
| `clip_v2.mnn` | MNN | giriş `input_embedding [1,77,768]` → çıkış `last_hidden_state` |
| `unet.bin` | QNN **int8** | graf yükleme etiketi `"unet"` (etiket, ad eşleştirme değil) |
| `vae_decoder.bin` | QNN | çalışan referanslar ~96MB (**fp16**) |
| `vae_encoder.bin` | QNN | ~59MB; motor `--no_img2img` verilmedikçe **yükler** (eksikse çöker) |
| `tokenizer.json` | — | tüm SD1.5 için aynı CLIP tokenizer |

Kaynak: `xororz/local-dream` → `app/src/main/cpp/src/TextEncoder.hpp`,
`PipelineSd15Npu.hpp`, `QnnRuntime.hpp`.

## Hard-won teknik bulgular

- **Cihaz = Snapdragon 7 Gen 1 / 7s Gen 2 = HTP v69.** Hedef `dsp_arch=v69`.
- **16-bit aktivasyon (a16) MatMul yalnızca v73+ (HMX).** v69 → UNet **a8w8** şart.
- **Conv için v69 geçerli kombinasyonlar:** `a8w8` (INT8), `a16` (INT16, v69 yok),
  `fp16`. **`a8w16` GEÇERSİZ.**
- **HTP GroupNorm'u float(fp16) modda oluşturamıyor (QAIRT 2.39):** `q::QNN_GroupNorm
  could not create op`. Bu yüzden VAE'yi **int8 kuantize** ediyoruz (a8w8, 55MB).
  Ama çalışan referans VAE'ler **fp16/96MB** — yani **onlar 2.28 ile üretilmiş ve
  2.28 float VAE GroupNorm'u destekliyor.**
- **VAE decoder'a `latent/scaling_factor` (Div) KOYMA** — HTP float Div'i
  oluşturamıyor; ölçekleme uygulama tarafında.
- **UNet timestep RANK-1 `[1]` (float32)** olmalı; skaler (rank-0) HTP'de patlar.
- **CLIP'i EAGER attention ile export et** — SDPA yolu ONNX'e `IsNaN` ekler,
  MNN desteklemez.
- **QAIRT araçları Python 3.10 + libc++ ister** (`setup_qnn_python.sh`).
- **input_list yolları MUTLAK olmalı** (quantizer farklı CWD'den okur).
- HTP config: ana `--config_file` sadece `backend_extensions` içermeli;
  `graphs/devices` ayrı `config_file_path` dosyasında; `graphs` bloğu
  `graph_names` ister (biz kaldırdık, sadece `devices/dsp_arch`).

## KALAN ENGEL: `Could not free context`

Model NPU'ya yükleniyor ama context oluşturulamıyor. En güçlü hipotez:

**QNN sürüm uyumsuzluğu.** Binary'ler **QAIRT 2.39** ile üretildi; tüm çalışan
modeller **qnn2.28** ve resmi kılavuz **QNN 2.28 (v2.28.0.241029)** belirtiyor.
QNN context binary'leri **ileriye dönük uyumlu değil**: 2.28 runtime, 2.39
binary'sini yükleyemez. Uygulamanın NPU runtime'ı 2.28 ise sorun budur.

(Graf adı ELENDİ: `createModel(path,"unet")` ikinci argümanı yalnızca etiket.)

### Sürüm yolu muhtemelen çıkmaz (2026-07 bulgusu)

- Qualcomm Software Center'da **en eski erişilebilir sürüm 2.32.0.250228** —
  2.28 artık indirilemiyor.
- QNN context binary'leri **yalnızca geriye dönük** uyumlu: runtime ≥ binary
  sürümü olmalı.
- App build'i 2.39 ise → runtime muhtemelen 2.39 → bizim 2.39 binary'miz zaten
  uyumlu → `Could not free context` **sürüm değil**.
- Runtime 2.28 ise → 2.32 de çok yeni (2.28 < 2.32) → yine yüklenmez.
- **Sonuç:** Elde 2.28 olmadığından sürüm hipotezini ne test edebiliyoruz ne de
  düzeltebiliyoruz. **Gerçek anahtar `adb logcat` tam logu** —
  `Could not free context` öncesindeki satır asıl sebebi söyler.

## Nasıl devam edilir

1. **Kesinleştir (opsiyonel):** `adb logcat` ile motorun TAM logunu al.
   `Could not free context`'ten ÖNCEKİ satır gerçek sebebi (ör. "context binary
   version mismatch") söyler.
2. **QNN 2.28 ile yeniden üret:**
   - `Qualcomm AI Engine Direct SDK 2.28 (v2.28.0.241029)` edin (AI Hub/QPM).
   - Notebook'ta `QAIRT_REPO`/`QAIRT_TAG`'i 2.28'e çevir, `QNN_VERSION=2.28`.
   - **Yapılacak kod işi:** `scripts/03_convert_qnn.sh`'e eski `qnn-onnx-converter`
     + `qnn-model-lib-generator` yolu eklenmeli (2.28'de `qairt-converter` yok).
     Ayrıca 2.28'de VAE'yi **fp16** deneyebiliriz (GroupNorm float çalışırsa
     referansların 96MB'ıyla eşleşir, kalite artar).
3. Yüklenirse: kalite ayarı (VAE fp16, gerekirse `clip_skip_2`), ek çözünürlükler
   (`.patch` dosyaları — DarkSushi örneğinde 768.patch/1024.patch var).

## Alternatif (hemen çalışan çözüm)

Local Dream, `.safetensors`'ı **doğrudan CPU modeli** olarak içe aktarabiliyor
(uygulama otomatik MNN'e çevirir — `cyberrealisticLCM_cyberrealistic42` örneği
bu şekilde). NPU kadar hızlı değil ama **bugün çalışır**. Acil görüntü üretimi
için bu yol kullanılabilir; NPU dönüşümü beklerken.

---

# GÜNCEL DURUM (2026-07) — 16-bit UNet / v68

Yukarıdaki "sürüm uyumsuzluğu" hipotezi **ELENDİ**. Referans binary'nin
metadata'sı (`qnn-context-binary-utility`, notebook 4b) şunu gösterdi:

| tensör           | referans tip      | bizde (eski) |
|------------------|-------------------|--------------|
| `sample`         | `UFIXED_POINT_16` | `UFIXED_POINT_8` |
| `timestamp`      | `INT_32`          | `FLOAT_32` |
| `text_embedding` | `UFIXED_POINT_16` | `UFIXED_POINT_8` |
| `output`         | `UFIXED_POINT_16` | `UFIXED_POINT_8` |
| graf adı         | `model`           | `unet` |
| `dspArch`        | `68`              | `69` |

Yani `Could not free context` sürümden değil, **I/O tipi/graf adı
uyuşmazlığından** geliyordu. Hepsi düzeltildi (export v10/v11, graf adı `model`,
`min` tier varsayılanı `v68`).

## 16-bit MatMul tuzağı (asıl engel)

`a16w8` denendiğinde context-binary üretimi şununla düşüyor:

```
<E> [4294967295] has incorrect Value 68, expected >= 73.
<E> Failed to validate op .../attn1/MatMul with error 0xc26
```

v68/v69'da 16-bit MatMul **yalnızca kısıtlı kuantizasyon adımlarıyla**
destekleniyor. QAIRT `--help` de bunu söylüyor:

> `--restrict_quantization_steps` … *This argument is required for 16-bit
> Matmul operations.* (16-bit için değer: `"-0x8000 0x7F7F"`)

**Ama tek başına yetmiyor.** Bayrağı verince quantizer şu uyarıyı basıp
sessizce yok sayıyordu:

> `Restrict_quantization_steps is only supported for --param_quantizer =
> symmetric or per channel/row quantization. Value will be ignored.`

**Çözüm:** `--restrict_quantization_steps` ile birlikte **simetrik parametre
kuantalayıcı** verilmeli. `scripts/03_convert_qnn.sh` artık bunu otomatik
yapıyor: SDK `--help` çıktısını tarayıp mevcut bayrak adını seçiyor
(`--param_quantizer_schema symmetric`, yoksa `--param_quantizer symmetric`);
`PER_CHANNEL=1` ile per-channel da eklenebilir.

Ayrıca kuantalayıcı argümanları `<graf>_q.args` imza dosyasına yazılıyor;
argümanlar değişince `FORCE=1` gerekmeden yalnızca kuantizasyon + .bin
yenileniyor (ONNX export ve kalibrasyon korunuyor).

## UNET_MODE anahtarı

Notebook 1. adımda (form) `UNET_MODE` seçilebilir:

| değer | anlamı |
|-------|--------|
| `a16w8_restrict` (varsayılan) | referans reçete: 16-bit aktivasyon + restrict steps + simetrik param → v68'de 16-bit MatMul |
| `a16w8` | restrict yok — yalnızca v73+ derlenir |
| `a8w8` | tam 8-bit — her zaman derlenir, kalite daha düşük (yedek plan) |

Derleme yine `expected >= 73` derse: `UNET_MODE=a8w8` ile paket üretilir ve
cihazda **çalışan** ama düşük kaliteli bir model elde edilir; ya da
`DSP_ARCH=v73` (Snapdragon 8 Gen 2+) hedeflenir.

## Diğer sabit bulgular

- Zip **kök dizinde** paketlenmeli (klasör içinde klasör → motor açılmıyor).
- `vae_encoder.bin` **zorunlu** — motor img2img için varsayılan olarak yükler,
  yoksa "Motor süreci kapandı (kod 1)".
- Kalibrasyon tensör isimleri ONNX giriş isimleriyle birebir aynı olmalı:
  `sample`, `timestamp`, `text_embedding`.

## a8w8 bu uygulamada ASLA çalışmaz (2026-07-25 doğrulandı)

`UNET_MODE=a8w8` ile paket **üretildi** (829 MB unet.bin, zip 882 MB) ama
cihazda yine `Motor süreci beklenmedik şekilde kapandı (kod 1)` /
`Could not free context` verdi.

Sebep kesin: motor (`QnnModel.hpp`) UNet tamponlarına **sabit tiple** yazıyor —
`sample`/`text_embedding` → `uint16_t`, `timestamp` → `int32_t`. Binary
`UFIXED_POINT_8` beklerse motor eleman başına 2 bayt yazar, tampon taşar ve
süreç çöker. Yani **16-bit I/O zorunlu**, pazarlık payı yok.

Bu yüzden `scripts/check_bin_io.py` eklendi: üretilen `unet.bin`'in gerçek
girdi/çıktı tipleri `qnn-context-binary-utility` ile okunup motorun beklediğiyle
karşılaştırılıyor (`convert_all.sh` adım **6c**). Uyuşmazlık varsa paketi
telefona atmadan Colab çıktısında görülüyor.

Ayrıca kalibrasyon hatası düzeltildi: `timestep_*.raw` dosyaları **float32**
yazılıyordu, oysa ONNX girişi `INT_32`. Quantizer bit desenini tamsayı olarak
okuyordu (`1.0` → `1065353216`), zaman-gömme yolunun tüm encoding'leri
bozuluyordu. Artık int32 yazılıyor (`CALIB_VERSION=2`; eski kalibrasyon
otomatik yenileniyor).

## a16w8_restrict de yetmedi — yanlış şema bayrağı (2026-07-25)

`--param_quantizer_schema symmetric` eklemek "Value will be ignored" uyarısını
sustursa da v68 doğrulaması yine düştü. SDK `--help` metnini satır satır
okuyunca sebep ortaya çıktı — **yanlış bayrağı kullanmışız**:

| bayrak | kapsamı (SDK --help birebir) |
|--------|------------------------------|
| `--use_per_channel_quantization` | "per-channel quantization for **convolution-based** op weights" |
| `--use_per_row_quantization` | "rowwise quantization of **Matmul and FullyConnected** ops" |

Düşen op'lar tam olarak **MatMul**. `restrict_quantization_steps`'in aradığı
"symmetric **or per channel/row**" koşulunu MatMul için sağlayan bayrak
`--use_per_row_quantization`; per-channel yalnızca konvolüsyonları kapsıyor.
Artık `RESTRICT_STEPS` verildiğinde ikisi birden gönderiliyor.

Ayrıca `--help`'te daha önce fark edilmeyen iki mekanizma var:

- `--target_backend BACKEND` / `--target_soc_model SOC_MODEL` — backend-aware
  kuantizasyon; quantizer op bazında hedefin desteklediği hassasiyeti seçer.
- `--config CONFIG_FILE` — quantizer yapılandırma dosyası (op/tensor bazında
  bitwidth = gerçek karma hassasiyet).

Bunların bu SDK'daki tam sözdizimini tahmin etmemek için
`scripts/dump_sdk_help.sh` (notebook **4c**, ~30 sn) eklendi: `--help`
çıktılarının TAMAMINI ve SDK içindeki mixed-precision örneklerini döker.

`QUANT_EXTRA` form alanı ile quantizer'a kod değiştirmeden ek bayrak
geçilebiliyor.

### Kesinleşen mantık

Referans binary v68'de çalışıyor **ve** graf I/O'su 16-bit. 16-bit MatMul ise
v73+ istiyor. Dolayısıyla referansın **iç hesapları 8-bit, yalnızca graf sınırı
16-bit** olmak zorunda — yani hedef tam a16w8 değil, **karma hassasiyet**.
Bu artık bir hipotez değil, mantıksal zorunluluk.

## ÇÖZÜM YOLU: backend-aware kuantizasyon (2026-07-25)

`head -150` yüzünden `qairt` yardım metninin **"Backend Options"** bölümü
aylardır görülmemiş. Tam döküm (`scripts/dump_sdk_help.sh`, notebook 4c):

```
Backend Options:
  --target_backend BACKEND
      Use this option to specify the backend on which the model needs to run.
      Providing this option will generate a graph optimized for the given
      backend and this graph may not run on other backends.
      Supported backends are CPU,GPU,DSP,HTP,HTA,LPAI.
  --target_soc_model SOC_MODEL
      Use this option to specify the SOC on which the model needs to run.
      NOTE: --target_backend option must be provided to use --target_soc_model
```

Bu seçenekler **hem `qairt-converter` hem `qairt-quantizer`** için var. Hedef
SoC verilmediğinde araçlar genel bir graf üretiyor ve v68/v69'da desteklenmeyen
op'lar ancak context-binary aşamasında `expected >= 73` ile reddediliyor.
SoC söylendiğinde quantizer op bazında hedefin desteklediği hassasiyeti seçiyor.

SDK'nın kendi sürüm notu a16w8'in v68'de çalıştığını doğruluyor:

> Op:HTP: Addressed performance issues when converting models with **w8a16**
> compared to w8a8 on **SM8350** by optimizing matmul and Gemm

SM8350 (Snapdragon 888) = **HTP v68**. Yani a16w8 v68'de destekleniyor.

### HTP mimarisi → hedef SoC eşlemesi (03_convert_qnn.sh)

| DSP_ARCH | TARGET_SOC | cihaz |
|----------|-----------|-------|
| v68 | SM8350 | Snapdragon 888 / 778G |
| v69 | SM7450 | **Snapdragon 7 Gen 1** / 8 Gen 1 |
| v73 | SM8550 | Snapdragon 8 Gen 2 |
| v75 | SM8650 | Snapdragon 8 Gen 3 |
| v79 | SM8750 | Snapdragon 8 Elite |

`min` tier → v68 → SM8350 (referansla birebir). Notebook'ta `TARGET_SOC`
alanından değiştirilebilir; `"yok"` seçilirse backend-aware kapanır.

### Ayrıca: akıllı yeniden üretim

Her aşama artık argüman imzası tutuyor (`<graf>.args`, `<graf>_q.args`,
`<ad>.bin.args`). Bir ayar değişince yalnızca etkilenen aşamalar yeniden
çalışıyor; `FORCE=1` ile her şeyi baştan yapmak gerekmiyor. `convert_all.sh`
içindeki "`.bin` varsa atla" kontrolleri kaldırıldı — bu kontroller yüzünden
ayar değişse bile eski binary korunuyordu.

## SoC listesi sabit kodlanamaz (2026-07-25)

`--target_soc_model SM8350` denendi, QAIRT 2.39 reddetti:

```
ERROR - Encountered Error: SOC model SM8350 is not supported.
  File ".../converters/common/backend_awareness.py", line 59, in get_instance
    raise Exception("SOC model {} is not supported.".format(soc_model))
```

Desteklenen SoC listesi SDK sürümüne göre değişiyor (2.39 eski çipleri
düşürmüş). Bu yüzden `scripts/list_soc_models.py` eklendi: listeyi
`backend_awareness` modülünü introspect ederek (tutmazsa SDK kaynaklarını
tarayarak) çalışma anında çıkarıyor ve hedef HTP mimarisine uyan
**desteklenen** bir SoC seçiyor. Kullanıcının verdiği `TARGET_SOC` listede
yoksa uyarı basılıp SoC atlanıyor — koşu ölmüyor, `--target_backend HTP`
tek başına devam ediyor.

## Otomatik mimari yedeklemesi

Context binary üretimi hedef mimaride düşerse bir üst mimari deneniyor
(`BIN_ARCH_FALLBACK`, varsayılan `auto` → v68 başarısızsa **v69**). Snapdragon
7 Gen 1 zaten **v69** olduğundan v69 binary telefonda çalışır; yalnızca v68
cihazlarda çalışmaz. Böylece tek koşuda sonuç alınıyor, 30 dk'lık tur
tekrarlanmıyor. Kullanılan mimari `<ad>.arch` dosyasına yazılıp adım 6c'de
raporlanıyor.

## restrict_quantization_steps AĞIRLIK bit genişliğine uygulanıyor (2026-07-25)

```
Restricting number of quantization steps to: min: -32768 - max: 32639
ERROR: Cannot restrict quantization steps to -32768 - 32639 for bitwidth: 8 for symmetric
```

`--help`'teki `"-0x8000 0x7F7F"` örneği **16-bit ağırlık** içindir. Bizde
`--weights_bitwidth 8` olduğu için aralık da 8-bit olmalı: `-0x80 0x7F`.
Aktivasyon 16-bit olsa bile restrict, **parametre (ağırlık)** kuantalayıcısına
uygulanıyor.

`RESTRICT_STEPS=auto` eklendi: aralık `WEIGHT_BW`'den türetiliyor
(w8 → `-0x80 0x7F`, w16 → `-0x8000 0x7F7F`).

Not: bu hata **saniyeler içinde** çıkıyor (kalibrasyondan önce), yani bu
aşamadaki denemeler ucuz.

## Çalışma zamanı kapanınca ilerlemenin kaybolması

Colab çalışma zamanı kapandığında her deneme sıfırdan başlıyordu (~35 dk).
Notebook **2c** hücresi eklendi:

- SDK arşivi Drive'da önbelleğe alınıyor (`QAIRT_CACHE`, ~2 GB) — indirme
  atlanıyor. Açma **yerel diske** yapılıyor; Drive üzerinden binary
  çalıştırmak izin/hız sorunu çıkarıyor.
- `work/` isteğe bağlı olarak Drive'a bağlanıyor (~11 GB/model) — ONNX export,
  kalibrasyon ve DLC'ler korunuyor, sonraki koşu ~5 dk'ya iniyor.

`setup_qnn_sdk.py` artık `--cache-dir` alıyor ve SDK zaten açıksa hiç
dokunmuyor.

## MatMul aşıldı, sıra Conv'da (2026-07-25)

`--param_quantizer_schema symmetric` + `--use_per_row_quantization` +
`--restrict_quantization_steps -0x80 0x7F` ile **16-bit MatMul doğrulamayı
geçti** — `expected >= 73` hatası artık yok. Yeni hata ilk konvolüsyonda:

```
<E> [4294967295] has incorrect Value 320, expected equal to 320.
<E> Failed to validate op /unet/conv_in/Conv with error 0xc26
```

320 = `conv_in`'in çıkış kanal sayısı; mesaj kendi içinde çelişkili
("320, expected equal to 320"). Bu, **per-channel** ağırlık kodlamasının
Conv'da tutmadığının işareti. Per-channel zaten gereksizdi: restrict'in aradığı
şema koşulunu `symmetric` + `per_row` sağlıyor, `--use_per_channel_quantization`
yalnızca konvolüsyon ağırlıklarını değiştiriyor.

→ `PER_CHANNEL` varsayılanı **kapalı**. Açmak için `PER_CHANNEL=1`.

## FAST_TRIAL — deneme turlarını 8 dk'dan 1 dk'ya indirir

Kuantizasyon süresi kalibrasyon örneği sayısıyla doğru orantılı (her örnek
CPU'da ~28 sn). `FAST_TRIAL=1` örnek sayısını asgariye indiriyor
(UNet 1 prompt × 2 adım, VAE 1 × 2):

| mod | kalibrasyon | kuantizasyon |
|-----|-------------|--------------|
| normal | 16 örnek | ~8 dk |
| `FAST_TRIAL=1` | 2 örnek | ~1 dk |

Boru hattının **derlenip derlenmediğini** sınamak için kullanılır; üretilen
model çalışır ama kalitesi düşüktür. Derleme başarılı olunca `FAST_TRIAL=0`
ile bir kez daha koşulur. Notebook'ta `HIZLI_DENEME` kutusu.

Kalibrasyon damgalarına örnek sayıları eklendi; mod değişince ilgili
kalibrasyon ve kuantize DLC otomatik yenileniyor.

## Conv aşıldı, sıra LayerNorm'da — tam a16 v68'de mümkün değil (2026-07-25)

`PER_CHANNEL` kapatılınca `/unet/conv_in/Conv` geçti. Yeni hata ilk transformer
bloğunda:

```
<E> None of the combinations match the provided case
<E> Failed to validate op .../transformer_blocks.0/norm1/LayerNormalization
```

"None of the combinations match" = op'un dtype kombinasyonu HTP tablosunda yok.
v68 **ve** v69'da aynı hata. Yani **LayerNorm 16-bit aktivasyonu bu
mimarilerde desteklenmiyor**.

### Kesin sonuç

Tam `a16w8` bu cihazda mümkün değil. Adım adım öğrendiklerimiz:

| op | a16 v68'de | çözüm |
|----|-----------|-------|
| MatMul | ✗ → ✓ | symmetric + per_row + restrict steps |
| Conv | ✓ (per_channel kapalı) | `PER_CHANNEL=0` |
| LayerNorm | ✗ | **yok** — 8-bit olmalı |

Referans binary v68'de çalışıyor ve I/O'su 16-bit olduğuna göre iç hesabı
8-bit olmak **zorunda**: karma hassasiyet.

### İki yol

1. **`UNET_MODE=a8w8_io16`** — iç hesap 8-bit (hepsi geçerli), yalnızca
   `sample`/`text_embedding`/`output` 16-bit (`--quantization_overrides`,
   `gen_io_encodings.py`). QNN sınır ile iç graf arasına Convert op'ları ekler.
2. **`qairt-converter --config <yaml>`** — SDK'nın resmi I/O yapılandırması.
   `--dump_config_template` ile şema dökülüyor. Şemayı tahmin etmemek için
   `dump_sdk_help.sh` bölüm 8'e ve adım 4'ün loguna eklendi.

Ayrıca `dump_sdk_help.sh` bölüm 10, `htp_opdef_version_history.html`'i düz
metne çevirip hangi HTP sürümünde hangi op'un 16-bit'e açıldığını listeliyor.

## config.env — not defterini bir daha açmaya gerek yok (2026-07-25)

**Sorun:** Her ayar/kod değişikliğinde not defterini GitHub'dan yeniden açmak
gerekiyordu. Colab'da bu **yeni çalışma zamanı** demek: `work/`, SDK, pipeline
— hepsi silinir, her deneme baştan ~35 dk.

**Çözüm:** Ayarlar artık depodaki `config.env`'den okunuyor.
`OVERRIDE_<AD>=<değer>` satırları not defterindeki seçimleri **ezer**.

`config.env` not defterinin **2. adımındaki** `git reset --hard` ile
güncellendiği için akış şu hale geldi:

```
(aynı oturum)  2. adım  →  6. adım
```

Not defteri dosyası bir daha değişmeyecek; tüm ayar/mod değişiklikleri
`config.env` üzerinden yapılacak.

## 16-bit etiketi iç grafa sızıyor — Clip bariyeri (2026-07-25)

`a8w8_io16` (yalnız sınır tensörleri 16-bit) denendi:

```
mixedPrecisionForWeights: /unet/.../attn2/MatMul_1
because 8 bit activations with 16 bit weights are not supported on backend
```

Mekanizma: `text_embedding`'in 16-bit encoding'i cross-attention'ın
`to_k`/`to_v` MatMul'leri üzerinden **iç grafa taşınıyor**. QNN, MatMul'ün
ikinci operandını "ağırlık" saydığı için K/V 16-bit oluyor, aktivasyonlar
8-bit kalıyor → desteklenmeyen kombinasyon.

Ayrıca kısmi override dosyası ("Processed 5 quantization encodings") yüzünden
grafın büyük kısmı **float'a fallback** ediyordu — HTP için ayrıca sorunlu.

**Çözüm (export v12):** ONNX grafında 16-bit sınır ile 8-bit iç graf arasına
**ağırlıksız bir bariyer** konuyor — `torch.clamp` → ONNX `Clip` →
QNN `ReluMinMax`:

```python
s   = torch.clamp(sample, -1e4, 1e4)
e   = torch.clamp(encoder_hidden_states, -1e4, 1e4)
out = torch.clamp(unet(s, timestep, e).sample, -1e4, 1e4)
```

Clip'in ağırlığı olmadığı için `mixedPrecisionForWeights` tetiklenmiyor; QNN
16-bit giriş ile 8-bit çıkış arasına Convert ekleyebiliyor ve etiket daha ileri
gitmiyor. Sınırlar gerçek dağılımın çok üzerinde (kırpma yapmaz), yalnızca op
sadeleştirilmesin diye sonlu.

## SDK 2.28 arayışı — Software Center'ın doğrudan API yolu (2026-07-25)

Software Center'ın **arayüzü** yalnızca güncel sürümleri listeliyor (en eski
2.32), ama **doğrudan API yolu** genelde eski sürümleri de sunmaya devam
ediyor. Radxa ve sherpa dokümanları bu URL'yi girişsiz düz `wget` ile
kullanıyor:

```
https://softwarecenter.qualcomm.com/api/download/software/sdks/
    Qualcomm_AI_Runtime_Community/All/<SÜRÜM>/v<SÜRÜM>.zip
```

`scripts/probe_qairt_versions.py` 2.24 – 2.32 arası adayları HEAD isteğiyle
yokluyor ve indirilebilenleri boyutlarıyla listeliyor. `config.env`'de
`OVERRIDE_PROBE_SDK=1` ile koşunun başında çalışıyor (~30 sn, dönüşümü
engellemiyor).

Bir sürüm bulunursa `OVERRIDE_QAIRT_ASSET_URL` ile kullanılır:
`setup_qnn_sdk.py` artık `config.env`'i **kendisi okuyor**, yani SDK sürümünü
değiştirmek için de not defterine dokunmak gerekmiyor.

## DÖNÜM NOKTASI: sorun eski SDK değil, YENİ SDK gerekiyor (2026-07-25)

QAIRT resmi sürüm notları (2.34 → 2.48) incelendi. **Bizim iki hatamız da
2.40.0'da (Ekim 2025) düzeltilmiş — yani 2.39'dan hemen sonra:**

| bizim hata | 2.40.0 sürüm notu |
|---|---|
| `/unet/conv_in/Conv` "has incorrect Value 320, expected equal to 320" | *"Tool:Converter: Resolved an issue where models with **Conv2d ops failed on the HTP backend due to unsupported input or output data types**. {153277}"* |
| `norm1/LayerNormalization` "None of the combinations match the provided case" | *"Tool:Converter: Resolved an issue where the **LayerNorm Op failed validation due to an unsupported data type**. {153276}"* |

Ayrıca **2.39.0**'ın kendi notunda karma hassasiyet hatamızın sebebi yazıyor:

> *"Tool:Converter: **Enabled support for dynamic 16-bit weights by default** in
> qairt-converter and qairt-quantizer. … **A new `--disable_dynamic_16_bit_weights`
> flag has been added to revert to 8-bit conversion if needed.** {147008}"*

Bu tam olarak `mixedPrecisionForWeights: … 8 bit activations with 16 bit weights`
hatasının sebebi: 2.39 MatMul'ün dinamik ikinci operandını **varsayılan olarak**
16-bit'e çekiyor. Bayrak `--help`'te görünmüyor (gizli), bu yüzden
`has_hidden_flag` ile doğrudan denenerek varlığı sınanıyor.

2.47.0'da ek olarak: *"Tool:Converter: Fixed a Convert Op issue in the
**mixed-precision stage**. {165230}"*

### Sonuç

Aylardır 2.28'i aradık; oysa referansın 2.28 ile çalışması, 2.39'un **geçici
olarak bozuk** olmasıyla açıklanıyor. **2.40+ indirilebilir durumda** —
2.28'in aksine.

- `config.env` artık QAIRT **2.40.0.251030**'u kullanıyor
  (`OVERRIDE_QAIRT_ASSET_URL`), `UNET_MODE=a16w8_restrict`.
- `setup_qnn_sdk.py` sürümleri **ayrı dizinlere** açıyor (2.39 üzerine yazmıyor).
- Yoklama listesi ileri sürümlerle güncellendi (2.40 – 2.48).
- İnmezse yoklama çıktısındaki başka bir sürüme geçmek `config.env`'de tek satır.

Honor 90 / Snapdragon 7 Gen 1 (HTP v69) desteğiyle ilgili **kaldırma yok** —
sürüm notlarında böyle bir madde geçmiyor; sorun baştan beri araç zinciriydi.

## 2.40 de LayerNorm'u çözmedi — ama I/O config şablonu geldi (2026-07-27)

QAIRT 2.40.0 indirildi ve kullanıldı. `a16w8_restrict` yine aynı yerde düştü:

```
None of the combinations match the provided case
Failed to validate op .../norm1/LayerNormalization
```

v68 ve v69'da aynı. 2.40'ın "LayerNorm Op failed validation due to an
unsupported data type" düzeltmesi bizim durumumuz değilmiş. **Tam 16-bit
aktivasyon bu donanımda kesin kapalı.**

Aynı koşuda `--dump_config_template` nihayet çalıştı ve aradığımız mekanizmanın
şeması döküldü:

```yaml
Input Tensor Configuration:
  - Name: sample
    Src Model Parameters:
        DataType:
        Layout:
    Desired Model Parameters:
        DataType:          # <- uint16
        QuantParams:
          Scale:           # <- (max-min)/65535
          Offset:          # <- round(min/Scale), negatif
Output Tensor Configuration:
  - Name: output
    ...
```

Bu, `--quantization_overrides`'tan **temel olarak farklı**: yalnızca graf
sınırındaki tensörlerin istenen tipini belirtir, iç grafın kuantizasyonuna
hiç karışmaz. Dolayısıyla 16-bit etiketi `to_k`/`to_v` üzerinden MatMul
ağırlıklarına sızamaz.

`scripts/gen_io_config.py` bu YAML'i kalibrasyondan hesaplanan Scale/Offset ile
üretiyor; `UNET_MODE=a8w8_io16cfg` iç hesabı a8w8 bırakıp YAML'i
`qairt-converter --config` ile veriyor.

Ayrıca Clip bariyeri kaldırıldı (export v13): converter opset-13 Clip'i
desteklemiyor ("Expected operator version: [1, 6, 11, 12]") ve `--config`
yoluyla artık gereksiz. `UNET_CLIP_BARRIER=1` ile geri açılabilir.

## `--config` kabul edildi, kalan tek engel dinamik 16-bit ağırlıklar (2026-07-27)

`a8w8_io16cfg` koşusu: converter YAML'i kabul etti
("Validating user provided custom IO") ve dönüşüm başarılı. Kuantalayıcı yine
aynı yerde düştü — **ama sebebi artık farklı:**

```
[quantizer] a8 w8 b32 --target_backend HTP        <- --disable_dynamic_16_bit_weights YOK
mixedPrecisionForWeights: .../attn2/MatMul_1
```

Bayrak, kodda yanlışlıkla `QUANT_OVERRIDES` doluysa eklenecek şekilde
koşullanmıştı; `io16cfg` modunda o değişken boş olduğu için hiç geçmedi.
Oysa 2.39 sürüm notuna göre bu davranış **varsayılan olarak açık** ve
override/config'ten bağımsız:

> *"Enabled support for dynamic 16-bit weights **by default** … A new
> `--disable_dynamic_16_bit_weights` flag has been added to revert to 8-bit
> conversion if needed."*

Artık bayrak koşulsuz veriliyor (converter + quantizer), yani 2.39 öncesi —
referansın üretildiği — davranışa dönülüyor. `DISABLE_DYN16W=0` ile kapatılır.

## Kuantizasyon geçti; sıra `/unet/Expand`'de (2026-07-27)

`--disable_dynamic_16_bit_weights` koşulsuz verilince **kuantizasyon başarıyla
tamamlandı** — `mixedPrecisionForWeights` hatası tarihe karıştı. Context-binary
üretimi bir sonraki op'ta düştü ve HTP bu kez kabul ettiği kombinasyonların
TAM LİSTESİNİ bastı:

```
'Reshape' in '/unet/Expand'
  istenen:  in[0]:INT_32              out[0]:UFIXED_POINT_8    ✗
  OTHERS-3: in[0]:INT_32              out[0]:INT_32            ✓
  OTHERS-4: in[0]:FLOAT_32            out[0]:UFIXED_POINT_8    ✓
```

`/unet/Expand`, diffusers'ın `timesteps.expand(batch)` adımı. `timestamp`
girişi INT_32 olduğu için Expand int32 alıyor, ama çıktısı kuantize graf'a
girdiğinden uint8 isteniyor — bu ikili tabloda yok.

**Çözüm (export v14):** `timestep` graf girişinden hemen sonra float32'ye
çevriliyor (`timestep.to(torch.float32)`). Böylece Expand FLOAT_32 girdili olur
ve OTHERS-4 kombinasyonu geçerli hale gelir. **Graf girişinin tipi değişmez** —
`timestamp` hâlâ INT_32, motorun yazdığı gibi.

## `/unet/Expand` — Cast işe yaramadı, düğümü grafa hiç yazdırmıyoruz (v15)

v14'te `timestep` girişte float32'ye çevrildi ama converter Cast'i katlayıp
attı ve Expand yine int32 aldı:

```
Only numerical type cast is supported. The cast op: /Cast will be interpreted
at conversion time
```

`/unet/Expand`, diffusers'ın `timesteps.expand(sample.shape[0])` satırı.
`timesteps` şekli `[1]`, batch de 1 → bu bir **kimlik işlemi**; sadece izleme
sırasında düğüm olarak yazılıyor. `timestamp` girişi INT_32 kalmak zorunda
(motor öyle yazıyor), dolayısıyla çözüm düğümü hiç oluşturmamak.

**v15:** export sırasında `torch.Tensor.expand` geçici olarak sarmalanıyor;
1-boyutlu tensörde kendi uzunluğuna expand çağrısı `self` döndürüyor, ONNX'e
düğüm yazılmıyor. Diğer expand'lar (dikkat maskeleri vb.) etkilenmiyor —
koşul dar. Aynı teknik CLIP gömme ayırmada da kullanılıyor.

Not: `a8w8` (io-config'siz) derlemesinin daha önce sorunsuz geçmesinin sebebi,
o modda `timestamp`'ın da uint8'e kuantize edilmesiydi — Expand uint8→uint8
oluyordu. io-config ile INT_32 olunca kombinasyon geçersizleşti.

## Expand hâlâ duruyor — kanca tetiklenmemiş (v16)

v15'te `torch.Tensor.expand` sarmalandı ama `/unet/Expand` ONNX'te kaldı.
Koşuldaki `isinstance(sizes[0], int)` izleme sırasında tutmuyor — `sample.shape[0]`
düz `int` yerine izlenmiş bir değer dönüyor.

**v16 iki katmanlı:**

1. Kanca tip-bağımsız hale getirildi (`int(tgt)` denemesi, başarısızsa orijinal
   expand'a düşer) ve kaç düğüm elendiği loglanıyor:
   `[expand] N kimlik expand elendi`
2. **ONNX düzeyinde emniyet ağı** (`_strip_timestamp_expand`): kanca tutmazsa,
   `timestamp` girişinden beslenen ve hedef şekli girişin kendi şekline eşit
   olan Expand düğümü doğrudan grafdan çıkarılıp tüketiciler yeniden bağlanıyor.
   Model >1.8 GB ise external-data ile kaydediliyor.

Emniyet ağı birim testten geçti: kimlik Expand siliniyor ve `Cast` doğrudan
`timestamp`'a bağlanıyor; şekil büyüten gerçek Expand korunuyor.

## Köstebek oyunu bitti: int32 yolu ONNX'ten tamamen kaldırıldı (v17)

v16 çalıştı — `[expand] 1 kimlik expand elendi` — ama bu sefer bir sonraki
düğüm patladı:

```
QnnDsp <E> graph_prepare.cc: Op 0x... preparation failed with err:-1
Failed to validate op /unet/time_proj/Unsqueeze with error 0xc26
in[0] INT_32 ... out[0] UFIXED_POINT_8 ... None of the combinations match
```

Yani sorun `Expand` değil: **int32 `timestamp` üzerindeki HER şekil işlemi**
(`Expand`, `Unsqueeze`, `Reshape`, `Concat`...) aynı kurala takılıyor. HTP'nin
kabul listesinde "girişi INT_32, çıkışı UFIXED_POINT_8" diye bir kombinasyon
yok, çünkü int32 girdi ancak int32 çıktı verebilir; kuantize edilmiş çıkış
istendiğinde eşleşme kalmıyor. Düğümleri tek tek elemek köstebek oyunu.

Araya `Cast` koymak da işe yaramıyor — v14'te denendi, converter Cast'i katlayıp
atıyor ("will be interpreted at conversion time").

**Kök çözüm — v17:** int32 yolunu ONNX tarafında hiç oluşturmuyoruz.
`--config` YAML'ının iki ayrı alanı tam bunun için var:

```yaml
- Name: timestamp
  Src Model Parameters:
      DataType: float32      # ONNX modelindeki tip
  Desired Model Parameters:
      DataType: int32        # QNN graf sınırındaki tip (motorun yazdığı)
```

Böylece:
* ONNX'te `timestamp` float32 → `time_proj` ve tüm zaman-gömme yolu
  baştan sona float/kuantize; hiçbir INT_32 şekil işlemi yok.
* QNN graf **sınırında** tip yine INT_32 → motor (`local-dream`/Ruya) sabit
  yazdığı int32 ile uyumlu, referans binary ile aynı.

Yan değişiklikler:
* `01_export_onnx.py`: `timestep = torch.tensor([1], dtype=torch.float32)`
* `02_gen_quant_data.py` (calib v3): timestep raw'ları tekrar float32 —
  ONNX girişi float olduğu için quantizer'ın doğru okuması bu şekilde.
  (v2'de int32'ye çevrilmişti; o, ONNX girişi int32 iken doğruydu.)
* `gen_io_config.py`: `_tensor_block(..., src_dtype=...)` eklendi.

v16'nın expand kancası ve ONNX emniyet ağı **yerinde bırakıldı** — artık
tetiklenmesi gerekmiyor ama zararsız ve geri dönülürse hazır.

## Kaynak koddan kesin cevap + gerçek çözüm (v18)

v17 (ONNX'te float32 timestamp) da aynı hatayı verdi: io-config'deki
`Desired: int32` sınırda yine INT_32 ürettiği için `time_proj/Unsqueeze` yine
`INT_32 -> UFIXED_POINT_8` istedi. Tahmin etmeyi bırakıp local-dream'in
kaynağına bakıldı — `app/src/main/cpp/src/QnnModel.hpp`, `executeUnetGraphs`:

```cpp
// latents  (inputs[0])
uint16_t *latents_uint16 = (uint16_t*)QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data;
datautil::floatToTfN(latents_uint16, latents,
                     inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
                     inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, n);

// timestep (inputs[1])
int32_t *positionData = (int32_t*)QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data;
positionData[0] = timestep;          // HAM int32 — kuantizasyon yok

// text_embedding (inputs[2])  -> yine uint16 + floatToTfN
```

Kesinleşen kurallar:
* `timestamp` graf sınırında **INT_32 olmak zorunda**. Tartışmaya kapalı.
* `sample` ve `text_embedding` **16-bit olmak zorunda** (`uint16_t*` cast).
  Ama scale/offset binary'den okunuyor → bizim hesapladığımız değerler geçerli.
* Tensörler **isimle değil sıra ile** eşleşiyor (0=sample, 1=timestamp,
  2=text_embedding) ve `numInputTensors != 3` ise reddediliyor.
* Çıkış `convertToFloatInto` ile genel olarak okunuyor.

Referans `_min` paketi (cyberrealistic_final_qnn2.28_min): clip_v2.mnn,
pos_emb.bin, token_emb.bin, tokenizer.json, unet.bin (**893 MB** → ağırlıklar
8-bit), vae_decoder.bin, vae_encoder.bin. `clip.mnn` (tam CLIP) pakette YOK.

### Çözüm: time_proj → önceden hesaplanmış tablo + Gather

HTP'nin hata mesajındaki kabul listesi, int32'yi kuantize dünyaya bağlayan
hiçbir şekil işlemi tanımıyor. Ama **Gather** tanıyor: veri kuantize, **indeks
int32**, çıktı kuantize — indeksler zaten tanımı gereği int32.

`time_proj` yalnızca t'nin fonksiyonu ve t tamsayı (motor `static_cast<int>`
yapıyor). Dolayısıyla 1000 timestep'in tamamı önceden hesaplanıp `[1000, 320]`
sabit tabloya konabilir ve grafta `Gather(table, timestamp)` kalır. Sonuç
**birebir aynı** — yaklaşıklık değil, sin/cos'un aynı değerleri.

```python
class TimeProjTable(torch.nn.Module):
    def __init__(self, time_proj, num_train_timesteps=1000):
        super().__init__()
        ts = torch.arange(num_train_timesteps, dtype=torch.float32)
        with torch.no_grad():
            self.register_buffer("table", time_proj(ts).float())
    def forward(self, timesteps):
        return self.table.index_select(0, timesteps)   # -> ONNX Gather
```

Yan etki (olumlu): `timesteps.expand(1)` düğümü hayatta kalsa bile artık
zararsız — çıktısı Gather indeksi olarak int32 kalıyor ve HTP `Reshape`
INT_32→INT_32'yi destekliyor (hata mesajındaki OTHERS #3). v16'nın expand
kancası ve ONNX emniyet ağı yine de yerinde bırakıldı.

Kalibrasyon (v4) ve io-config timestamp'i tekrar int32'ye döndü.

## Disk doldu (adım 4) — alınan önlemler

v18 export'u sorunsuz geçti (`[time_proj] 1000x320 tablo`), ama dönüşüm bu sefer
başka bir yerde öldü:

```
OSError: [Errno 28] No space left on device:
  '.../unet.down_blocks.3.resnets.1.time_emb_proj.weight' -> '/tmp/tmpXXXX/...'
RuntimeError: Failed to copy external data to the disk at: /tmp/tmpXXXX.
  Try setting QAIRT_TMP_DIR environment variable to different location.
```

`qairt-converter` ONNX'in harici ağırlıklarını (UNet için ~3.4 GB) geçici bir
dizine **kopyalıyor**, üstüne fp32 DLC (~3.4 GB) üretiyor. Colab'da toplam
ihtiyaç ~20 GB'ı geçiyor.

Önlemler:
* `QAIRT_TMP_DIR` artık `work/<model>/tmp` — yerini biliyoruz, koşu başında ve
  adım 4 sonrasında temizleniyor.
* `pipeline/` yazıldıktan sonra `input.safetensors` siliniyor (`KEEP_CKPT=1`
  ile korunur). Birkaç GB.
* `clip.onnx` (tam CLIP) artık **üretilmiyor** — referans `_min` paketinde
  `clip.mnn` yok, ekran görüntüsüyle doğrulandı. ~500 MB ONNX + 156 MB MNN.
  Gerekirse `EXPORT_FULL_CLIP=1`.
* `clip_v2.onnx` MNN'e çevrildikten sonra siliniyor (~500 MB).
* `FREE_FP_DLC=1` (config.env'de açık): kuantize DLC hazır olunca fp32 DLC
  siliniyor (~3.4 GB). Bedeli: sonraki koşuda dönüşüm + kuantizasyon baştan.
* `disk_report` — adım 0/2/4 öncesi-sonrası boş alan ve en büyük klasörler
  yazdırılıyor, bir daha körlemesine tahmin etmeyelim.

## ÇALIŞTI — unet.bin v68 için üretildi (v18)

```
[context-bin] model_q.dlc -> unet.bin (dsp_arch=v68)
====== DDR bandwidth summary ======
[+] work/CyberRealistic/qnn/unet.bin
```

`/unet/time_proj/...` hatası gitti; Gather çözümü tuttu. Adım 6c'nin okuduğu
gerçek tipler:

```
graf: model
  girdi text_embedding   QNN_DATATYPE_UFIXED_POINT_16  [1, 77, 768]
  girdi timestamp        QNN_DATATYPE_INT_32           [1]
  girdi sample           QNN_DATATYPE_UFIXED_POINT_16  [1, 4, 64, 64]
  cikti output           QNN_DATATYPE_UFIXED_POINT_16  [1, 4, 64, 64]
```

Yani motorun `QnnModel.hpp`'de yazdığı tiplerin **tamamı tutuyor**. VAE decoder
ve encoder de sorunsuz derlendi. Paket: `dist/CyberRealistic_qnn2.40_min.zip`
(unet.bin 830 MB; referans 893 MB).

### check_bin_io.py hatası (düzeltildi)

Script "TIP UYUSMAZLIGI" yazdı ama listelediği değerler beklenenle aynıydı —
karşılaştırma hatalıydı: araç `QNN_DATATYPE_UFIXED_POINT_16` döndürüyor,
`EXPECT` ise ön eksiz `UFIXED_POINT_16` tutuyor. Karşılaştırmadan önce
`QNN_DATATYPE_` öneki kırpılıyor artık. Yanlış alarmdı, paket sağlam.

### Sırada

Bu paket `FAST_TRIAL=1` ile üretildi (1 prompt x 2 adım kalibrasyon) — cihazda
yüklenmesi beklenir ama **görüntü kalitesi düşüktür**. Yükleme doğrulanınca
`OVERRIDE_FAST_TRIAL=0` ile bir kez daha koşulmalı.

## Cihazda yüklenmedi — sebep: SDK sürümü (2.40 ≠ 2.39)

Paket telefonda reddedildi:

```
[WARNING] QnnDsp <W> This META does not have Alloc2 Support
[ INFO ] QnnDsp <I> QnnDevice_free done. status 0x0
1940.6ms [ ERROR ] Could not free context
Hata: Motor süreci beklenmedik şekilde kapandı (kod 1).
```

Sebep tipler değil (6c zaten UFIXED_POINT_16 / INT_32 gösteriyordu, o uyarı
scriptin karşılaştırma hatasıydı). Sebep **QNN sürümü**. local-dream'in
`app/src/main/cpp/CMakeLists.txt` dosyası:

```cmake
# QNN SDK PATH
set(QNN_SDK_ROOT /data/qairt/2.39.0.250926)
...
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ...)
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so ...)
```

Yani uygulama QNN runtime'ını **2.39**'dan alıp APK'ya gömüyor. Context
binary'ler **geriye** uyumludur, **ileriye değildir**:

| binary | runtime | sonuç |
|---|---|---|
| 2.28 (referans paket) | 2.39 | çalışıyor ✓ |
| **2.40 (bizim paket)** | 2.39 | **reddediliyor ✗** |

Kaynakta bu uyumluluk açıkça düşünülmüş — `QnnModel.hpp`, eski binary'leri
tanıyor: *"old (pre-2.35) context binaries report spillFillBufferSize == 0 in
their metadata"*. Yani eskiyi yüklemeye hazır, yeniyi değil.

Yapılanlar:
* `config.env` → 2.39.0.250926.
* `03_convert_qnn.sh`: önbellek imzalarına `SDK_TAG` eklendi. SDK sürümü
  değişince DLC'ler ve `.bin` bayat sayılıp yeniden üretiliyor — yoksa 2.40 ile
  üretilmiş DLC'den 2.39 binary'si çıkarmaya çalışırdık.
* `probe_qairt_versions.py`: aday listesi uygulamanın sürümü etrafında yeniden
  sıralandı (2.39 ve daha eskiler öncelikli).

**Genel kural:** dönüşümde kullanılacak QAIRT sürümü, uygulamanın derlendiği
QAIRT sürümüne eşit ya da ondan eski olmalı.

## 2.39 ile de yüklenmedi — sürüm tek başına sebep değilmiş

`CyberRealistic_qnn2.39_min` üretildi (yani tarif 2.39'da da sorunsuz derleniyor
— 2.40'a geçmemize aslında gerek yokmuş) ama cihazda **birebir aynı** hata:

```
[WARNING] QnnDsp <W> This META does not have Alloc2 Support
[ INFO ] QnnDsp <I> QnnDevice_free done. status 0x0
2096.9ms [ ERROR ] Could not free context
Hata: Motor süreci beklenmedik şekilde kapandı (kod 1).
```

Bu üçüncü kez aynı kuyruk: düz `a8w8` paketinde, 2.40 paketinde, şimdi 2.39
paketinde. Kullanıcı ayrıca **Snapdragon 8 Gen 1 için dönüştürülmüş bir modelin
de aynı hatayı verdiğini** söylemişti. Yani bu mesaj bir teşhis değil, sadece
sürecin ölürken bastığı son satır — asıl hata `Son çıktı` penceresinin
üstünde kalıyor ve göremiyoruz.

Not: `main.cpp` bir `--log_level <n>` seçeneği kabul ediyor (varsayılan
`QNN_LOG_LEVEL_ERROR`). Ruya bunu geçirebilirse tam hata görülebilir.

### Yaklaşım değişikliği: referansla alan alan karşılaştır

Tahmin etmeyi bırakıyoruz. Yeni adım **6d** (`COMPARE_REF=1`), HuggingFace
`xororz/sd-qnn` deposundan **çalışan** bir `_min` paketi indirip `unet.bin`
metadata'sını bizimkiyle karşılaştırıyor:

* `dsp_arch`, VTCM boyutu, optimizasyon seviyesi
* graf adı, tensör sırası/isimleri/tipleri/kuantizasyon parametreleri
* spill-fill buffer boyutu, backend build id

Şüphelendiğim yer VTCM: `gen_htp_config.py` yalnızca `dsp_arch` yazıyor, VTCM'yi
varsayılana bırakıyor. v68 varsayılanı 8 MB olabilir; Snapdragon 7 Gen 1'in
VTCM'si daha küçükse bağlam cihazda kurulamaz — ve bu, "8 Gen 1 için üretilmiş
model de aynı hatayı veriyor" gözlemini birebir açıklar. Ama artık tahminle
değil, referansın ne yazdığını okuyarak karar vereceğiz.

## ASIL SEBEP BULUNDU: girdi sırası ters (6d karşılaştırması)

Referans `QteaMix_qnn2.28_min` indirilip metadata karşılaştırıldı. 63 alan aynı,
12 alan farklı — ve farkların çoğu tek bir şeyden:

```
info.graphs[0].info.graphInputs[0].info.name
    referans: 'sample'          [1,4,64,64]  rank 4
    bizim   : 'text_embedding'  [1,77,768]   rank 3
info.graphs[0].info.graphInputs[2].info.name
    referans: 'text_embedding'
    bizim   : 'sample'
```

Motor (`QnnModel.hpp::executeUnetGraphs`) tensörleri **isimle değil indeksle**
yazıyor:

```cpp
inputs[0] -> latents         16384 uint16
inputs[1] -> timestep        int32
inputs[2] -> text_embedding  59136 uint16
```

Bizim binary'de `inputs[2]` = `sample` (16384 elemanlık tampon). Motor oraya
59136 eleman yazıyor → **tampon taşması** → süreç `kod 1` ile ölüyor.
"Could not free context" tam olarak bunun kuyruğu.

Bu, tüm gizemi açıklıyor: düz `a8w8` paketi, 2.40 paketi ve 2.39 paketi hep
aynı hatayı verdi çünkü **üçünde de sıra tersti**. Kuantizasyonla, SDK
sürümüyle, VTCM ile ilgisi yokmuş. (VTCM şüphesi de çürüdü — o alanlar birebir
aynı.)

İkinci gerçek fark: `optimizationLevel` referansta **3**, bizde **0**.

Kalan farklar zararsız ve sürümden geliyor: `backendApiVersion` 5.28→5.39,
`contextBlobVersion` 3.2.0→3.3.3, `coreApiVersion` 2.21→2.29, ayrıca 2.39'un
eklediği `graphBlobInfoV2` / `platformInfo` alanları.

### Düzeltme

ONNX'te sıra DOĞRU (`sample, timestamp, text_embedding`) olmasına rağmen binary
ters çıkıyor — yani sırayı qairt araçları değiştiriyor. Hangi kurala göre
olduğunu tahmin etmek yerine **ölçüp tersini uyguluyoruz**:

* `scripts/fix_input_order.py` — binary'nin gerçek sırasını okur, ONNX sırasıyla
  karşılaştırıp araçların uyguladığı pozisyon permütasyonunu (ölçülen: `[2,1,0]`)
  çıkarır, tersini ONNX'e uygular ve çıkış kodu 2 ile "yeniden üret" der.
* `convert_all.sh` adım **4b** — kodu 2 alırsa io-config'i yeniler ve UNet'i
  `FORCE=1` ile bir kez yeniden üretir, sonra tekrar doğrular. Düzelmezse
  koşuyu hata ile durdurur (bozuk paket üretilmez).
* `gen_io_config.py --onnx` — YAML'daki girdi sırası artık ONNX'ten okunuyor,
  tek kaynak.
* `check_bin_io.py` — adım 6c artık tiplerin yanı sıra **sırayı** da denetliyor.
* `gen_htp_config.py --graph` — `graphs[{graph_names, O}]` bloğu eklendi,
  `O=3` (referansla aynı).

## Sıra ONNX'ten gelmiyor — ölçüldü

Adım 4b çalıştı, permütasyonu ölçtü, ONNX girdi sırasını tersine çevirdi ve
UNet'i yeniden üretti. Sonuç:

```
1. üretim:  onnx [sample, timestamp, text_embedding]
            binary [text_embedding, timestamp, sample]
2. üretim:  onnx [text_embedding, timestamp, sample]     <- ters cevrildi
            binary [text_embedding, timestamp, sample]   <- DEGISMEDI
```

Yani binary'nin `graphInputs` sırası **ONNX'teki bildirim sırasından bağımsız**;
sabit. `--config` YAML'ındaki sıra da etkilemiyor (birinci üretimde YAML
kanonik, ikincide ters — binary aynı kaldı).

Elenen mekanizmalar:
1. ONNX `graph.input` sırası — etkisiz (ölçüldü)
2. `--config` YAML'daki girdi bloğu sırası — etkisiz (ölçüldü)

Kalan resmi aday: **`--source_model_input_shape` (-s)**. Yardım metni "the name
and dimension of **all the input buffers** to the network" diyor; dönüştürücünün
girdi listesini bu argümanlardan kurması makul. `scripts/onnx_input_shapes.py`
şekilleri ONNX'ten istenen sırada üretiyor, `03_convert_qnn.sh` bunları
`IO_ORDER` ile `-s` argümanlarına çeviriyor.

Olası açıklama: referans paket 2.28'in **eski** `qnn-onnx-converter → model.cpp
→ .so` yolundan üretilmiş olabilir; orada girdi sırası üretilen C++'taki
bildirim sırasıdır. Bizim yol DLC → `QnnSystemDlc_composeGraphs`, farklı.

Adım 4b artık ONNX'i permute etmeyi denemiyor (işe yaramadığı ölçüldü):
doğruluyor, başarısızsa **SDK bin envanterini ve
`qnn-context-binary-generator --help` çıktısını döküp** koşuyu durduruyor —
bozuk paket üretilmesin ve alternatif yolu bir tur kaybetmeden görelim.

Ayrıca bu koşuda doğrulandı: `optimizationLevel` artık `O=3` yazılıyor ve
2.28 SDK'sı Software Center'dan **inmiyor** (404); en eski erişilebilir sürüm
2.32.

## `-s` de etkisiz — sıra DLC yolunda sabit

`--source_model_input_shape` üçü de istenen sırada verildi; binary sırası yine
`[text_embedding, timestamp, sample]`. Üç mekanizma da ölçümle elendi:

| mekanizma | sonuç |
|---|---|
| ONNX `graph.input` sırası | etkisiz |
| `--config` YAML girdi sırası | etkisiz |
| `--source_model_input_shape` sırası | etkisiz |

### SDK envanteri: eski yol duruyor

Envanter kritik: 2.39 hâlâ **`qnn-onnx-converter`** ve
**`qnn-model-lib-generator`** içeriyor, ve `qnn-context-binary-generator`
`--model=<qnn_model_name.so>` girdisini kabul ediyor:

```
[ --model=<val> ]  Path to the <qnn_model_name.so> file containing a QNN network.
```

Yani referansın üretildiği yol bu SDK'da mevcut:
`ONNX → qnn-onnx-converter (model.cpp+bin) → qnn-model-lib-generator (.so) →
qnn-context-binary-generator --model`. O yolda girdi sırası üretilen C++'taki
bildirim sırasıdır, dolayısıyla kontrol edilebilir.

### Önce ucuz teşhis: kural ne?

Gerçek UNet ile her deneme ~4 dakika. `scripts/probe_input_order.py` aynı girdi
imzasına (`sample [1,4,64,64]`, `timestamp [1] int32`,
`text_embedding [1,77,768]`) sahip **oyuncak** modellerle saniyeler içinde dört
senaryoyu ölçüyor:

* **A** taban
* **B** bildirim sırası ters
* **C** isimler değiştirilmiş — gözlenen sıra azalan isim uzunluğuyla birebir
  örtüşüyor (`text_embedding`=14 > `timestamp`=9 > `sample`=6); latent'e en uzun
  ad verilirse sıra düzeliyor mu?
* **D** grafta tüketim sırası ters

**C önemli**, çünkü motor isimlere hiç bakmıyor — `QnnModel.hpp` yalnızca
`inputs[0..2]` indekslerini kullanıyor, isim araması yok. Kural isim uzunluğuysa
girdileri yeniden adlandırmak yeterli olur ve tüm boru hattı olduğu gibi kalır.
Değilse eski yola geçeriz.

`PROBE_ORDER=1` ile adım 0a olarak koşuyor. `COMPARE_REF` ve `PROBE_SDK`
kapatıldı (işlerini gördüler; 2.28 indirilemiyor, en eski erişilebilir 2.32).

## Yoklama v1 bilgi vermedi — oyuncak modeller float'tı

Dört senaryonun dördü de aynı yerde öldü, dolayısıyla sıra hakkında hiçbir şey
öğrenilemedi:

```
graph_prepare.cc:219::ERROR:could not create op: q::Gather
  Input 0: op=[ConvLayer.opt.bias_to_vtcm@Ff.ff.] output0=[...PlainFloat_TCM...]
  Input 2: op=[flat_from_vtcm@fi.Fi.]             output0=[...Int32...]
```

Sebep bende: oyuncak modelleri **kuantize etmedim**. HTP, verisi float olan bir
`Gather`'ı kabul etmiyor; gerçek UNet'te tablo 8-bit kuantize olduğu için orada
sorun çıkmıyor. Yoklama artık gerçek hattı birebir taklit ediyor:
`ONNX → qairt-converter → qairt-quantizer (a8w8, 2 örnek) → context binary`.

Ayrıca:
* Yoklama başarısız olursa `qnn-onnx-converter` ve `qnn-model-lib-generator`
  arayüzleri (`--help`) aynı koşuda dökülüyor — yedek plan için bir tur daha
  kaybetmeyelim.
* `PROBE_ONLY=1`: yoklamadan sonra koşu duruyor. Sıra çözülmeden tam dönüşüm
  zaten `4b`'de başarısız bitiyor; 4 dakikayı boşa harcamanın anlamı yok.

## KURAL BULUNDU: sıra = grafta ilk tüketim sırası

Yoklama v2 kesin sonuç verdi:

| senaryo | tüketim sırası | binary sırası |
|---|---|---|
| A | s, t, e | t, e, s |
| B | s, t, e (**bildirim ters**) | t, e, s — bildirim etkisiz |
| C | s, t, e (**adlar değişik**) | t, e, s — isim etkisiz |
| D | **e, t, s** | **e, t, s** — tüketim etkili |

A'nın sapması da açıklandı: orada `sample`'ın ilk tüketicisi `Identity`'ydi ve
dönüştürücü onu eliyor, `sample` en sondaki `Add`'e kayıyor. Elenmeyen bir op
(`Mul`) konunca A da kurala uyuyor.

**Kural: QNN graf girdi sırası = optimize grafta ilk tüketim sırası.**
Bildirim sırası, tensör adları ve `--source_model_input_shape` etkisiz — üçü de
ayrı ayrı ölçüldü.

### Düzeltme: ONNX düğümlerini topolojik olarak yeniden sırala

`scripts/reorder_onnx_nodes.py`, Kahn topolojik sıralamasını öncelikli kuyrukla
çalıştırıyor: her düğüme transitif olarak bağlı olduğu graf girdilerinin en
küçük öncelik değeri atanıyor, hazır düğümler arasından en düşük öncelikli
seçiliyor. Sonuç: `sample`'a bağlı zincir önce, `timestamp` sonra,
yalnızca `text_embedding`'e bağlı olanlar en son — bağımlılıklar bozulmadan.

Düğüm sırasını değiştirmek ONNX semantiğini değiştirmez; `graph.node` yalnızca
topolojik olarak geçerli olmak zorundadır.

Birim testi (diffusers UNet'in düğüm sırasını taklit eden sahte graf):

```
eski dugum sirasi : gather_t, lin1, conv_in, res1, attn2, out
eski ilk tuketim  : timestamp, sample, text_embedding
yeni dugum sirasi : conv_in, gather_t, lin1, res1, attn2, out
yeni ilk tuketim  : sample, timestamp, text_embedding   <- hedef
```

### Açık kalan soru

Gerçek UNet'te gözlenen sıra `[text_embedding, timestamp, sample]` idi, oysa
diffusers önce `time_proj`, sonra `conv_in`, en son attention çalıştırıyor —
yani beklenen `[timestamp, sample, text_embedding]`. Aradaki fark muhtemelen
oyuncakta kullanılmayan `--config` (uint16 sınır). Yoklama v3 bunu da ölçüyor:
E/F/G senaryoları aynı testleri `--config` **açıkken** tekrarlıyor.

Yedek plan için not: `qnn-onnx-converter` bu SDK'da var ama `pandas` eksikliği
yüzünden açılmıyor (`ModuleNotFoundError: No module named 'pandas'`) — gerekirse
`pip install pandas` yeter. `qnn-model-lib-generator` sorunsuz çalışıyor.

## QNN 2.28 indirme linki bulundu (farklı ürün yolu)

Mr-J-369 HuggingFace tartışmasından gelen yanıt, Local Dream rehberini
(`ld-guide.chino.icu/conversion/sd15`) ve çalışan indirme linkini verdi:

```
https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip
```

Bizim yoklamamızın 2.28'e 404 vermesinin sebebi anlaşıldı: **farklı ürün yolu**.

| ürün | yol | kapsam |
|---|---|---|
| Qualcomm AI Runtime Community | `softwarecenter.../Qualcomm_AI_Runtime_Community/All/{v}/v{v}.zip` | 2.3x ve üstü |
| Qualcomm Neural Processing SDK | `apigwx-aws.../qualcomm_neural_processing_sdk/v{v}.zip` | **eski sürümler, 2.28 dahil** |

`probe_qairt_versions.py` artık ikisini de deniyor. `config.env`'e 2.28 seçeneği
(yorumlu) eklendi. `setup_qnn_sdk.py`'nin `find_sdk_root` fonksiyonu zaten
`qnn-onnx-converter`'ı da tanıdığı için 2.28 sorunsuz algılanır.

**Ama önce 2.39 + düğüm yeniden sıralaması denenmeli.** Sebep: 2.28'de büyük
ihtimalle `qairt-converter` ve `qnn-context-binary-generator --dlc_path` yok;
o sürüme geçmek eski hattı (`qnn-onnx-converter → qnn-model-lib-generator →
--model <.so>`) baştan yazmayı gerektirir. Düğüm sıralaması tutarsa hiç gerek
kalmaz; tutmazsa zaten o hatta geçeceğiz ve 2.28 elimizde olacak.

## Resmi rehber elde edildi — yön değişiyor

`ld-guide.chino.icu/conversion/sd15` içeriği paylaşıldı. Kritik maddeler:

* **"Qualcomm AI Engine Direct SDK 2.28 — please use v2.28 to avoid potential
  issues."** Sürüm tercih değil, şart olarak yazılmış.
* **Resmi script paketi var: `npuconvertv2.zip`.**
* Akış:
  ```
  prepare_data.py -> gen_quant_data.py -> export_onnx.py
  -> scripts/convert_all.sh --min_soc min
  ```
  Çıktı `output/qnn_models_min/unet.bin`, paket `<ad>_qnn2.28_min.zip`.
* Tier'lar: `min` = Hexagon V68+ bayrak dışı çipler (bizim hedefimiz),
  `8gen1`, `8gen2`.
* "Conversion process is extremely slow — several hours per resolution per chip
  tier." Bizim hattımız kalibrasyonu küçülttüğü için dakikalar sürüyordu; bu
  fark kaliteye yansıyor olabilir.
* Ekstra çözünürlükler taban 512×512 UNet'e göre `zstd --patch-from` ile patch
  olarak paketleniyor — bizim `_min` paketimizde bu yok, gerekmiyor.

### Neden bu her şeyi değiştiriyor

Günlerdir tersine mühendislikle aradığımız şey — `unet.bin`'in hangi araç
zinciriyle üretildiği, dolayısıyla **girdi sırasının nasıl kontrol edildiği** —
`scripts/convert_all.sh` içinde zaten yazılı. Tahmin etmeyi bırakıp onu okumak
tek doğru adım.

`scripts/fetch_official_scripts.py` (adım **0b**, `FETCH_OFFICIAL=1`):
rehber sayfasından `npuconvert*.zip` linkini çıkarır (ya da
`OFFICIAL_SCRIPTS_URL` ile elle verilir), indirir, açar ve
`scripts/convert_all.sh`, `export_onnx.py`, `gen_quant_data.py`,
`prepare_data.py` dosyalarını **loga döker**. `FETCH_ONLY=1` ile koşu hemen
duruyor — saniyeler sürer.

Not: 2.28 linki de elimizde (bkz. bir önceki bölüm), ama önce scriptleri okuyup
hangi araçların gerektiğini görmek gerekiyor.

## 2.28'e geçildi + indirme kaldığı yerden devam

Soru geldi: "2.28'i mi 2.39'u mu kullanıyoruz?" — o ana kadar **2.39**. Artık
**2.28 aktif**. Gerekçe: rehber sürümü şart koşuyor ("please use v2.28 to avoid
potential issues"), okuyacağımız resmi scriptler o sürüm için yazılmış ve her
oturumda 1.3 GB'ı iki kez indirmenin anlamı yok.

```
OVERRIDE_QAIRT_ASSET_URL=https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip
OVERRIDE_QNN_VERSION=2.28
```

2.39 ve 2.40 satırları yorumlu olarak duruyor.

Ayrıca SDK indirmesi **%53'te koptu**. Colab'da her yeni çalışma zamanı SDK'yı
baştan indirmek demek olduğu için tek bir kopma pahalı. `setup_qnn_sdk.py`
artık HTTP `Range` ile **kaldığı yerden devam** ediyor ve üstel bekleyişle
5 kez deniyor; yarım kalan dosya silinmiyor, üstüne ekleniyor.

Uyarı: 2.28'de `qairt-converter` ve `qnn-context-binary-generator --dlc_path`
olmayabilir. O durumda bizim adım 4 hattımız çalışmaz ve resmi akışa
(`qnn-onnx-converter → qnn-model-lib-generator → --model <.so>`) geçeceğiz.
Bu koşuda `FETCH_ONLY=1` olduğu için dönüşüm zaten çalışmıyor.

## Yeni çalışma zamanında hücre sırası

`NameError: name 'DSP_ARCH' is not defined` geldi. Sebep: not defterinin en
üstündeki **"Dönüşüm ayarları"** form hücresi (cell 2) çalıştırılmamış. O hücre
`SAFETENSORS_URL`, `MODEL_NAME`, `TIER`, `RESOLUTIONS` ve `DSP_ARCH`,
`UNET_MODE`, `REBUILD_BIN`, `TARGET_SOC`, `QUANT_EXTRA` değişkenlerini
tanımlıyor; 6. adım hücresi (cell 21) bunları `os.environ`'a yazıyor. Çalışma
zamanı sıfırlanınca değişkenler kayboluyor.

**Doğru sıra:** `[Dönüşüm ayarları] → 2 → 4 → 5 → 6`

Seçilen değerler önemsiz — `config.env` hepsini eziyor. `convert_all.sh`'in
checkpoint uyarısı artık bu sırayı ve hatayı açıkça yazıyor.

## Resmi scriptler elde edildi — büyük bulgular

`npuconvertv2.zip` (7.3 MB, 62 dosya) indi. Beklediğimden çok daha fazlasını
açıklıyor.

### 1. `convert_all.sh` sadece bir sarmalayıcı

```bash
QNN_SDK_ROOT=/data/qairt/2.28.0.241029
cd $QNN_SDK_ROOT/bin && source envsetup.sh
bash scripts/convert_clip.sh
bash scripts/convert_vae_encoder.sh
bash scripts/convert_vae_decoder.sh
bash scripts/convert_unet.sh
```

Asıl araç zinciri **`scripts/convert_unet.sh`** içinde — ve o dosyayı dökme
listesine koymayı atlamışım. Liste güncellendi; zip zaten indirilmiş durumda,
bir sonraki koşu saniyeler sürecek.

### 2. ONNX üretimi bizimkinden yapısal olarak farklı

```python
from redefined_modules.diffusers.models.unet_2d_condition import UNet2DConditionModel
replace_mha_with_sha_blocks(unet)   # CrossAttention: Linear -> Conv
```

* **`redefined_modules/`** — diffusers'ın `unet_2d_condition.py`,
  `attention.py`, `embeddings.py`, `resnet.py`, `unet_2d_blocks.py`
  modüllerinin **değiştirilmiş kopyaları**. Yani ONNX'i HTP dostu yapan şey
  dönüştürücü bayrakları değil, **modelin kendisinin yeniden yazılması**.
* **MHA → SHA**: cross-attention'daki Linear katmanlar Conv'a çevriliyor.
  Bu, HTP'de çok daha verimli ve bizim `attn2/MatMul` sorunlarımızın da
  muhtemel açıklaması.
* `embeddings.py` değiştirilmiş — bizim günlerce uğraştığımız `time_proj`
  int32 sorununun oradaki çözümü ne, görmemiz lazım.

Girdi adları ve sırası **bizimkiyle aynı**:
`["sample", "timestamp", "text_embedding"]` → `["output"]`.
Fark: `timestamp` orada `torch.tensor([0], dtype=torch.long)`.

### 3. token_emb.bin fp16 (bizde fp32'ydi)

```python
token_embedding.weight.data.to(torch.float16).numpy().tofile("clip/token_emb.bin")
```

Referans paket doğruluyor: 49408×768×2 = **75.89 MB**; bizim çıktımız 144 MB
(fp32). `TextEncoder.hpp` ikisini de kabul ediyor ("SD1.5 token_emb may still
be legacy FP32, detected by file size"), yani bu bir yükleme hatası değildi —
ama resmi hatta hizalanıp paketi ~70 MB küçültmek için fp16'ya geçildi
(export v19).

### 4. Kalibrasyon: 400 örneğe kadar

`gen_quant_data.py` UNet için **400 örnek** kullanıyor (ve `|sample|>7.2` olan
örnekleri atıyor). Rehberdeki "several hours" bundan. Bizim `FAST_TRIAL=1`
ile 2 örnek kullanmamız kaliteyi düşürüyor — boru hattı çalışınca artırılmalı.

### 5. Kendi MNN dönüştürücülerini taşıyorlar

Zip'te `MNNConvert` ikili dosyası var; CLIP dönüşümü onunla yapılıyor.

## RESMİ TARİF ÇÖZÜLDÜ — `convert_unet.sh`

```bash
qnn-onnx-converter -n --input_network ./unet/model.onnx \
    --preserve_io layout --input_list ./input_list_unet.txt \
    --use_per_channel_quantization --bias_bitwidth 32 --act_bitwidth 16
qnn-model-lib-generator -c ./unet/model.cpp -b ./unet/model.bin \
    -t x86_64-linux-clang -o ./qnn_unet
qnn-context-binary-generator --model ./qnn_unet/x86_64-linux-clang/libmodel.so \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --output_dir output/qnn_models_min --binary_file unet \
    --config_file ./htp_backend_min.json
```

Bizim yaptığımız her yanlışın karşılığı burada:

| bizim | resmi | sonuç |
|---|---|---|
| `qairt-converter` → DLC → `--dlc_path` | `qnn-onnx-converter` → `model.cpp` → `.so` → `--model` | **girdi sırası** ancak `.so` yolunda kontrol edilebiliyor (model.cpp'deki bildirim sırası) |
| `--act_bitwidth 8` + `--config` io16 hilesi | **`--act_bitwidth 16`** | uint16 sınır doğal olarak oluşuyor; io-config'e hiç gerek yok |
| `PER_CHANNEL=0` (Conv hatası yüzünden kapattık) | **`--use_per_channel_quantization`** | o hata 2.39'a ve bizim ONNX'imize özgüydü |
| stok diffusers | **`redefined_modules/`** | ONNX'i HTP'ye uygun yapan şey bayraklar değil, modelin yeniden yazılması (CrossAttention Linear→Conv, MHA→SHA) |
| VTCM ayarsız | **`"vtcm_mb": 2`** | Snapdragon 7 Gen 1'in VTCM'si küçük; ilk şüphem doğruymuş |
| — | `perf_profile: burst`, `rpc_control_latency: 100`, `O: 3.0`, `hvx_threads: 0` | |

`a16w8`'i "v68'de 16-bit LayerNorm yok" diye elemiştik — o hata **qairt-converter
2.39 + bizim ONNX'imize** özgüymüş. Resmi hat 2.28 + yeniden yazılmış modüllerle
tam 16-bit aktivasyon kullanıyor ve çalışıyor.

### Yeni hat: `scripts/06_official_pipeline.sh`

`USE_OFFICIAL=1` ile devreye giriyor ve resmi scriptleri Colab'da koşuyor:

1. `uv venv -p 3.10 && uv sync` — resmi `pyproject.toml` (diffusers 0.31.0,
   transformers 4.46.1, numpy 1.26.4, onnx, pandas, pyyaml). Bu venv aynı
   zamanda QNN 2.28 araçlarının Python'u oluyor.
2. `prepare_data.py` → `gen_quant_data.py` → `export_onnx.py`
3. `CALIB_LIMIT` ile kalibrasyon listesi kırpılıyor (resmi 400; ilk denemede
   24 ile boru hattı doğrulanır, sonra 0 = tam kalite)
4. `scripts/convert_all.sh --min_soc min` (SDK yolu bizimkine `sed`'lenir)
5. Çıktı `output/qnn_models_min/` → `dist/<ad>_qnn2.28_min.zip` (dosyalar zip
   kökünde; referans paketler böyle)

Kendi hattımız (adım 0-7) yerinde duruyor; `USE_OFFICIAL=0` ile geri dönülür.

## Resmi hat ilk deneme: yol hatası (düzeltildi)

```
06_official_pipeline.sh: line 79: work/CyberRealistic/_official/npuconvertv2/.venv/bin/python:
  No such file or directory
```

Benim hatam: `$SRC` göreli bir yoldu ve script `cd "$SRC"` yaptıktan sonra
`$SRC/.venv/bin/python` artık `$SRC/$SRC/.venv/...` anlamına geliyordu.
Düzeltmeler:

* `SRC` ve `DIST` en başta mutlaklaştırılıyor.
* Venv oluşmazsa script açık mesajla duruyor (sessizce devam edip ikinci bir
  hatayla ölmüyordu).
* `VIRTUAL_ENV` de ayarlanıyor — QNN 2.28 araçları `python3`'ü PATH'ten
  bulacak.
* Checkpoint kontrolü eklendi: resmi hat `.safetensors`'ı **doğrudan**
  kullanıyor (bizim `pipeline/` klasörünü değil), o yüzden `KEEP_CKPT=1`
  yapıldı — kendi hattımızın "pipeline hazır, checkpoint'i sil" davranışı
  resmi hatta zarar verirdi.
* `FETCH_OFFICIAL=0`: `USE_OFFICIAL` zaten eksikse indiriyor (döküm yapmadan),
  her koşuda script içeriklerini basmaya gerek yok.

## Yeni not defteri: resmi hat + CUDA (SD15_NPU_Official_Colab.ipynb)

Eski not defteri kendi hattımız (adım 0-7, `qairt-converter` → DLC) için
yazılmıştı ve artık kullanmadığımız onlarca seçenek taşıyor. Yeni defter
yalnızca resmi hattı koşuyor, 5 adım:

1. **Ayarlar** — `SAFETENSORS_URL`, `MODEL_NAME`, `SOC`, `CLIP_SKIP`,
   `REALISTIC`, `CALIB_LIMIT`, `CUDA_TORCH`
2. **Depo + uv**
3. **QNN SDK 2.28** (kaldığı yerden devam eden indirme)
4. **Modeli indir** (`.safetensors` doğrudan kullanılıyor)
5. **Dönüştür** → `scripts/06_official_pipeline.sh`
6. **Paketi indir**

`config.env` bu defterde devrede değil — form değerleri doğrudan ortam
değişkeni olarak veriliyor. Eski defter ve `config.env` duruyor.

### CUDA torch

Resmi `pyproject.toml` torch'un **CPU** sürümünü sabitliyor
(`torch==2.5.1+cpu` + `whl/cpu` indeksi), bu yüzden GPU'lu bir çalışma
zamanında bile difüzyon CPU'da koşuyordu (~3.85 sn/adım). Rehber de bunu
söylüyor: *"If you have a CUDA-capable GPU, you can edit pyproject.toml to use
the GPU build of torch."*

`06_official_pipeline.sh` artık `CUDA_TORCH` (varsayılan `auto`, `nvidia-smi`
ile tespit) ile pyproject'i CUDA sürümüne çeviriyor (`CUDA_WHL`, varsayılan
`cu121`) ve pin değiştiği için `.venv`/`uv.lock`'u yeniliyor.

**Ölçü:** GPU yalnızca `prepare_data.py`'yi hızlandırır (~35 dk → ~3 dk).
`qnn-onnx-converter` kuantizasyonu, `qnn-model-lib-generator` ve
`qnn-context-binary-generator` **tamamen CPU**'dur; rehberdeki "saatler"in
büyük kısmı orada ve GPU bunu değiştirmez. Yüksek bellekli çalışma zamanı ise
şart (rehber: ~20 GB).

Ayrıca `06_official_pipeline.sh` resmi scriptleri eksikse **kendisi indiriyor**,
böylece yeni defterde ayrı bir adım gerekmiyor.

## Resmi hat: her şey geçti, tek engel çalıştırma izni

İlk tam koşuda **tüm ağır adımlar başarılı** oldu:

* `prepare_data.py` — 20 prompt × difüzyon, ~35 dk, `images/1..20.png` üretildi
* `gen_quant_data.py` — **924 geçerli UNet örneği**, 400'e indirildi,
  `CALIB_LIMIT=24` ile 24'e kırpıldı
* `export_onnx.py` — `redefined_modules` ile UNet/CLIP/VAE ONNX'leri
* QNN 2.28 ortamı — `[INFO] AISW SDK environment set`, `QNN_SDK_ROOT` doğru

Tek hata:

```
scripts/convert_clip.sh: line 6: ./MNNConvert: Permission denied
```

ZIP çalıştırma bitlerini korumuyor; pakette gelen `MNNConvert` ikilisi ve `.sh`
scriptleri bu yüzden çalışmıyor. (QNN SDK'da da aynısını yaşamış ve
`make_bins_executable` ile çözmüştük.)

Düzeltme iki yerde:
* `fetch_official_scripts.py` — açtıktan sonra `MNNConvert` ve `*.sh` için
  `chmod +x`
* `06_official_pipeline.sh` — `cd "$SRC"` sonrası aynı `chmod` (zaten açılmış
  paketler için; yeniden indirmeye gerek kalmasın)

`data.pkl`, `images/` ve `unet/model.onnx` önbellekte olduğu için koşu
tekrarlandığında doğrudan 4. adımdan (QNN dönüşümü) devam eder — 35 dakika
yeniden ödenmez.

## GPU koşusu: CUDA çalıştı, MNN düzeldi, sırada libc++

Yeni defter + A100 ile:

* `[cuda] GPU bulundu -> pyproject.toml CUDA torch'a cevriliyor`
* `torch==2.5.1+cu121` kuruldu, `[torch] 2.5.1+cu121 cuda: True NVIDIA A100-SXM4-40GB`
* `prepare_data.py` **adım süresi basılmayacak kadar hızlı** bitti
  (CPU'da 3.85 sn/adım × ~500 adım ≈ 35 dk idi)
* `gen_quant_data.py` → 924 örnek, `CALIB_LIMIT=150` ile 150'ye kırpıldı
* `export_onnx.py` sorunsuz
* **MNNConvert düzeldi** — `Converted Success!`, `clip_v2.mnn` üretildi

Yeni hata `qnn-onnx-converter`'da:

```
ImportError: cannot import name 'libPyIrGraph' ...
ImportError: libc++.so.1: cannot open shared object file: No such file or directory
```

QNN 2.28'in Python bağlantıları LLVM **libc++**'a bağlı; Colab imajında yok ve
SDK kendi kopyasını taşımıyor. `06_official_pipeline.sh` artık adım 3b'de
`ldconfig` ile kontrol edip eksikse `libc++1 libc++abi1` (yedek:
`libc++1-14 libc++abi1-14`) kuruyor.

Kalan zincir: `qnn-onnx-converter` → `qnn-model-lib-generator` (C++ derlemesi)
→ `qnn-context-binary-generator`. Sonraki muhtemel eksikler derleyici/`make`
tarafında olabilir; Colab'da `build-essential` kurulu olduğu için sorun
beklemiyorum.
