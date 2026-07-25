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
