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
