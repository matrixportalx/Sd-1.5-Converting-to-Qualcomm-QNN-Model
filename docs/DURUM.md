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
