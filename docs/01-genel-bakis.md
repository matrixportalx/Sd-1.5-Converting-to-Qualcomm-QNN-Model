# 01 — Genel Bakış

## Bu toolkit ne işe yarar?

civitai.red / Hugging Face üzerindeki **Stable Diffusion 1.5** tabanlı
`.safetensors` modellerini, **Ruya / Local Dream** uygulamasının Snapdragon
NPU'sunda çalıştırdığı **`<isim>_qnn2.28_min.zip`** paketine dönüştürür.

Hedef: Snapdragon 8 için yapılmış dönüşümleri kullanamayan **Snapdragon 7**
(ve diğer "flagship olmayan") cihazlarda güncel modelleri çalıştırabilmek.

## Local Dream nasıl çalışır? (kısa)

Stable Diffusion üç ana bileşenden oluşur:

1. **Text Encoder (CLIP):** Prompt metnini sayısal gömme vektörüne çevirir.
2. **UNet:** Asıl "difüzyon" işini yapan büyük ağ. Gürültüyü adım adım azaltır.
   Hesaplama yükünün ~%90'ı buradadır.
3. **VAE:** UNet'in ürettiği "latent" görüntüyü gerçek piksellere çözer.

Local Dream bunları **iki farklı motora** dağıtır:

- **UNet → QNN (NPU/Hexagon):** En ağır kısım donanım hızlandırmalı NPU'da.
  Bu, mimariye (v68/v69/v73/...) duyarlı olan ve `_min` / `_8gen3` ayrımını
  doğuran kısımdır.
- **Text Encoder + VAE → MNN (CPU/GPU):** Görece hafif olduklarından Alibaba'nın
  MNN motorunda çalışırlar; mimariden bağımsızdır.

## `_min` ne demek?

`_min` = **minimum HTP mimarisi hedefi = maksimum cihaz uyumluluğu**.

UNet context binary'si düşük bir DSP mimarisine (v68) derlendiğinde, o sürümden
yukarı **tüm** Snapdragon cihazlarında (7 Gen 1, 8 Gen 1, 8 Gen 2, 8 Gen 3 ...)
çalışır. Bu yüzden Snapdragon 7 kullanıcıları her zaman `_min` varyantını indirir.

Ayrıntı için [`04-soc-htp-tablosu.md`](04-soc-htp-tablosu.md).

## Dönüşüm hattı (pipeline)

```
.safetensors
   │  prepare_data.py — 20 prompt x difuzyon (kalibrasyon girdileri)
   ▼
data.pkl + input_list_*.txt
   │  export_onnx.py — redefined_modules ile ONNX (sabit sekil)
   ▼
onnx/  ── text_encoder ──► clip_v2.mnn        (MNN, CPU/GPU)
   │     ── vae ─────────► vae_*.bin          (QNN, fp16)
   └───── unet ──────────► unet.bin           (QNN, act_bitwidth 16)
                                   │
              ek cozunurluk varsa: zstd --patch-from ──► *.patch
                                   │
                                   ▼
                    <isim>_qnn2.28_<soc>.zip
```
