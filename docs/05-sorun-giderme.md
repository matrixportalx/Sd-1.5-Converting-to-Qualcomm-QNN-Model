# 05 — Sorun Giderme

## "QNN_SDK_ROOT ayarlı değil"
`export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029` yapın ve
`ls $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter` ile doğrulayın.

## `qnn-onnx-converter: command not found`
SDK ortamını yükleyin: `source $QNN_SDK_ROOT/bin/envsetup.sh`. Bu, PATH ve
PYTHONPATH'i ayarlar. Ayrıca converter'ın Python bağımlılıklarını kurun
(`check-python-dependency`).

## Bellek yetersiz (OOM) / süreç öldürülüyor
- Önce yalnızca `512x512` çözünürlüğünü dönüştürün (`--resolutions 512x512`).
- Swap ekleyin (bkz. `02-gereksinimler.md`).
- Aynı anda tek çözünürlük/tier işleyin.

## Dönüşüm çok yavaş
Normaldir. INT8/INT16 kuantizasyonu CPU'da saatler sürebilir. Bir kez üretip
ZIP'i saklayın; tekrar dönüştürmeniz gerekmez.

## `from_single_file` hata veriyor / config bulunamıyor
- `pip install omegaconf` kurulu olmalı.
- Model gerçekten **SD1.5** mimarisinde mi? SDXL/SD2.x/Pony-XL bu hattı
  kullanamaz (SDXL için Local Dream yalnızca 8 Gen 3+ destekler).
- Bozuk/kısmi indirilmiş `.safetensors` olabilir; yeniden indirin.

## MNNConvert çalışmıyor
`-DMNN_BUILD_CONVERTER=ON` ile derlediğinizden ve `MNNCONVERT` değişkeninin
`build/MNNConvert` dosyasını gösterdiğinden emin olun.

## Telefonda "Import" başarısız / model açılmıyor
En olası sebep: ZIP içindeki **dosya adları veya `model_info.json` düzeni**
uygulamanın sürümüyle birebir uyuşmuyor. Çözüm:

1. Halihazırda **çalışan** bir resmi `*_qnn2.28_min.zip` indirin
   (ör. Hugging Face `Mr-J-369/*-SD1.5-qnn2.28` veya uygulama içi Absolute Reality).
2. İçini açın; dosya adlarını (`unet_*.bin`, `text_encoder.mnn`, `vae.mnn`,
   `model_info.json`) ve klasör düzenini not edin.
3. `scripts/05_package.py` çıktısını bu düzene **birebir** eşitleyin (gerekirse
   dosya adlarını / `model_info.json` alanlarını script içinde düzenleyin).

## Görüntü bozuk / gürültülü çıkıyor (NPU'da)
Kuantizasyon kalitesi düşük olabilir:
- Kalibrasyonu `--mode real` ve daha fazla örnekle çalıştırın (`--num-samples 12`).
- UNet için 16-bit aktivasyon deneyin: `ACT_BW=16 WEIGHT_BW=8`.
- Bazı modeller agresif kuantizasyona diğerlerinden hassastır; farklı bir SD1.5
  checkpoint deneyin.

## NPU yerine CPU'da çalışıyor gibi (yavaş)
- Cihazınızın HTP mimarisi seçtiğiniz tier'den **yukarı** olmalı. Snapdragon 7
  için `min` (v68) kullanın; `high` (v75) Snapdragon 7'de NPU'da **çalışmaz**.
- Uygulamada NPU modunun seçili olduğundan emin olun.

## Yardımcı olabilecek kesin kaynak
Resmi dönüştürme kılavuzu (Çince, tarayıcı çevirisiyle okunur):
<https://ld-guide.chino.icu/zh/conversion/sd15>
