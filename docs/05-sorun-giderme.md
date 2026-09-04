# 05 — Sorun Giderme

## "QNN_SDK_ROOT ayarlı değil"
`export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029` yapın ve
`ls $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter` ile doğrulayın.

## `Python 3.12 is unsupported` / `libc++.so.1: cannot open shared object file`
QAIRT/QNN python konvertörleri **Python 3.10** ve **libc++** ister. Çözüm:
```bash
export QNN_SDK_ROOT=...     # setup_qnn_sdk.py çıktısı
bash scripts/setup_qnn_python.sh          # 3.10 venv + libc++ kurar
export QNN_PYTHON="$(cat /content/qnn_py.path)"   # veya venv/bin/python yolu
```
`03_convert_unet_qnn.sh` `QNN_PYTHON` env'ini (veya `/content/qnn_py.path`
dosyasını) otomatik kullanır. Colab notebook'ta 4. adım bunu otomatik yapar.

## `Permission denied` (qairt-converter vb.)
ZIP'ten açılan araçlar çalıştırma bitini kaybetmiş. `setup_qnn_sdk.py` bunu
otomatik düzeltir; elle: `chmod -R +x "$QNN_SDK_ROOT/bin"`.

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

## Çözünürlük listesinde yalnızca 512×512 görünüyor
Paketin içinde `*.patch` dosyası yoktur. Uygulama listeyi tam olarak bu
dosyalardan çıkarır:

```bash
unzip -l dist/<isim>_qnn2.28_<soc>.zip | grep '\.patch'
```

Boşsa `RESOLUTIONS` verilmeden koşulmuş demektir:

```bash
RESOLUTIONS="512x768,768x512,768x768" bash scripts/06_official_pipeline.sh ...
```

Taban binary hazır olduğu için o tur atlanır; yalnızca eksik boyutlar koşar.
Ayrıntı: [`07-cozunurlukler.md`](07-cozunurlukler.md)

## `HATA: zstd YOK — ek cozunurluk yamasi uretilemez`
Yamalar `zstd --patch-from` ile üretilir. Hat bunu döngünün **önünde** kurmayı
dener (`apt-get install -y zstd`); root olmayan bir ortamda elle kurun.
Kontrol dönüşümün başında yapılır — saatler süren kuantizasyondan sonra eksik
araç yüzünden durulmaz.

## `GPU VAR ama torch ... CUDA goremiyor`
`prepare_data` CPU'da koşuyor demektir: çözünürlük başına ~3 dk yerine ~35 dk.

Sebebi resmi `pyproject.toml`'un torch'u **CPU sürümüne sabitlemesidir**
(`torch==2.5.1+cpu`). Üstüne `torch==2.5.1` istemek işe yaramaz: PEP 440'a göre
yerel etiketsiz `==2.5.1`, kurulu `2.5.1+cpu` tarafından **karşılanır** ve
uv/pip "already satisfied" deyip geçer. Hat bunu `--reinstall-package torch`
ile aşar ve sonucu yazıya değil **ölçüme** göre raporlar.

Yine de CUDA görünmüyorsa farklı bir tekerlek deneyin:

```bash
CUDA_WHL=cu124 bash scripts/06_official_pipeline.sh ...   # ya da cu118
```

Not: GPU **yalnızca** `prepare_data`'yı hızlandırır. Kuantizasyon, model-lib ve
context-binary tamamen CPU'dur; onların kısıtı RAM'dir. Ücretsiz Colab'da GPU
seçmek RAM'i düşürdüğü için kuantizasyonu OOM'a sokar.

## Ruya "bu boyut için yama yok" diyip başlamıyor
Beklenen davranış. Yamasız bir boyutta motor latent'i 96×96 üretir ama graf
64×64 bekler; çıktı sessizce renkli gürültü olurdu. Boyutu paketle birlikte
üretin ya da 512×512'de kalın.

## Yardımcı olabilecek kesin kaynak
Resmi dönüştürme kılavuzu (Çince, tarayıcı çevirisiyle okunur):
<https://ld-guide.chino.icu/zh/conversion/sd15>
