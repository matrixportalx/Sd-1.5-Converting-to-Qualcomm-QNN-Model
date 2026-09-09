# 05 — Sorun Giderme

## "QNN_SDK_ROOT ayarlı değil"
`export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029` yapın ve
`ls $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter` ile doğrulayın.

## `Python 3.12 is unsupported` / `libc++.so.1: cannot open shared object file`
QNN python konvertörleri **Python 3.10** ve **libc++** ister; Colab ise 3.12
kullanır. `06_official_pipeline.sh` ikisini de kendisi halleder: sistem
python3.10 üzerinde bir `uv` venv'i kurar ve `libc++1`/`libc++abi1` eksikse
apt'tan çeker. Elle bir şey kurmanız gerekmez.

Yine de bu hatayı görüyorsanız venv bozuk kurulmuş demektir; silin, hat
yeniden kurar:

```bash
rm -rf work/<model>/_official/npuconvertv2/.venv
```

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
3. Hattın ürettiği ZIP ile karşılaştırın. Paketleme
   `scripts/06_official_pipeline.sh` sonundaki "Paket" bölümünde yapılır;
   düzen orada tek yerde tanımlıdır.

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

## `libpython3.10.so.1.0: cannot open shared object file`
Tam hali, 4. adımda (QNN dönüşümü) şöyle görünür:

```
ImportError: cannot import name 'libPyIrGraph' from partially initialized
             module 'qti.aisw.converters.common' ...
ImportError: libpython3.10.so.1.0: cannot open shared object file
```

İlk satır örtücüdür, ikinci satır gerçek nedendir. SDK'nın C++ eklentisi
`libPyIrGraph.so` **sistem python3.10'una karşı** derlenmiştir; `DT_NEEDED`
listesinde `libpython3.10.so.1.0` vardır ve import sırasında onu arar.
`uv venv -p 3.10` sistemde 3.10 bulamazsa kendi python-build-standalone
yapısını indirir; o yapı paylaşımlı bir libpython sunmaz ve eklenti çözülemez.
(Dosya uv'nin dizininde bulunsa bile yetmez: dlopen edilen `.so`, DT_NEEDED
çözerken yorumlayıcının RUNPATH'ini miras almaz.)

Hat artık venv'i **sistem** python3.10'undan kurar
(`uv venv -p /usr/bin/python3.10 --python-preference only-system`) ve gerekirse
`python3.10 libpython3.10` paketlerini kendisi kurar. Eski, bozuk bir venv
elde kalmışsa bir kez temizleyin:

```bash
rm -rf work/<model>/_official/npuconvertv2/.venv
```

Sonraki koşuda ortam damgası (`v4`) zaten yeniden kurulmasını sağlar. Ayrıca
hat, `prepare_data`'ya girmeden önce `qnn-onnx-converter --help` ile bir ön
kontrol yapar; ortam bozuksa ~40 dakika sonra değil, ilk saniyelerde durur.

**Uyarı:** yalnızca `apt-get install libpython3.10` deyip uv'nin python'uyla
devam etmeyin. Statik libpython'lu bir yorumlayıcının içine ikinci bir
libpython yüklemek tek süreçte çift çalışma zamanı demektir; segfault olarak
geri döner.

## Yardımcı olabilecek kesin kaynak
Resmi dönüştürme kılavuzu (Çince, tarayıcı çevirisiyle okunur):
<https://ld-guide.chino.icu/zh/conversion/sd15>
