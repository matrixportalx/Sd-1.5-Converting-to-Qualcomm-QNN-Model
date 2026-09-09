# 07 — Çözünürlükler ve yamalar

## Neden yama?

QNN context binary'sinde tensör şekilleri **derleme zamanında** sabitlenir.
`unet.bin` 512×512 için derlendiğinde latent 64×64'tür; 768×768 istenirse
motorun ürettiği latent 96×96 olur ve graf onu kabul etmez. Bu yüzden
çözünürlük bir istek parametresi değildir.

Local Dream'in çözümü (ve resmi `export.sh`'in yaptığı) şudur: her ek boyut
için UNet **yeniden** dönüştürülür, çıkan `unet.bin` taban 512×512
binary'sine karşı `zstd --patch-from` ile farklanır ve yalnızca fark paketin
içine konur. Uygulama açılışta yamayı bellekte uygular:

```
ruya-dream --type sd15npu --model_dir <model> --patch <model>/768.patch
```

İki sonucu var:

* Çözünürlük değiştirmek **motoru yeniden başlatmayı** gerektirir.
* Yaması olmayan bir boyut **seçilemez**. Ruya bu durumda üretime hiç
  başlamaz; yamasız 768 istenirse çıktı sessizce renkli gürültü olurdu.

MNN (CPU / `sd15cpu`) tarafında böyle bir kısıt yoktur — orada tensörler istek
başına yeniden boyutlandırılır.

## Adlandırma

Uygulamanın tarayıcısı iki kalıba bakar:

| Kalıp | Örnek | Anlamı |
|---|---|---|
| `<N>.patch` | `768.patch` | kare: 768×768 |
| `<W>x<H>.patch` | `512x768.patch` | dikdörtgen: 512×768 |

Hat bu kuralı uygular: kare boyutlar tek sayıyla, dikdörtgenler `WxH` ile
adlandırılır. Taban 512×512'nin yaması **yoktur** (binary'nin kendisidir).

## Kullanım

```bash
RESOLUTIONS="512x768,768x512,768x768" \
  bash scripts/06_official_pipeline.sh <ckpt> <isim> <work> 8gen2
```

* Ayırıcı: virgül, noktalı virgül veya boşluk. `512X768` de kabul edilir.
* `512x512` yazılırsa sessizce çıkarılır — zaten üretiliyor.
* Kenarlar **64'ün katı** olmalıdır (UNet latent'i 3 kez yarılar).
* Colab'da 1. adımdaki kutulardan seçilir.

## Bedeli ve devam edebilirlik

Her ek çözünürlük tam bir tur demektir: `prepare_data` (o boyutta difüzyon) →
`gen_quant_data` → `export_onnx_unet_only` → `convert_all_unet_only` → `zstd`.
CLIP ve VAE tekrar dönüştürülmez.

Hat, uzun koşuyu kurtaracak şekilde yazılmıştır:

| Durum | Davranış |
|---|---|
| Yaması pakette var | o çözünürlük **atlanır** |
| Taban `unet.bin` var | taban turu **atlanır** |
| `CACHE_REPO` dolu | `<slug>/res_<WxH>_cs<N>[_real]/` — çözünürlük başına kalibrasyon verisi |
| `CACHE_REPO` + `CACHE_OUTPUT=1` | `<slug>/out_<soc>_cs<N>[_real]_calib<N>/` — taban binary + birikmiş yamalar |

Yani kopan bir Colab oturumundan sonra aynı hücreyi tekrar çalıştırmak
yeterlidir; yalnızca eksik boyutlar koşar.

**Anahtar ayarları da içerir** (`cs<CLIP_SKIP>`, `_real` = `REALISTIC=1`,
çıktıda ayrıca `calib<CALIB_LIMIT>`). Sebebi: ayarı değiştirip aynı modeli
yeniden dönüştürdüğünüzde önbellek eski veriyi geri yükleyip adımı atlarsa
sonuç sessizce tutarsız olur — kalibrasyon verisi clip skip 2 ile üretilmişken
ONNX clip skip 1 ile dışa aktarılır, ya da `CALIB_LIMIT=0` istenen koşuda 24
ile üretilmiş `unet.bin` geri gelir. Ayar değiştirdiğinizde artık o adım
yeniden koşar; bu bir gecikme değil, doğru davranıştır.

2026-09-09'dan eski önbellek klasörlerinde ayar bilgisi yok. Hat bunlara
yalnızca `CLIP_SKIP=2` + `REALISTIC=1` kombinasyonunda geri düşer (o koşuların
ayarı buydu) ve düştüğünde log'a not basar. Çıktı tarafında ayrıca uyarır:
o paketin `CALIB_LIMIT`'i bilinemez.

Ara dosyalar her turun sonunda silinir (`unet/`, `qnn_unet/`, `output/`) —
tur başına ~7 GB, Colab diski buna dayanmaz.

## Sınırlar

* **`min` (v68):** resmi tarif ek çözünürlükleri yalnızca `8gen1`/`8gen2` için
  üretir — *"Non-flagship SOC versions can't run higher resolutions"*. `min`
  ile de yama üretilir, uyarı basılır, ama cihazda yüklenmeyebilir.
* **1024 kenar:** kuantizasyon RAM'i 512'ye göre kabaca 4×'tir ve cihaz
  tarafında VTCM'ye sığmayabilir. Önce 768 ile doğrulayın.
* **VAE yamalanmaz.** Yalnızca UNet. Yüksek çözünürlükte VAE tarafını
  uygulama kendi hallediyor.

## Doğrulama

Paket, gerçekte hangi boyutları desteklediğini kendi içinde taşır:

```bash
unzip -l dist/<isim>_qnn2.28_8gen2.zip | grep '\.patch'
```

`scripts/upload_hf.py` model kartındaki "Çözünürlük(ler)" satırını da tam
olarak bu dosyalardan çıkarır — elle bir liste tutulmaz.
