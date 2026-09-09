# 03 — Dönüşüm Adımları

Dönüşümün tamamını **tek script** yapar:

```bash
scripts/06_official_pipeline.sh <ckpt> <isim> <work_dir> [min|8gen1|8gen2]
```

Pratikte bu script elle değil, [Colab defterinden](../notebook/SD15_NPU.ipynb)
çalıştırılır — bkz. [`06-otomasyon.md`](06-otomasyon.md). Aşağıdaki bölüm,
scriptin içeride ne yaptığını ve bir adım takıldığında nereye bakılacağını
anlatır.

> Hat, Local Dream'in kendi `npuconvertv2` scriptlerini koşar; onları
> `scripts/fetch_official_scripts.py` indirir. Depoda kendi elle kurulmuş bir
> boru hattımız **yok**: denenmişti, ürettiği paketler cihazda yüklenmiyordu.
> Nedeni `06_official_pipeline.sh` başlığında yazılı (araç zinciri, `--act_bitwidth 16`
> ve HTP için yeniden yazılmış diffusers modülleri).

## Ön koşullar

```bash
export QNN_SDK_ROOT=...   # 2.28 OLMALI; scripts/setup_qnn_sdk.py çıktısı verir
```

SDK sürümü pazarlık konusu değil: resmi scriptler 2.28 araç zinciriyle test
edilmiş, paket adı da (`_qnn2.28_min`) buradan geliyor. Python tarafını script
kendi kuruyor (sistem python3.10 üzerinde bir `uv` venv'i) — elle bir şey
kurmanız gerekmez.

## Adımlar (script bunları çözünürlük başına koşar)

| # | Ne yapar | Süre / kısıt |
|---|----------|--------------|
| 1 | `prepare_data.py` — 20 prompt × difüzyon, kalibrasyon girdilerini toplar | **en uzun adım**; GPU'da ~3 dk, CPU'da ~35 dk |
| 2 | `gen_quant_data.py` — girdileri `data.pkl` + `input_list_*.txt` haline getirir | dakikalar |
| 3 | `export_onnx.py` — diffusers → ONNX (MHA→SHA, Linear→Conv) | dakikalar |
| 4 | resmi `convert_all.sh` (npuconvertv2) — CLIP → MNN, VAE + UNet → QNN context binary | **RAM kısıtı burada**: ~20 GB+ |
| 5 | Paketleme — `dist/<isim>_qnn2.28_<soc>.zip` | saniyeler |

Ek çözünürlük istendiyse 1–4 arası her çözünürlük için **yeniden** koşulur ve
çıkan `unet.bin`, taban 512×512 binary'sine karşı `zstd --patch-from` ile
farklanıp pakete `*.patch` olarak konur. Ayrıntı:
[`07-cozunurlukler.md`](07-cozunurlukler.md).

## Ayarlar (ortam değişkeni)

| Değişken | Varsayılan | Ne işe yarar |
|----------|-----------|--------------|
| `RESOLUTIONS` | `512x512` | Ek çözünürlükler; 512×512 her zaman üretilir |
| `CLIP_SKIP` | `2` | Modelin eğitildiği değer |
| `REALISTIC` | `1` | Foto-gerçekçi modeller için kalibrasyon promptlarını değiştirir |
| `CALIB_LIMIT` | `0` (=400 örnek) | Kırpar: `24` boru hattını doğrular, `150` iyi denge |
| `CACHE_REPO` + `HF_TOKEN` | boş | Pahalı adımları HF'e yedekler; kopan oturum kaldığı yerden devam eder |
| `CACHE_OUTPUT` | `1` | Çıktıyı (binary + yamalar) da yedekle |
| `CUDA_TORCH` / `CUDA_WHL` | `auto` / `cu121` | GPU'lu çalışma zamanında torch'un CUDA sürümünü kurar |

`CACHE_REPO` verildiğinde `prepare_data` çözünürlük başına **bir kez** ödenir;
sonraki oturumlar indirip atlar. Colab sekmesi kapanıp oturum düştüğünde asıl
kurtarıcı budur.

## Çıktı

```
dist/<isim>_qnn2.28_<soc>.zip
```

ZIP'in kökünde `unet.bin`, `vae_decoder.bin`, `vae_encoder.bin`, `clip_v2.mnn`,
`tokenizer.json`, `token_emb.bin`, `pos_emb.bin` ve varsa `*.patch` durur —
uygulamanın beklediği düz yapı budur, klasör yok.

## Telefona aktarma

1. ZIP'i (veya içindeki klasörü) telefona kopyalayın.
2. **Ruya / Local Dream → Settings → Import Custom Model** → klasörü/zip'i seçin.
3. Model listesinde görünür; NPU modunda üretin.

> İçe aktarma başarısız olursa [`05-sorun-giderme.md`](05-sorun-giderme.md)'ye bakın.
