# 03 — Dönüşüm Adımları (elle, tek tek)

`./convert_all.sh` hepsini otomatik yapar. Aşağıda her adımı **ayrı ayrı**
çalıştırmak isteyenler (veya bir adımı yeniden denemek isteyenler) için ayrıntı
var. Örnek model adı: `AbsoluteReality`, tier: `min` (Snapdragon 7).

Ön hazırlık:

```bash
source .venv/bin/activate
python scripts/setup_qnn_sdk.py --dest ./qairt   # release'ten indir
export QNN_SDK_ROOT="$(python scripts/setup_qnn_sdk.py --dest ./qairt | sed -n 's/^QNN_SDK_ROOT=//p' | tail -1)"
export MNNCONVERT=/opt/MNN/build/MNNConvert
NAME=AbsoluteReality
WORK=work/$NAME
```

## Adım 0 — safetensors → diffusers

```bash
python scripts/00_load_safetensors.py \
    --checkpoint /indirilenler/AbsoluteReality.safetensors \
    --output $WORK/pipeline
```

Sonuç: `$WORK/pipeline/{text_encoder,unet,vae,tokenizer,...}`.
Model **SD1.5** olmalı (SDXL/SD2.x değil).

## Adım 1 — diffusers → ONNX (sabit şekil)

```bash
python scripts/01_export_onnx.py \
    --pipeline $WORK/pipeline \
    --output $WORK/onnx \
    --resolutions 512x512,512x768,768x512
```

Sonuç: `text_encoder.onnx`, `vae_decoder.onnx`, `unet_512x512.onnx`, ...
NPU sabit şekil istediği için her çözünürlük ayrı UNet dosyasıdır.

## Adım 4 — text_encoder + VAE → MNN (CPU/GPU)

(UNet'ten bağımsız; önce yapmak pratik.)

```bash
./scripts/04_convert_mnn.sh $WORK/onnx $WORK/mnn
```

Sonuç: `$WORK/mnn/text_encoder.mnn`, `$WORK/mnn/vae.mnn`.
`FP16=0 ./scripts/04_convert_mnn.sh ...` ile fp16'yı kapatabilirsiniz.

## Adım 2 — UNet kalibrasyon verisi (her çözünürlük)

```bash
for TAG in 512x512 512x768 768x512; do
  python scripts/02_gen_quant_data.py \
      --pipeline $WORK/pipeline \
      --resolution $TAG \
      --output $WORK/calib/$TAG \
      --mode real --num-samples 6 --steps 15
done
```

`--mode real` gerçek difüzyon adımlarından girdi toplar (daha iyi kuantizasyon
kalitesi). Hız için `--mode random` kullanabilirsiniz ama kalite düşer.

## Adım 3 — UNet ONNX → QNN context binary (NPU)

```bash
for TAG in 512x512 512x768 768x512; do
  ./scripts/03_convert_unet_qnn.sh \
      $WORK/onnx/unet_${TAG}.onnx \
      $WORK/calib/${TAG}/input_list.txt \
      min \
      $WORK/qnn \
      $TAG
done
```

Sonuç: `$WORK/qnn/unet_512x512.bin`, ... (hedef `tier=min → v68`).
Kuantizasyon genişliğini değiştirmek için:
`ACT_BW=16 WEIGHT_BW=8 ./scripts/03_convert_unet_qnn.sh ...`

## Adım 5 — Paketle → `_qnn2.39_min.zip`

```bash
python scripts/05_package.py \
    --name $NAME --tier min \
    --qnn $WORK/qnn --mnn $WORK/mnn \
    --tokenizer $WORK/pipeline/tokenizer \
    --resolutions 512x512,512x768,768x512 \
    --output dist
```

Sonuç: `dist/AbsoluteReality_qnn2.39_min.zip`

## Telefona aktarma

1. ZIP'i (veya içindeki klasörü) telefona kopyalayın.
2. **Ruya / Local Dream → Settings → Import Custom Model** → klasörü/zip'i seçin.
3. Model listesinde görünür; NPU modunda üretin.

> İçe aktarma başarısız olursa [`05-sorun-giderme.md`](05-sorun-giderme.md)'ye bakın.
