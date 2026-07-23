#!/usr/bin/env bash
#
# convert_all.sh — Uctan uca: safetensors -> Local Dream _qnn2.39_min.zip
#
# Tum adimlari (0..5) tek komutta calistirir. Snapdragon 7 icin varsayilan
# tier "min"dir (en genis uyumluluk).
#
# On kosullar:
#   * Linux (veya WSL2), 20 GB+ RAM (512px), yuksek cozunurluk icin 64 GB+.
#   * export QNN_SDK_ROOT=...   (scripts/setup_qnn_sdk.py ile release'ten indirin)
#   * MNNConvert derlenmis  (export MNNCONVERT=/yol/MNN/build/MNNConvert)
#   * pip install -r requirements.txt
#
# Kullanim:
#   export QNN_SDK_ROOT="$(python scripts/setup_qnn_sdk.py --dest ./qairt | sed -n 's/^QNN_SDK_ROOT=//p' | tail -1)"
#   export MNNCONVERT=mnnconvert   # pip install MNN
#   ./convert_all.sh /indirilenler/AbsoluteReality.safetensors AbsoluteReality min
#
set -euo pipefail

CKPT="${1:?safetensors dosya yolu gerekli}"
NAME="${2:?Model adi gerekli (or. AbsoluteReality)}"
TIER="${3:-min}"                 # min | mid | high
RES="${4:-512x512,512x768,768x512}"
QNN_VERSION="${QNN_VERSION:-2.39}"   # ZIP/model_info surum etiketi

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
WORK="work/$NAME"

# Devam edilebilirlik: tamamlanmis adimlarin ciktisi varsa atlanir. Boylece
# ayni runtime'da bu betigi tekrar calistirmak, kaldigi yerden devam eder
# (yeniden export/MNN yapmaz). Zorla bastan yapmak icin: FORCE=1
FORCE="${FORCE:-0}"

if [ "$FORCE" = "0" ] && [ -f "$WORK/pipeline/model_index.json" ]; then
  echo "############ 0) safetensors -> diffusers  [ATLANDI - zaten var]"
else
  echo "############ 0) safetensors -> diffusers"
  python3 "$SDIR/00_load_safetensors.py" --checkpoint "$CKPT" \
      --output "$WORK/pipeline"
fi

if [ "$FORCE" = "0" ] && [ -f "$WORK/onnx/text_encoder.onnx" ] \
   && [ -f "$WORK/onnx/unet_${RES%%,*}.onnx" ]; then
  echo "############ 1) diffusers -> ONNX  [ATLANDI - zaten var]"
else
  echo "############ 1) diffusers -> ONNX"
  python3 "$SDIR/01_export_onnx.py" --pipeline "$WORK/pipeline" \
      --output "$WORK/onnx" --resolutions "$RES"
fi

if [ "$FORCE" = "0" ] && [ -f "$WORK/mnn/text_encoder.mnn" ] \
   && [ -f "$WORK/mnn/vae.mnn" ]; then
  echo "############ 4) text_encoder + vae -> MNN  [ATLANDI - zaten var]"
else
  echo "############ 4) text_encoder + vae -> MNN"
  ( cd "$SDIR" && ./04_convert_mnn.sh "../$WORK/onnx" "../$WORK/mnn" )
fi

echo "############ 2+3) her cozunurluk icin: kalibrasyon + UNet -> QNN"
IFS=',' read -ra RES_ARR <<< "$RES"
for tag in "${RES_ARR[@]}"; do
  tag="$(echo "$tag" | xargs)"
  if [ "$FORCE" = "0" ] && [ -f "$WORK/qnn/unet_${tag}.bin" ]; then
    echo "  --- $tag: UNet .bin zaten var [ATLANDI]"
    continue
  fi
  if [ "$FORCE" = "0" ] && [ -f "$WORK/calib/$tag/input_list.txt" ]; then
    echo "  --- $tag: kalibrasyon [ATLANDI - zaten var]"
  else
    echo "  --- $tag: kalibrasyon verisi"
    python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
        --resolution "$tag" --output "$WORK/calib/$tag" --mode real \
        --num-samples 6 --steps 15
  fi
  echo "  --- $tag: UNet -> QNN (tier=$TIER)"
  ( cd "$SDIR" && ./03_convert_unet_qnn.sh \
      "../$WORK/onnx/unet_${tag}.onnx" \
      "../$WORK/calib/${tag}/input_list.txt" \
      "$TIER" "../$WORK/qnn" "$tag" )
done

echo "############ 5) paketle -> _qnn${QNN_VERSION}_${TIER}.zip"
python3 "$SDIR/05_package.py" --name "$NAME" --tier "$TIER" \
    --qnn-version "$QNN_VERSION" \
    --qnn "$WORK/qnn" --mnn "$WORK/mnn" \
    --tokenizer "$WORK/pipeline/tokenizer" \
    --resolutions "$RES" --output dist

TAIL=""; [ "$TIER" = "min" ] && TAIL="_min"; [ "$TIER" = "high" ] && TAIL="_8gen3"
echo
echo "########################################################"
echo " BITTI. Cikti: dist/${NAME}_qnn${QNN_VERSION}${TAIL}.zip"
echo " Bu ZIP'i telefondaki Ruya/Local Dream 'Import Custom Model' ile alin."
echo "########################################################"
