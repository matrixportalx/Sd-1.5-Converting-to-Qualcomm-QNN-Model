#!/usr/bin/env bash
#
# convert_all.sh — Uctan uca: safetensors -> Local Dream _qnn2.28_min.zip
#
# Tum adimlari (0..5) tek komutta calistirir. Snapdragon 7 icin varsayilan
# tier "min"dir (en genis uyumluluk).
#
# On kosullar:
#   * Linux (veya WSL2), 20 GB+ RAM (512px), yuksek cozunurluk icin 64 GB+.
#   * export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029   (AI Engine Direct 2.28)
#   * MNNConvert derlenmis  (export MNNCONVERT=/yol/MNN/build/MNNConvert)
#   * pip install -r requirements.txt
#
# Kullanim:
#   export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029
#   export MNNCONVERT=/opt/MNN/build/MNNConvert
#   ./convert_all.sh /indirilenler/AbsoluteReality.safetensors AbsoluteReality min
#
set -euo pipefail

CKPT="${1:?safetensors dosya yolu gerekli}"
NAME="${2:?Model adi gerekli (or. AbsoluteReality)}"
TIER="${3:-min}"                 # min | mid | high
RES="${4:-512x512,512x768,768x512}"

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
WORK="work/$NAME"

echo "############ 0) safetensors -> diffusers"
python3 "$SDIR/00_load_safetensors.py" --checkpoint "$CKPT" \
    --output "$WORK/pipeline"

echo "############ 1) diffusers -> ONNX"
python3 "$SDIR/01_export_onnx.py" --pipeline "$WORK/pipeline" \
    --output "$WORK/onnx" --resolutions "$RES"

echo "############ 4) text_encoder + vae -> MNN"
( cd "$SDIR" && ./04_convert_mnn.sh "../$WORK/onnx" "../$WORK/mnn" )

echo "############ 2+3) her cozunurluk icin: kalibrasyon + UNet -> QNN"
IFS=',' read -ra RES_ARR <<< "$RES"
for tag in "${RES_ARR[@]}"; do
  tag="$(echo "$tag" | xargs)"
  echo "  --- $tag: kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$tag" --output "$WORK/calib/$tag" --mode real \
      --num-samples 6 --steps 15
  echo "  --- $tag: UNet -> QNN (tier=$TIER)"
  ( cd "$SDIR" && ./03_convert_unet_qnn.sh \
      "../$WORK/onnx/unet_${tag}.onnx" \
      "../$WORK/calib/${tag}/input_list.txt" \
      "$TIER" "../$WORK/qnn" "$tag" )
done

echo "############ 5) paketle -> _qnn2.28_${TIER}.zip"
python3 "$SDIR/05_package.py" --name "$NAME" --tier "$TIER" \
    --qnn "$WORK/qnn" --mnn "$WORK/mnn" \
    --tokenizer "$WORK/pipeline/tokenizer" \
    --resolutions "$RES" --output dist

echo
echo "########################################################"
echo " BITTI. Cikti: dist/${NAME}_qnn2.28_${TIER}.zip"
echo " Bu ZIP'i telefondaki Ruya/Local Dream 'Import Custom Model' ile alin."
echo "########################################################"
