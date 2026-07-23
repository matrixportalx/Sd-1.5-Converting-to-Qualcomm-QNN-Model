#!/usr/bin/env bash
#
# Adim 4 — text_encoder + vae_decoder ONNX -> MNN (.mnn) [CPU/GPU parcasi]
#
# Local Dream'de metin kodlayici (CLIP) ve VAE, NPU'da DEGIL, MNN motorunda
# (CPU/GPU) calisir. Bu yuzden bunlari QNN'e degil MNN'e ceviririz.
#
# Gereksinim: MNN'in "MNNConvert" araci derlenmis olmali.
#   git clone https://github.com/alibaba/MNN
#   cd MNN && mkdir build && cd build
#   cmake .. -DMNN_BUILD_CONVERTER=ON && make -j
#   -> build/MNNConvert
#
# Kullanim:
#   MNNCONVERT=/yol/MNN/build/MNNConvert \
#   ./04_convert_mnn.sh work/onnx work/mnn
#
set -euo pipefail

ONNX_DIR="${1:?ONNX klasoru gerekli}"
OUT="${2:-work/mnn}"
MNNCONVERT="${MNNCONVERT:-MNNConvert}"
FP16="${FP16:-1}"   # 1 = fp16 kaydet (daha kucuk/dosya, mobilde hizli)

mkdir -p "$OUT"

fp16_flag=""
if [[ "$FP16" == "1" ]]; then fp16_flag="--fp16"; fi

convert() {
  local in="$1" out="$2"
  echo "==> $in -> $out"
  "$MNNCONVERT" -f ONNX \
    --modelFile "$in" \
    --MNNModel "$out" \
    --bizCode localdream $fp16_flag
}

convert "$ONNX_DIR/text_encoder.onnx" "$OUT/text_encoder.mnn"
convert "$ONNX_DIR/vae_decoder.onnx"  "$OUT/vae.mnn"

echo "[+] MNN cikti -> $OUT (text_encoder.mnn, vae.mnn)"
