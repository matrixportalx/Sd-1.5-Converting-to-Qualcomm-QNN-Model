#!/usr/bin/env bash
#
# Adim 4 — clip_v2.onnx -> clip_v2.mnn  (CLIP transformer, CPU/GPU icin MNN)
#
# Local Dream CLIP transformer'i MNN olarak calistirir (giris 'input_embedding',
# cikis 'last_hidden_state'). Gomme tablolari (token_emb/pos_emb) ayri .bin'dir.
#
# Gereksinim: MNNConvert (pip install MNN -> mnnconvert)
#
# Kullanim:
#   MNNCONVERT=mnnconvert ./04_convert_mnn.sh work/onnx work/mnn
#
set -euo pipefail
ONNX_DIR="${1:?ONNX klasoru}"
OUT="${2:-work/mnn}"
MNNCONVERT="${MNNCONVERT:-mnnconvert}"
FP16="${FP16:-1}"
mkdir -p "$OUT"
fp16_flag=""; [ "$FP16" = "1" ] && fp16_flag="--fp16"

echo "==> clip_v2.onnx -> clip_v2.mnn"
"$MNNCONVERT" -f ONNX --modelFile "$ONNX_DIR/clip_v2.onnx" \
  --MNNModel "$OUT/clip_v2.mnn" --bizCode localdream $fp16_flag

echo "[+] MNN cikti -> $OUT/clip_v2.mnn"
