#!/usr/bin/env bash
#
# Adim 3 — UNet ONNX -> QNN context binary (.bin) [NPU parcasi]
#
# Uc asama:
#   1) qnn-onnx-converter      : ONNX -> QNN model (.cpp + .bin) + KUANTIZASYON
#   2) qnn-model-lib-generator : model -> paylasimli kutuphane (.so)
#   3) qnn-context-binary-gen  : .so -> hedef HTP mimarisine gore context .bin
#
# Cikti: <OUT>/unet_<WxH>.bin  (Local Dream'in NPU'da yukledigi dosya)
#
# ONEMLI:
#   * $QNN_SDK_ROOT ayarlanmis olmali (Qualcomm AI Engine Direct SDK 2.28).
#     Indirme: Qualcomm AI Hub / QPM -> "v2.28.0.241029".
#   * Kesin kuantizasyon bayraklarinin (act_bw/weight_bw) en iyi degeri
#     modele gore degisir. a16w8 (16-bit aktivasyon, 8-bit agirlik) SD1.5
#     UNet icin kalite/performans dengesi acisindan iyi bir varsayilandir.
#   * Bu betik resmi Local Dream "convert_all.sh" mantigini yeniden uretir;
#     kesin referans icin: https://ld-guide.chino.icu/zh/conversion/sd15
#
# Kullanim:
#   export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029
#   ./03_convert_unet_qnn.sh work/onnx/unet_512x512.onnx \
#       work/calib/512x512/input_list.txt min work/qnn 512x512
#
set -euo pipefail

ONNX="${1:?UNet ONNX yolu gerekli}"
INPUT_LIST="${2:?Kalibrasyon input_list.txt gerekli}"
TIER="${3:-min}"          # min | mid | high
OUT="${4:-work/qnn}"
TAG="${5:-512x512}"       # cozunurluk etiketi (dosya adi icin)

# Kuantizasyon genislikleri (gerekirse degistirin)
ACT_BW="${ACT_BW:-16}"
WEIGHT_BW="${WEIGHT_BW:-8}"
BIAS_BW="${BIAS_BW:-32}"

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "HATA: QNN_SDK_ROOT ayarli degil. Ornek:"
  echo "  export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
LIB="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
export PATH="$BIN:$PATH"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}"

WORK="$OUT/build/unet_${TAG}"
mkdir -p "$WORK" "$OUT"

echo "==> [1/4] HTP config uretiliyor (tier=$TIER)"
HTP_CFG="$WORK/htp_${TIER}.json"
python3 "$SCRIPT_DIR/gen_htp_config.py" --tier "$TIER" --output "$HTP_CFG"

echo "==> [2/4] qnn-onnx-converter (kuantizasyon: a${ACT_BW}w${WEIGHT_BW})"
qnn-onnx-converter \
  --input_network "$ONNX" \
  --input_list "$INPUT_LIST" \
  --act_bw "$ACT_BW" \
  --weight_bw "$WEIGHT_BW" \
  --bias_bw "$BIAS_BW" \
  --float_bias_bw 32 \
  --output_path "$WORK/unet.cpp"

echo "==> [3/4] qnn-model-lib-generator (.cpp -> .so)"
qnn-model-lib-generator \
  -c "$WORK/unet.cpp" \
  -b "$WORK/unet.bin" \
  -o "$WORK/lib" \
  -t x86_64-linux-clang

echo "==> [4/4] qnn-context-binary-generator (-> hedef HTP binary)"
qnn-context-binary-generator \
  --model "$WORK/lib/x86_64-linux-clang/libunet.so" \
  --backend "$LIB/libQnnHtp.so" \
  --config_file "$HTP_CFG" \
  --binary_file "unet_${TAG}" \
  --output_dir "$OUT"

echo "[+] Bitti -> $OUT/unet_${TAG}.bin"
