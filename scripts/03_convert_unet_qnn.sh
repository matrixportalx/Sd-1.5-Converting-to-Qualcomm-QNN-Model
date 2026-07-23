#!/usr/bin/env bash
#
# Adim 3 — UNet ONNX -> QNN/QAIRT context binary (.bin) [NPU parcasi]
#
# Iki arac zincirini de destekler (SDK'da hangisi varsa onu kullanir):
#
#   * QAIRT 2.3x+  (varsayilan, ONERILEN):
#       qairt-converter   : ONNX -> float DLC
#       qairt-quantizer   : float DLC + kalibrasyon -> kuantize DLC
#       qnn-context-binary-generator --dlc_path : DLC -> hedef HTP binary
#
#   * Eski QNN 2.28:
#       qnn-onnx-converter (kuantizasyon dahil) -> .cpp/.bin
#       qnn-model-lib-generator -> .so
#       qnn-context-binary-generator --model -> hedef HTP binary
#
# Cikti: <OUT>/unet_<WxH>.bin
#
# Gereksinim: $QNN_SDK_ROOT (matrixportalx/qairt-sdk v2.39 ile kurulmus olabilir;
#             bkz. scripts/setup_qnn_sdk.py).
#
# Kullanim:
#   export QNN_SDK_ROOT=/content/qairt/qairt/2.39.0.250926
#   ./03_convert_unet_qnn.sh work/onnx/unet_512x512.onnx \
#       work/calib/512x512/input_list.txt min work/qnn 512x512
#
set -euo pipefail

ONNX="${1:?UNet ONNX yolu gerekli}"
INPUT_LIST="${2:?Kalibrasyon input_list.txt gerekli}"
TIER="${3:-min}"          # min | mid | high
OUT="${4:-work/qnn}"
TAG="${5:-512x512}"

# Kuantizasyon genislikleri
ACT_BW="${ACT_BW:-16}"
WEIGHT_BW="${WEIGHT_BW:-8}"
BIAS_BW="${BIAS_BW:-32}"

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  echo "HATA: QNN_SDK_ROOT ayarli degil."
  echo "  python scripts/setup_qnn_sdk.py --dest ./qairt   ile kurabilirsiniz."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
LIB="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
# ZIP'ten acilan araclar calistirma izni kaybetmis olabilir
chmod -R +x "$QNN_SDK_ROOT/bin" 2>/dev/null || true
export PATH="$BIN:$PATH"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}"

# QAIRT/QNN python konvertorleri Python 3.10 ister. setup_qnn_python.sh bir
# 3.10 venv kurup yolunu dosyaya yazar; QNN_PYTHON env veya o dosya kullanilir.
QNN_PY="${QNN_PYTHON:-}"
if [ -z "$QNN_PY" ] && [ -f /content/qnn_py.path ]; then
  QNN_PY="$(cat /content/qnn_py.path)"
fi
[ -z "$QNN_PY" ] && QNN_PY="python3"
echo "==> QAIRT python: $QNN_PY"

WORK="$OUT/build/unet_${TAG}"
mkdir -p "$WORK" "$OUT"

echo "==> HTP config uretiliyor (tier=$TIER)"
HTP_CFG="$WORK/htp_${TIER}.json"
python3 "$SCRIPT_DIR/gen_htp_config.py" --tier "$TIER" --output "$HTP_CFG"

HTP_BACKEND="$LIB/libQnnHtp.so"

if command -v qairt-converter >/dev/null 2>&1; then
  echo "==> QAIRT arac zinciri (qairt-converter + qairt-quantizer)"

  echo "  [1/3] qairt-converter: ONNX -> float DLC"
  "$QNN_PY" "$BIN/qairt-converter" \
    --input_network "$ONNX" \
    --output_path "$WORK/unet_fp.dlc"

  echo "  [2/3] qairt-quantizer: kalibrasyon (a${ACT_BW}w${WEIGHT_BW})"
  "$QNN_PY" "$BIN/qairt-quantizer" \
    --input_dlc "$WORK/unet_fp.dlc" \
    --input_list "$INPUT_LIST" \
    --act_bitwidth "$ACT_BW" \
    --weights_bitwidth "$WEIGHT_BW" \
    --bias_bitwidth "$BIAS_BW" \
    --output_dlc "$WORK/unet_quant.dlc"

  echo "  [3/3] qnn-context-binary-generator: DLC -> HTP binary"
  qnn-context-binary-generator \
    --dlc_path "$WORK/unet_quant.dlc" \
    --backend "$HTP_BACKEND" \
    --config_file "$HTP_CFG" \
    --binary_file "unet_${TAG}" \
    --output_dir "$OUT"

elif command -v qnn-onnx-converter >/dev/null 2>&1; then
  echo "==> Eski QNN arac zinciri (qnn-onnx-converter)"

  echo "  [1/3] qnn-onnx-converter (kuantizasyon dahil)"
  "$QNN_PY" "$BIN/qnn-onnx-converter" \
    --input_network "$ONNX" \
    --input_list "$INPUT_LIST" \
    --act_bw "$ACT_BW" --weight_bw "$WEIGHT_BW" --bias_bw "$BIAS_BW" \
    --float_bias_bw 32 \
    --output_path "$WORK/unet.cpp"

  echo "  [2/3] qnn-model-lib-generator (.cpp -> .so)"
  qnn-model-lib-generator \
    -c "$WORK/unet.cpp" -b "$WORK/unet.bin" \
    -o "$WORK/lib" -t x86_64-linux-clang

  echo "  [3/3] qnn-context-binary-generator (.so -> HTP binary)"
  qnn-context-binary-generator \
    --model "$WORK/lib/x86_64-linux-clang/libunet.so" \
    --backend "$HTP_BACKEND" \
    --config_file "$HTP_CFG" \
    --binary_file "unet_${TAG}" \
    --output_dir "$OUT"

else
  echo "HATA: Ne qairt-converter ne de qnn-onnx-converter bulundu."
  echo "      $BIN icindeki araclari kontrol edin."
  exit 1
fi

echo "[+] Bitti -> $OUT/unet_${TAG}.bin"
