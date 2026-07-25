#!/usr/bin/env bash
#
# Adim 3 (genel) — ONNX -> QNN context binary (.bin)
#
# UNet = int8 kuantize (v69 icin sart). VAE = fp16 (kuantize DEGIL; kalite +
# v69 fp16 destegi). Graf adi, uygulamanin createModel(path, "<ad>") cagrisiyla
# eslesmeli; bunu DLC dosya adiyla ayarlariz.
#
# Kullanim:
#   03_convert_qnn.sh <onnx> <graph_name> <mode:quant|float> <input_list|-> \
#                     <tier> <out_dir> <out_name>
# Ornek (UNet, int8):
#   03_convert_qnn.sh work/onnx/unet_512x512.onnx unet quant \
#       work/calib/512x512/input_list.txt min work/qnn unet
# Ornek (VAE decoder, fp16):
#   03_convert_qnn.sh work/onnx/vae_decoder.onnx vae_decoder float - min work/qnn vae_decoder
#
set -euo pipefail

ONNX="${1:?ONNX yolu}"
GRAPH="${2:?graf adi (unet|vae_decoder|vae_encoder)}"
MODE="${3:?mode: quant|float}"
INPUT_LIST="${4:-'-'}"
TIER="${5:-min}"
OUT="${6:-work/qnn}"
OUT_NAME="${7:-$GRAPH}"

ACT_BW="${ACT_BW:-8}"; WEIGHT_BW="${WEIGHT_BW:-8}"; BIAS_BW="${BIAS_BW:-32}"

: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
LIB="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
chmod -R +x "$QNN_SDK_ROOT/bin" 2>/dev/null || true
export PATH="$BIN:$PATH"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}"

QNN_PY="${QNN_PYTHON:-}"
[ -z "$QNN_PY" ] && [ -f /content/qnn_py.path ] && QNN_PY="$(cat /content/qnn_py.path)"
[ -z "$QNN_PY" ] && QNN_PY="python3"

WORK="$OUT/build/$GRAPH"
mkdir -p "$WORK" "$OUT"
HTP_CFG="$WORK/htp_${TIER}.json"
python3 "$SCRIPT_DIR/gen_htp_config.py" --tier "$TIER" --output "$HTP_CFG"
HTP_BACKEND="$LIB/libQnnHtp.so"

_QHELP=""
quant_help() {  # qairt-quantizer --help ciktisini bir kez yakala (bayrak adi tespiti)
  if [ -z "$_QHELP" ]; then
    _QHELP="$("$QNN_PY" "$BIN/qairt-quantizer" --help 2>&1 || true)"
  fi
  printf '%s' "$_QHELP"
}
has_qflag() { quant_help | grep -q -- "$1"; }

run_tool() {  # mode(py|native) tool args...
  local m="$1"; shift; local tool="$1"; shift; local rc=0
  if [ "$m" = py ]; then "$QNN_PY" "$BIN/$tool" "$@" || rc=$?; else "$BIN/$tool" "$@" || rc=$?; fi
  if [ "$rc" -ne 0 ]; then
    echo ""; echo "!!! '$tool' BASARISIZ (kod $rc). Arayuz (--help) — BU CIKTIYI PAYLAS:"
    if [ "$m" = py ]; then "$QNN_PY" "$BIN/$tool" --help 2>&1 | head -400 || true
    else "$BIN/$tool" --help 2>&1 | head -400 || true; fi
    exit "$rc"
  fi
}

command -v qairt-converter >/dev/null 2>&1 || { echo "HATA: qairt-converter yok"; exit 1; }

# Graf adi = DLC dosya adi (uygulama bu adi bekler)
FP_DLC="$WORK/${GRAPH}.dlc"
if [ -f "$FP_DLC" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "  [converter] ATLANDI ($FP_DLC var)"
else
  echo "  [converter] $ONNX -> $FP_DLC (graf: $GRAPH)"
  # QUANT_OVERRIDES: karma hassasiyet (16-bit graf I/O + 8-bit ic hesap).
  # Referans UNet binary'si UFIXED_POINT_16 I/O kullanir; v68'de 16-bit MatMul
  # desteklenmedigi icin YALNIZCA sinir tensorleri 16-bit yapilir.
  if [ -n "${QUANT_OVERRIDES:-}" ] && [ -f "${QUANT_OVERRIDES}" ]; then
    echo "    [overrides] $QUANT_OVERRIDES (16-bit I/O)"
    run_tool py qairt-converter --input_network "$ONNX" --output_path "$FP_DLC" \
      --quantization_overrides "$QUANT_OVERRIDES"
  else
    run_tool py qairt-converter --input_network "$ONNX" --output_path "$FP_DLC"
  fi
fi

if [ "$MODE" = quant ]; then
  # input_list'i mutlak yap
  ABS_LIST="$WORK/input_list_abs.txt"
  LIST_DIR="$(cd "$(dirname "$INPUT_LIST")" && pwd)"
  python3 - "$INPUT_LIST" "$ABS_LIST" "$LIST_DIR" <<'PY'
import os,sys
src,dst,d=sys.argv[1],sys.argv[2],sys.argv[3]
out=[]
for line in open(src):
    if not line.strip(): continue
    toks=[]
    for t in line.split():
        if ":=" in t:
            n,p=t.split(":=",1); toks.append(f"{n}:={os.path.join(d,os.path.basename(p))}")
        else: toks.append(os.path.join(d,os.path.basename(t)))
    out.append(" ".join(toks))
open(dst,"w").write("\n".join(out)+"\n")
PY
  Q_DLC="$WORK/${GRAPH}_q.dlc"
  QARGS=()
  # RESTRICT_STEPS: 16-bit MatMul icin QAIRT tarafindan GEREKLI
  # (--help: "This argument is required for 16-bit Matmul operations")
  #
  # ONEMLI: restrict_quantization_steps YALNIZCA parametre kuantalayici
  # simetrik ya da per-channel/row oldugunda uygulanir; aksi halde QAIRT
  # "Value will be ignored" uyarisini basar ve 16-bit MatMul, HTP
  # dogrulamasinda "expected >= 73" hatasiyla duser. Bu yuzden simetrik
  # sema bayragini birlikte veriyoruz.
  if [ -n "${RESTRICT_STEPS:-}" ]; then
    QARGS+=(--restrict_quantization_steps "$RESTRICT_STEPS")
    if has_qflag "--param_quantizer_schema"; then
      QARGS+=(--param_quantizer_schema symmetric)
    elif has_qflag "--param_quantizer"; then
      QARGS+=(--param_quantizer symmetric)
    fi
    # --use_per_row_quantization = "rowwise quantization of Matmul and
    # FullyConnected ops" (SDK --help). Sorunlu op'lar tam olarak MatMul
    # oldugu icin restrict_quantization_steps'in aradigi "per channel/row"
    # semasini saglayan DOGRU bayrak budur; per_channel yalnizca konvolusyon
    # agirliklarini kapsar.
    if [ "${PER_ROW:-1}" = "1" ] && has_qflag "--use_per_row_quantization"; then
      QARGS+=(--use_per_row_quantization)
    fi
    if [ "${PER_CHANNEL:-1}" = "1" ] && has_qflag "--use_per_channel_quantization"; then
      QARGS+=(--use_per_channel_quantization)
    fi
  fi
  # QUANT_EXTRA: notebook'tan serbest bayrak gecisi (kod duzenlemeden deneme).
  # ornek: QUANT_EXTRA="--target_backend HTP --act_quantizer_calibration mse"
  if [ -n "${QUANT_EXTRA:-}" ]; then
    # shellcheck disable=SC2206
    QARGS+=(${QUANT_EXTRA})
  fi
  # Kuantalayici argumanlari degistiyse DLC'yi yeniden uret (FORCE gerekmeden).
  Q_SIG="$WORK/${GRAPH}_q.args"
  Q_SIG_NEW="a${ACT_BW} w${WEIGHT_BW} b${BIAS_BW} ${QARGS[*]:-}"
  if [ -f "$Q_DLC" ] && [ "${FORCE:-0}" != "1" ] \
     && [ "$(cat "$Q_SIG" 2>/dev/null)" = "$Q_SIG_NEW" ]; then
    echo "  [quantizer] ATLANDI ($Q_DLC guncel)"
  else
    [ -f "$Q_DLC" ] && echo "  [quantizer] argumanlar degisti -> yeniden kuantize"
    echo "  [quantizer] $Q_SIG_NEW"
    run_tool py qairt-quantizer --input_dlc "$FP_DLC" --input_list "$ABS_LIST" \
      --act_bitwidth "$ACT_BW" --weights_bitwidth "$WEIGHT_BW" \
      --bias_bitwidth "$BIAS_BW" "${QARGS[@]+"${QARGS[@]}"}" \
      --output_dlc "$Q_DLC"
    echo "$Q_SIG_NEW" > "$Q_SIG"
    # Kuantizasyon degisti -> eski context binary gecersiz
    rm -f "$OUT/${OUT_NAME}.bin"
  fi
  DLC_FOR_BIN="$Q_DLC"
else
  echo "  [float] kuantizasyon yok (fp16)"
  DLC_FOR_BIN="$FP_DLC"
fi

echo "  [context-bin] $DLC_FOR_BIN -> $OUT/${OUT_NAME}.bin"
run_tool native qnn-context-binary-generator \
  --dlc_path "$DLC_FOR_BIN" --backend "$HTP_BACKEND" --config_file "$HTP_CFG" \
  --binary_file "$OUT_NAME" --output_dir "$OUT"

echo "[+] $OUT/${OUT_NAME}.bin"
