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

_CHELP=""
conv_help() {
  if [ -z "$_CHELP" ]; then
    _CHELP="$("$QNN_PY" "$BIN/qairt-converter" --help 2>&1 || true)"
  fi
  printf '%s' "$_CHELP"
}
has_cflag() { conv_help | grep -q -- "$1"; }

# QAIRT 2.39 surum notu: "Enabled support for dynamic 16-bit weights BY DEFAULT
# in qairt-converter and qairt-quantizer ... A new --disable_dynamic_16_bit_weights
# flag has been added to revert to 8-bit conversion if needed." {147008}
#
# Bizim hatamiz tam olarak bu: karma hassasiyette MatMul'un ikinci operandi
# ("agirlik") 16-bit'e cekiliyor ve
#   "8 bit activations with 16 bit weights are not supported on backend"
# cikiyor. Bayrak --help'te GORUNMUYOR (gizli), bu yuzden dogrudan deneyerek
# varligini sinariyoruz: yoksa "unrecognized arguments" der.
has_hidden_flag() {  # tool flag
  local out
  out="$("$QNN_PY" "$BIN/$1" "$2" 2>&1 || true)"
  ! printf '%s' "$out" | grep -q "unrecognized arguments"
}

# Graf I/O'sunu 16-bit yapip ic hesabi 8-bit birakmanin resmi yolu:
# qairt-converter --dump_config_template <yaml> -> duzenle -> --config <yaml>
# Sablonu bir kez dokup logliyoruz (saniyeler surer, semayi gormek icin).
IO_TPL="$OUT/io_config_template.yaml"
if [ "${DUMP_IO_TEMPLATE:-1}" = "1" ] && [ ! -f "$IO_TPL" ] \
   && has_cflag "--dump_config_template"; then
  # --input_network zorunlu arguman; once onsuz dene, olmazsa ONNX ile.
  "$QNN_PY" "$BIN/qairt-converter" --dump_config_template "$IO_TPL" >/dev/null 2>&1 || true
  [ -f "$IO_TPL" ] || "$QNN_PY" "$BIN/qairt-converter" --input_network "$ONNX" \
      --dump_config_template "$IO_TPL" >/dev/null 2>&1 || true
  if [ -f "$IO_TPL" ]; then
    echo "  [io-config sablonu] $IO_TPL — BU BLOGU PAYLAS:"
    sed 's/^/    | /' "$IO_TPL" | head -80
  fi
fi

# ---- Backend-aware kuantizasyon -------------------------------------------
# qairt --help "Backend Options":
#   --target_backend BACKEND     "generate a graph optimized for the given backend"
#   --target_soc_model SOC_MODEL "the SOC on which the model needs to run"
# Hedef SoC verilmezse quantizer genel bir graf uretir ve v68/v69'da
# desteklenmeyen (or. 16-bit MatMul) op'lar context-binary asamasinda
# "expected >= 73" ile reddedilir. SoC'yi soyleyince quantizer op bazinda
# hedefin destekledigi hassasiyeti secer.
#
# HTP mimarisi -> temsili SoC (SDK sürüm notu w8a16'yi SM8350/v68 icin anlatir)
_arch="${DSP_ARCH:-}"
if [ -z "$_arch" ]; then
  case "$TIER" in min) _arch=v68 ;; mid) _arch=v73 ;; high) _arch=v75 ;; esac
fi
# SoC listesi SDK surumune gore degisir (2.39 "SM8350 is not supported" der),
# bu yuzden SABIT KODLAMIYORUZ: desteklenen liste SDK'dan okunur ve hedef
# mimariye uyan ilk SoC secilir.
SOC_LIST="$OUT/soc_models.txt"
if [ ! -s "$SOC_LIST" ]; then
  "$QNN_PY" "$SCRIPT_DIR/list_soc_models.py" > "$SOC_LIST" 2>/dev/null || true
fi
soc_supported() { [ -s "$SOC_LIST" ] && cut -f1 "$SOC_LIST" | grep -qx -- "$1"; }

TARGET_BACKEND="${TARGET_BACKEND:-HTP}"
if [ -z "${TARGET_SOC+x}" ]; then     # kullanici belirtmediyse otomatik sec
  TARGET_SOC="$("$QNN_PY" "$SCRIPT_DIR/list_soc_models.py" --arch "$_arch" 2>/dev/null | head -1)"
fi
if [ -n "$TARGET_SOC" ] && [ -s "$SOC_LIST" ] && ! soc_supported "$TARGET_SOC"; then
  echo "  [backend-aware] UYARI: '$TARGET_SOC' bu SDK'da desteklenmiyor -> SoC atlaniyor"
  echo "                  Desteklenen (ilk 20): $(cut -f1 "$SOC_LIST" | head -20 | tr '\n' ' ')"
  TARGET_SOC=""
fi

BE_Q=(); BE_C=()
if [ -n "$TARGET_BACKEND" ]; then
  BE_Q=(--target_backend "$TARGET_BACKEND")
  BE_C=(--target_backend "$TARGET_BACKEND")
  if [ -n "$TARGET_SOC" ]; then
    BE_Q+=(--target_soc_model "$TARGET_SOC")
    BE_C+=(--target_soc_model "$TARGET_SOC")
  fi
  has_qflag "--target_backend" || BE_Q=()
  has_cflag "--target_backend" || BE_C=()
  echo "  [backend-aware] $TARGET_BACKEND / ${TARGET_SOC:-(SoC yok)} (htp $_arch)"
fi

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
CARGS=("${BE_C[@]+"${BE_C[@]}"}")
# QUANT_OVERRIDES: karma hassasiyet (16-bit graf I/O + 8-bit ic hesap).
if [ -n "${QUANT_OVERRIDES:-}" ] && [ -f "${QUANT_OVERRIDES}" ]; then
  CARGS+=(--quantization_overrides "$QUANT_OVERRIDES")
fi
# 2.39'dan beri dinamik 16-bit agirliklar VARSAYILAN OLARAK ACIK; MatMul'un
# dinamik ikinci operandi 16-bit'e cekiliyor ve 8-bit aktivasyonla birlesince
#   "8 bit activations with 16 bit weights are not supported on backend"
# cikiyor. Bu davranis override/config'ten BAGIMSIZ, o yuzden kosulsuz kapatiyoruz
# (2.39 oncesi davranis = referansin uretildigi davranis).
DYN16W_OFF=0
if [ "${DISABLE_DYN16W:-1}" = "1" ] \
   && has_hidden_flag qairt-converter --disable_dynamic_16_bit_weights; then
  CARGS+=(--disable_dynamic_16_bit_weights)
  DYN16W_OFF=1
  echo "  [dyn16w] --disable_dynamic_16_bit_weights (converter)"
fi
# IO_CONFIG: SDK'nin I/O yapilandirma YAML'i (--dump_config_template semasi).
# 16-bit graf sinirini YALNIZCA sinirda tutar; --quantization_overrides gibi
# etiketi ic grafa (to_k/to_v MatMul) tasimaz.
if [ -n "${IO_CONFIG:-}" ] && [ -f "${IO_CONFIG}" ]; then
  CARGS+=(--config "$IO_CONFIG")
  echo "  [io-config] $IO_CONFIG"
fi
C_SIG="$WORK/${GRAPH}.args"
C_SIG_NEW="${CARGS[*]:-}"
if [ -f "$FP_DLC" ] && [ "${FORCE:-0}" != "1" ] \
   && [ "$(cat "$C_SIG" 2>/dev/null)" = "$C_SIG_NEW" ]; then
  echo "  [converter] ATLANDI ($FP_DLC guncel)"
else
  [ -f "$FP_DLC" ] && echo "  [converter] argumanlar degisti -> yeniden donusum"
  echo "  [converter] $ONNX -> $FP_DLC (graf: $GRAPH) ${C_SIG_NEW}"
  run_tool py qairt-converter --input_network "$ONNX" --output_path "$FP_DLC" \
    "${CARGS[@]+"${CARGS[@]}"}"
  echo "$C_SIG_NEW" > "$C_SIG"
  # fp DLC degisti -> kuantize DLC ve context binary gecersiz
  rm -f "$WORK/${GRAPH}_q.dlc" "$WORK/${GRAPH}_q.args" "$OUT/${OUT_NAME}.bin"
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
  QARGS=("${BE_Q[@]+"${BE_Q[@]}"}")
  # RESTRICT_STEPS: 16-bit MatMul icin QAIRT tarafindan GEREKLI
  # (--help: "This argument is required for 16-bit Matmul operations")
  #
  # ONEMLI: restrict_quantization_steps YALNIZCA parametre kuantalayici
  # simetrik ya da per-channel/row oldugunda uygulanir; aksi halde QAIRT
  # "Value will be ignored" uyarisini basar ve 16-bit MatMul, HTP
  # dogrulamasinda "expected >= 73" hatasiyla duser. Bu yuzden simetrik
  # sema bayragini birlikte veriyoruz.
  # RESTRICT_STEPS=auto -> araligi AGIRLIK bit genisliginden turet.
  # Quantizer araligi parametre (agirlik) kuantalayicisina uyguluyor:
  #   "Cannot restrict quantization steps to -32768 - 32639 for bitwidth: 8"
  # w8  -> 8-bit aralik  (-0x80  0x7F)
  # w16 -> 16-bit aralik (-0x8000 0x7F7F)
  if [ "${RESTRICT_STEPS:-}" = "auto" ]; then
    case "$WEIGHT_BW" in
      16) RESTRICT_STEPS="-0x8000 0x7F7F" ;;
      *)  RESTRICT_STEPS="-0x80 0x7F" ;;
    esac
  fi
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
    # PER_CHANNEL VARSAYILAN KAPALI: acikken HTP ilk Conv'u reddediyor —
    #   "has incorrect Value 320, expected equal to 320" (/unet/conv_in/Conv)
    # Zaten gereksiz: restrict'in aradigi sema kosulunu symmetric + per_row
    # sagliyor; per_channel yalnizca konvolusyon agirliklarini degistiriyor.
    if [ "${PER_CHANNEL:-0}" = "1" ] && has_qflag "--use_per_channel_quantization"; then
      QARGS+=(--use_per_channel_quantization)
    fi
  fi
  if [ "${DISABLE_DYN16W:-1}" = "1" ] \
     && has_hidden_flag qairt-quantizer --disable_dynamic_16_bit_weights; then
    QARGS+=(--disable_dynamic_16_bit_weights)
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

# Context binary imzasi: kaynak DLC (boyut+mtime) + hedef mimari. Degismediyse
# ~3 dk'lik uretimi atla; degistiyse otomatik yenile (FORCE gerekmez).
B_SIG="$OUT/${OUT_NAME}.bin.args"
B_SIG_NEW="$(stat -c '%s:%Y' "$DLC_FOR_BIN" 2>/dev/null) arch=${DSP_ARCH:-$TIER}"
if [ -f "$OUT/${OUT_NAME}.bin" ] && [ "${FORCE:-0}" != "1" ] \
   && [ "$(cat "$B_SIG" 2>/dev/null)" = "$B_SIG_NEW" ]; then
  echo "  [context-bin] ATLANDI ($OUT/${OUT_NAME}.bin guncel)"
else
  # Hedef mimaride uretim basarisiz olursa bir ust mimariyi dene. v68 binary'si
  # v69 donanimda da calisir; tersi degil. Snapdragon 7 Gen 1 = v69 oldugundan
  # v68 -> v69 yedeklemesi cihazda hala calisan bir model verir.
  ARCH_TRY="${DSP_ARCH:-}"
  [ -z "$ARCH_TRY" ] && ARCH_TRY="$_arch"
  case "${BIN_ARCH_FALLBACK-auto}" in
    auto) case "$ARCH_TRY" in v68) ARCH_CHAIN="v68 v69" ;; *) ARCH_CHAIN="$ARCH_TRY" ;; esac ;;
    "")   ARCH_CHAIN="$ARCH_TRY" ;;
    *)    ARCH_CHAIN="$ARCH_TRY ${BIN_ARCH_FALLBACK}" ;;
  esac

  bin_ok=0
  for a in $ARCH_CHAIN; do
    echo "  [context-bin] $DLC_FOR_BIN -> $OUT/${OUT_NAME}.bin (dsp_arch=$a)"
    DSP_ARCH="$a" python3 "$SCRIPT_DIR/gen_htp_config.py" --tier "$TIER" \
      --output "$HTP_CFG" >/dev/null
    rc=0
    "$BIN/qnn-context-binary-generator" \
      --dlc_path "$DLC_FOR_BIN" --backend "$HTP_BACKEND" --config_file "$HTP_CFG" \
      --binary_file "$OUT_NAME" --output_dir "$OUT" || rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$OUT/${OUT_NAME}.bin" ]; then
      bin_ok=1
      echo "$a" > "$OUT/${OUT_NAME}.arch"
      [ "$a" != "$ARCH_TRY" ] && \
        echo "  [context-bin] NOT: $ARCH_TRY basarisiz, $a ile uretildi"
      break
    fi
    echo "  [context-bin] dsp_arch=$a BASARISIZ (kod $rc)"
  done
  if [ "$bin_ok" -ne 1 ]; then
    echo ""
    echo "!!! qnn-context-binary-generator tum mimarilerde basarisiz: $ARCH_CHAIN"
    exit 1
  fi
  echo "$B_SIG_NEW" > "$B_SIG"
fi

echo "[+] $OUT/${OUT_NAME}.bin"
