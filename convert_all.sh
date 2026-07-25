#!/usr/bin/env bash
#
# convert_all.sh — Uctan uca: safetensors -> Local Dream (Ruya) _min.zip
#
# Local Dream SD1.5-NPU formati:
#   token_emb.bin, pos_emb.bin, clip_v2.mnn, unet.bin (QNN int8),
#   vae_decoder.bin + vae_encoder.bin (QNN fp16), tokenizer.json
#
# On kosullar:
#   export QNN_SDK_ROOT="$(python scripts/setup_qnn_sdk.py --dest ./qairt | sed -n 's/^QNN_SDK_ROOT=//p' | tail -1)"
#   export MNNCONVERT=mnnconvert      # pip install MNN
#   bash scripts/setup_qnn_python.sh  # QAIRT icin Python 3.10
#
# Kullanim:
#   ./convert_all.sh model.safetensors CyberRealisticLCM min 512x512
#
set -euo pipefail

CKPT="${1:?safetensors dosya yolu}"
NAME="${2:?Model adi}"
TIER="${3:-min}"
RES="${4:-512x512}"          # taban cozunurluk (tek). Coklu icin patch gerekir.
QNN_VERSION="${QNN_VERSION:-2.39}"
FORCE="${FORCE:-0}"
export FORCE

# min (v69) 16-bit MatMul'u desteklemez -> UNet 8-bit
if [ "$TIER" = "min" ]; then export ACT_BW="${ACT_BW:-8}"; else export ACT_BW="${ACT_BW:-16}"; fi

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
WORK="work/$NAME"
TAG="${RES%%,*}"            # ilk cozunurluk (taban)

# REBUILD_BIN=1 : sadece context binary'leri yeniden uret (kuantize DLC'ler
# korunur -> dakikalar, saatler degil). dsp_arch degistirmek icin ideal:
#   DSP_ARCH=v68 REBUILD_BIN=1 ./convert_all.sh ...
if [ "${REBUILD_BIN:-0}" = "1" ]; then
  echo "[*] REBUILD_BIN=1 -> mevcut .bin'ler siliniyor (DLC'ler korunuyor)"
  rm -f "$WORK/qnn/unet.bin" "$WORK/qnn/vae_decoder.bin" "$WORK/qnn/vae_encoder.bin"
fi
# Referans binary dspArch=68 -> min tier varsayilani v68 (v69 donanimda da calisir)
if [ "$TIER" = "min" ] && [ -z "${DSP_ARCH:-}" ]; then DSP_ARCH=v68; fi
if [ -n "${DSP_ARCH:-}" ]; then
  echo "[*] DSP_ARCH override = $DSP_ARCH"
  export DSP_ARCH
fi

# ---- 0) safetensors -> diffusers ------------------------------------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/pipeline/model_index.json" ]; then
  echo "### 0) safetensors -> diffusers [ATLANDI]"
else
  echo "### 0) safetensors -> diffusers"
  python3 "$SDIR/00_load_safetensors.py" --checkpoint "$CKPT" --output "$WORK/pipeline"
fi

# ---- 1) ONNX/emb export (surum damgali) -----------------------------------
# v11: UNet a16w8 + restrict steps (override yerine). v10: referans reçetesi — 16-bit I/O override, timestamp INT_32, graf "model", v68.
EXPORT_VERSION="11"
STAMP="$WORK/onnx/.export_version"
if [ "$FORCE" = 0 ] && [ -f "$WORK/onnx/clip_v2.onnx" ] \
   && [ -f "$WORK/onnx/unet_${TAG}.onnx" ] \
   && [ "$(cat "$STAMP" 2>/dev/null)" = "$EXPORT_VERSION" ]; then
  echo "### 1) ONNX/emb export [ATLANDI - guncel v$EXPORT_VERSION]"
else
  echo "### 1) ONNX/emb export (v$EXPORT_VERSION)"
  python3 "$SDIR/01_export_onnx.py" --pipeline "$WORK/pipeline" \
      --output "$WORK/onnx" --resolutions "$RES"
  echo "$EXPORT_VERSION" > "$STAMP"
  # unet.bin'i KORU (yeniden kuantizasyon 8-10 dk sürer); geri kalani temizle.
  echo "  [temizlik] mnn/ + qnn/build + vae bin'leri siliniyor (unet.bin korunuyor)"
  rm -rf "$WORK/mnn" "$WORK/qnn/build" \
         "$WORK/qnn/vae_decoder.bin" "$WORK/qnn/vae_encoder.bin"
fi

# ---- 2) clip_v2.onnx -> clip_v2.mnn ---------------------------------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/mnn/clip_v2.mnn" ]; then
  echo "### 2) clip_v2 -> MNN [ATLANDI]"
else
  echo "### 2) clip_v2 -> MNN"
  ( cd "$SDIR" && ./04_convert_mnn.sh "../$WORK/onnx" "../$WORK/mnn" )
fi

# ---- 3) UNet kalibrasyonu (int8 icin) -------------------------------------
# Tensor isimleri degistiyse (timestamp/text_embedding) kalibrasyon listesi
# gecersizdir -> yeniden uret.
if [ -f "$WORK/calib/$TAG/input_list.txt" ] && \
   ! grep -q "text_embedding:=" "$WORK/calib/$TAG/input_list.txt"; then
  echo "[*] Kalibrasyon listesi eski tensor isimleri iceriyor -> yenilenecek"
  rm -rf "$WORK/calib/$TAG"
fi
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib/$TAG/input_list.txt" ]; then
  echo "### 3) UNet kalibrasyon [ATLANDI]"
else
  echo "### 3) UNet kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib/$TAG" --mode real \
      --num-samples "${CALIB_PROMPTS:-4}" --steps "${CALIB_STEPS:-4}"
fi

# ---- 4) UNet -> QNN (int8, graf 'unet') -----------------------------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/qnn/unet.bin" ]; then
  echo "### 4) UNet -> QNN [ATLANDI]"
else
  # Referans: I/O UFIXED_16 -> TAM 16-bit aktivasyon (a16w8).
  # 16-bit MatMul icin --restrict_quantization_steps ZORUNLU (QAIRT --help) VE
  # restrict ancak simetrik/per-channel parametre kuantalayiciyla uygulanir
  # (bkz. scripts/03_convert_qnn.sh).
  #
  # UNET_MODE ile denenebilecek secenekler:
  #   a16w8_restrict (varsayilan) : referans recete, v68'de 16-bit MatMul
  #   a16w8                       : restrict yok (v73+ gerektirir)
  #   a8w8                        : tam 8-bit — her zaman calisir, kalite dusuk
  UNET_MODE="${UNET_MODE:-a16w8_restrict}"
  case "$UNET_MODE" in
    a16w8_restrict) U_ACT=16; U_RESTRICT="${UNET_RESTRICT:--0x8000 0x7F7F}" ;;
    a16w8)          U_ACT=16; U_RESTRICT="" ;;
    a8w8)           U_ACT=8;  U_RESTRICT="" ;;
    *) echo "HATA: bilinmeyen UNET_MODE=$UNET_MODE"; exit 1 ;;
  esac
  echo "### 4) UNet -> QNN ($UNET_MODE, graf 'model')"
  ( cd "$SDIR" && ACT_BW="$U_ACT" WEIGHT_BW="${UNET_WEIGHT_BW:-8}" \
      RESTRICT_STEPS="$U_RESTRICT" \
      ./03_convert_qnn.sh "../$WORK/onnx/unet_${TAG}.onnx" \
      model quant "../$WORK/calib/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" unet )
fi

# ---- 5) VAE decoder -> QNN (int8; HTP GroupNorm'u float'ta desteklemez) ----
# 5a) VAE decoder kalibrasyonu (nihai olceksiz latent'ler)
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib_vae/$TAG/input_list.txt" ]; then
  echo "### 5a) VAE decoder kalibrasyon [ATLANDI]"
else
  echo "### 5a) VAE decoder kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib_vae/$TAG" --target vae_decoder \
      --num-samples "${VAE_CALIB_N:-6}" --steps "${VAE_CALIB_STEPS:-10}"
fi
# 5b) VAE decoder -> QNN (int8, a8w8)
if [ "$FORCE" = 0 ] && [ -f "$WORK/qnn/vae_decoder.bin" ]; then
  echo "### 5b) VAE decoder -> QNN [ATLANDI]"
else
  echo "### 5b) VAE decoder -> QNN (a8w8)"
  ( cd "$SDIR" && WEIGHT_BW="${VAE_WEIGHT_BW:-8}" ./03_convert_qnn.sh "../$WORK/onnx/vae_decoder.onnx" \
      vae_decoder quant "../$WORK/calib_vae/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" vae_decoder )
fi

# ---- 6) VAE encoder -> QNN (int8) — motor varsayilan olarak yukler! ---------
# Motor --no_img2img verilmedikce VAE encoder'i yuklemeye calisir; dosya yoksa
# 'kod 1' ile coker. Bu yuzden encoder de uretilir (int8, goruntu kalibrasyonu).
# 6a) VAE encoder kalibrasyonu (decode edilmis goruntuler)
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib_venc/$TAG/input_list.txt" ]; then
  echo "### 6a) VAE encoder kalibrasyon [ATLANDI]"
else
  echo "### 6a) VAE encoder kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib_venc/$TAG" --target vae_encoder \
      --num-samples "${VAE_CALIB_N:-6}" --steps "${VAE_CALIB_STEPS:-10}"
fi
# 6b) VAE encoder -> QNN (int8, a8w8)
if [ "$FORCE" = 0 ] && [ -f "$WORK/qnn/vae_encoder.bin" ]; then
  echo "### 6b) VAE encoder -> QNN [ATLANDI]"
else
  echo "### 6b) VAE encoder -> QNN (a8w8)"
  ( cd "$SDIR" && WEIGHT_BW="${VAE_WEIGHT_BW:-8}" ./03_convert_qnn.sh "../$WORK/onnx/vae_encoder.onnx" \
      vae_encoder quant "../$WORK/calib_venc/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" vae_encoder )
fi

# ---- 7) paketle -----------------------------------------------------------
echo "### 7) paketle"
python3 "$SDIR/05_package.py" --name "$NAME" --tier "$TIER" --qnn-version "$QNN_VERSION" \
    --onnx "$WORK/onnx" --mnn "$WORK/mnn" --qnn "$WORK/qnn" \
    --tokenizer "$WORK/onnx" --output dist

TAIL=""; [ "$TIER" = min ] && TAIL="_min"; [ "$TIER" = high ] && TAIL="_8gen3"
echo
echo "########################################################"
echo " BITTI. Cikti: dist/${NAME}_qnn${QNN_VERSION}${TAIL}.zip"
echo " Ruya/Local Dream 'Import Custom Model' ile alin."
echo "########################################################"
