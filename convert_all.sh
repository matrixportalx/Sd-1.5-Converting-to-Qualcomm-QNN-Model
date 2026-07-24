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

# ---- 0) safetensors -> diffusers ------------------------------------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/pipeline/model_index.json" ]; then
  echo "### 0) safetensors -> diffusers [ATLANDI]"
else
  echo "### 0) safetensors -> diffusers"
  python3 "$SDIR/00_load_safetensors.py" --checkpoint "$CKPT" --output "$WORK/pipeline"
fi

# ---- 1) ONNX/emb export (surum damgali) -----------------------------------
# v6: VAE decoder'dan Div (latent/scale) kaldirildi (HTP float Div hatasi).
EXPORT_VERSION="6"
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
  echo "### 4) UNet -> QNN (int8, a${ACT_BW}w8)"
  ( cd "$SDIR" && ./03_convert_qnn.sh "../$WORK/onnx/unet_${TAG}.onnx" \
      unet quant "../$WORK/calib/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" unet )
fi

# ---- 5) VAE decoder -> QNN (fp16, graf 'vae_decoder') ---------------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/qnn/vae_decoder.bin" ]; then
  echo "### 5) VAE decoder -> QNN [ATLANDI]"
else
  echo "### 5) VAE decoder -> QNN (fp16)"
  ( cd "$SDIR" && ./03_convert_qnn.sh "../$WORK/onnx/vae_decoder.onnx" \
      vae_decoder float - "$TIER" "../$WORK/qnn" vae_decoder )
fi

# ---- 6) VAE encoder -> QNN (fp16, opsiyonel; hata olursa devam) ------------
if [ "$FORCE" = 0 ] && [ -f "$WORK/qnn/vae_encoder.bin" ]; then
  echo "### 6) VAE encoder -> QNN [ATLANDI]"
else
  echo "### 6) VAE encoder -> QNN (fp16, opsiyonel)"
  ( cd "$SDIR" && ./03_convert_qnn.sh "../$WORK/onnx/vae_encoder.onnx" \
      vae_encoder float - "$TIER" "../$WORK/qnn" vae_encoder ) \
    || echo "  [!] vae_encoder basarisiz — img2img olmadan devam (txt2img calisir)"
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
