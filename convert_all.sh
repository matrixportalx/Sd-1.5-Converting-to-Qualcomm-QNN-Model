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

# ---- config.env: NOT DEFTERINE DOKUNMADAN ayar degistirme -------------------
# Not defterini yeniden acmak Colab'da yeni calisma zamani = her seyi sifirdan
# derlemek demek. Bunu onlemek icin ayarlar depodaki config.env'den okunur;
# 2. adimdaki `git reset --hard` dosyayi guncelledigi icin AYNI OTURUMDA
# yeni ayarlarla kosulabilir.
#   OVERRIDE_<AD>=<deger>  ->  <AD> notebook ne derse desin bu degeri alir.
_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.env"
if [ -f "$_CFG" ]; then
  _applied=""
  while IFS='=' read -r _k _v; do
    case "$_k" in
      OVERRIDE_*)
        _name="${_k#OVERRIDE_}"
        _v="${_v%%#*}"; _v="${_v%"${_v##*[![:space:]]}"}"   # yorum + bosluk kirp
        export "$_name=$_v"
        _applied="$_applied $_name=$_v"
        ;;
    esac
  done < <(grep -E '^[[:space:]]*OVERRIDE_[A-Z_]+=' "$_CFG" | sed 's/^[[:space:]]*//')
  [ -n "$_applied" ] && echo "[config.env] EZILEN AYARLAR:$_applied"
fi

# VAE'ler her tier'da a8w8 (UNet'in bit genisligi adim 4'te ayrica verilir)
export ACT_BW="${ACT_BW:-8}"

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
# v12: 16-bit graf siniri ile 8-bit ic graf arasina Clip bariyeri (agirliksiz op;
# QNN 16-bit etiketini to_k/to_v uzerinden ic grafa tasiyamasin).
# v11: UNet a16w8 + restrict steps. v10: referans recete (16-bit I/O, graf "model", v68).
EXPORT_VERSION="12"
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
# FAST_TRIAL=1 : kalibrasyon ornegini asgariye indirir. Kuantizasyon suresi
# ornek sayisiyla dogru orantili (16 ornek ~8 dk, 2 ornek ~1 dk) — boru hattini
# dogrulamak icin ideal. NIHAI paket icin FAST_TRIAL=0 ile bir kez daha kosun.
if [ "${FAST_TRIAL:-0}" = "1" ]; then
  CALIB_PROMPTS="${CALIB_PROMPTS:-1}"; CALIB_STEPS="${CALIB_STEPS:-2}"
  VAE_CALIB_N="${VAE_CALIB_N:-1}";     VAE_CALIB_STEPS="${VAE_CALIB_STEPS:-2}"
  echo "[*] FAST_TRIAL=1 -> kalibrasyon kucultuldu (kalite dusuk, hizli dogrulama)"
else
  CALIB_PROMPTS="${CALIB_PROMPTS:-4}"; CALIB_STEPS="${CALIB_STEPS:-4}"
  VAE_CALIB_N="${VAE_CALIB_N:-6}";     VAE_CALIB_STEPS="${VAE_CALIB_STEPS:-10}"
fi

# v2: timestep raw'lari INT_32 (eskiden float32 -> quantizer bit desenini
# tamsayi okuyup zaman-gomme encoding'lerini bozuyordu)
# Damgaya ornek sayilari da giriyor -> FAST_TRIAL degisince kalibrasyon yenilenir.
CALIB_VERSION="2 n=${CALIB_PROMPTS} s=${CALIB_STEPS}"
CSTAMP="$WORK/calib/$TAG/.calib_version"
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib/$TAG/input_list.txt" ] \
   && [ "$(cat "$CSTAMP" 2>/dev/null)" = "$CALIB_VERSION" ]; then
  echo "### 3) UNet kalibrasyon [ATLANDI]"
else
  echo "### 3) UNet kalibrasyon verisi (v$CALIB_VERSION)"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib/$TAG" --mode real \
      --num-samples "$CALIB_PROMPTS" --steps "$CALIB_STEPS"
  echo "$CALIB_VERSION" > "$CSTAMP"
  # kalibrasyon degisti -> kuantize UNet DLC gecersiz
  rm -f "$WORK/qnn/build/model/model_q.dlc" "$WORK/qnn/build/model/model_q.args" \
        "$WORK/qnn/unet.bin"
fi

# ---- 4) UNet -> QNN (graf 'model') ----------------------------------------
# Referans binary: I/O UFIXED_POINT_16 + timestamp INT_32, dspArch 68.
# Motor bu tipleri SABIT bekledigi icin 16-bit I/O zorunlu.
#
# v68'de 16-bit MatMul'un kabul edilmesi icin quantizer'a hedef backend/SoC
# soylenir (--target_backend HTP --target_soc_model SM8350); ayrica
# --restrict_quantization_steps + simetrik/per-row sema verilir.
# Ayrintilar: scripts/03_convert_qnn.sh
#
# UNET_MODE:
#   a16w8_restrict (varsayilan) : referans recete
#   a16w8                       : restrict yok (v73+ gerektirir)
#   a8w8_io16                   : ic hesap 8-bit + graf sinirlari 16-bit
#                                 (karma hassasiyet — referansin yaptigi)
#   a8w8                        : tam 8-bit — DERLENIR ama motor uint16
#                                 yazdigi icin CIHAZDA YUKLENMEZ (yalnizca
#                                 boru hattini test etmek icin)
UNET_MODE="${UNET_MODE:-a16w8_restrict}"
U_OVERRIDES=""      # yalnizca UNet'e verilir; VAE adimlarina SIZMAMALI
case "$UNET_MODE" in
  a16w8_restrict) U_ACT=16; U_RESTRICT="${UNET_RESTRICT:-auto}" ;;
  a16w8)          U_ACT=16; U_RESTRICT="" ;;
  a8w8)           U_ACT=8;  U_RESTRICT="" ;;
  a8w8_io16)
    # KARMA HASSASIYET: ic hesap 8-bit (v68'de LayerNorm/Conv/MatMul hepsi
    # gecerli), YALNIZCA graf sinir tensorleri 16-bit — referansin
    # UFIXED_POINT_16 I/O'su boyle elde edilir. QNN sinirla ic graf arasina
    # Convert op'lari ekler.
    U_ACT=8; U_RESTRICT=""
    echo "  [io16] sinir tensorleri icin 16-bit encoding uretiliyor"
    python3 "$SDIR/gen_io_encodings.py" --calib "$WORK/calib/$TAG" \
        --output "$WORK/onnx/unet_io_encodings.json"
    U_OVERRIDES="$(cd "$WORK/onnx" && pwd)/unet_io_encodings.json"
    ;;
  *) echo "HATA: bilinmeyen UNET_MODE=$UNET_MODE"; exit 1 ;;
esac
echo "### 4) UNet -> QNN ($UNET_MODE, graf 'model')"
( cd "$SDIR" && ACT_BW="$U_ACT" WEIGHT_BW="${UNET_WEIGHT_BW:-8}" \
    RESTRICT_STEPS="$U_RESTRICT" QUANT_OVERRIDES="$U_OVERRIDES" \
    ./03_convert_qnn.sh "../$WORK/onnx/unet_${TAG}.onnx" \
    model quant "../$WORK/calib/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" unet )

# ---- 5) VAE decoder -> QNN (int8; HTP GroupNorm'u float'ta desteklemez) ----
# 5a) VAE decoder kalibrasyonu (nihai olceksiz latent'ler)
VSTAMP="$WORK/calib_vae/$TAG/.calib_version"
VCAL_V="n=${VAE_CALIB_N} s=${VAE_CALIB_STEPS}"
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib_vae/$TAG/input_list.txt" ] \
   && [ "$(cat "$VSTAMP" 2>/dev/null)" = "$VCAL_V" ]; then
  echo "### 5a) VAE decoder kalibrasyon [ATLANDI]"
else
  echo "### 5a) VAE decoder kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib_vae/$TAG" --target vae_decoder \
      --num-samples "$VAE_CALIB_N" --steps "$VAE_CALIB_STEPS"
  echo "$VCAL_V" > "$VSTAMP"
  rm -f "$WORK/qnn/build/vae_decoder/vae_decoder_q.dlc" "$WORK/qnn/vae_decoder.bin"
fi
# 5b) VAE decoder -> QNN (int8, a8w8)
echo "### 5b) VAE decoder -> QNN (a8w8)"
( cd "$SDIR" && WEIGHT_BW="${VAE_WEIGHT_BW:-8}" ./03_convert_qnn.sh "../$WORK/onnx/vae_decoder.onnx" \
    vae_decoder quant "../$WORK/calib_vae/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" vae_decoder )

# ---- 6) VAE encoder -> QNN (int8) — motor varsayilan olarak yukler! ---------
# Motor --no_img2img verilmedikce VAE encoder'i yuklemeye calisir; dosya yoksa
# 'kod 1' ile coker. Bu yuzden encoder de uretilir (int8, goruntu kalibrasyonu).
# 6a) VAE encoder kalibrasyonu (decode edilmis goruntuler)
ESTAMP="$WORK/calib_venc/$TAG/.calib_version"
if [ "$FORCE" = 0 ] && [ -f "$WORK/calib_venc/$TAG/input_list.txt" ] \
   && [ "$(cat "$ESTAMP" 2>/dev/null)" = "$VCAL_V" ]; then
  echo "### 6a) VAE encoder kalibrasyon [ATLANDI]"
else
  echo "### 6a) VAE encoder kalibrasyon verisi"
  python3 "$SDIR/02_gen_quant_data.py" --pipeline "$WORK/pipeline" \
      --resolution "$TAG" --output "$WORK/calib_venc/$TAG" --target vae_encoder \
      --num-samples "$VAE_CALIB_N" --steps "$VAE_CALIB_STEPS"
  echo "$VCAL_V" > "$ESTAMP"
  rm -f "$WORK/qnn/build/vae_encoder/vae_encoder_q.dlc" "$WORK/qnn/vae_encoder.bin"
fi
# 6b) VAE encoder -> QNN (int8, a8w8)
echo "### 6b) VAE encoder -> QNN (a8w8)"
( cd "$SDIR" && WEIGHT_BW="${VAE_WEIGHT_BW:-8}" ./03_convert_qnn.sh "../$WORK/onnx/vae_encoder.onnx" \
    vae_encoder quant "../$WORK/calib_venc/${TAG}/input_list.txt" "$TIER" "../$WORK/qnn" vae_encoder )

# ---- 6c) uretilen unet.bin'in I/O tiplerini dogrula ------------------------
# Motor sample/text_embedding'e uint16, timestamp'e int32 yazar. Tipler
# tutmuyorsa paketi telefona atmadan BURADA ogrenelim.
echo "### 6c) unet.bin I/O tip dogrulamasi"
if [ -f "$WORK/qnn/unet.arch" ]; then
  echo "  [dsp_arch] unet.bin -> $(cat "$WORK/qnn/unet.arch") (referans: v68)"
fi
python3 "$SDIR/check_bin_io.py" --bin "$WORK/qnn/unet.bin" --expect unet \
  ${STRICT_IO:+--strict} || true

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
