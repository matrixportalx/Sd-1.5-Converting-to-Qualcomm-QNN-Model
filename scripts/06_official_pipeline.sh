#!/usr/bin/env bash
#
# RESMI HAT — Local Dream'in kendi npuconvertv2 scriptlerini Colab'da kosar.
#
# NEDEN: kendi hattimiz (qairt-converter -> DLC -> context-bin) cihazda
# yuklenmeyen paketler uretti. Resmi scriptler elimize gecince sebep anlasildi;
# uc temel fark var ve ucu de bizim tarafta cozulemez cinsten:
#
#   1) ARAC ZINCIRI FARKLI
#        qnn-onnx-converter -> model.cpp/bin
#        qnn-model-lib-generator -> libmodel.so
#        qnn-context-binary-generator --model libmodel.so
#      Girdi sirasi bu yolda model.cpp'deki BILDIRIM sirasidir. Bizim DLC
#      yolumuzda sira "grafta ilk tuketim" kuralina gore olusuyordu ve
#      degistirilemiyordu (olculdu).
#
#   2) --act_bitwidth 16
#      Aktivasyonlar 16-bit; sample/text_embedding/output boylece dogal olarak
#      UFIXED_POINT_16 oluyor. Bizim ugrastigimiz --config/io16 numarasina
#      gerek yok. (Biz a16w8'i "LayerNorm 16-bit yok" diye elemistik; o hata
#      qairt-converter 2.39'a ve bizim ONNX'imize ozguydu.)
#
#   3) MODEL YENIDEN YAZILMIS
#      redefined_modules/ altinda diffusers'in HTP dostu kopyalari var
#      (CrossAttention'da Linear->Conv, MHA->SHA vb). ONNX'i HTP'ye uygun
#      yapan sey bayraklar degil, modelin kendisi.
#
#   Ayrica htp_config_min.json'da "vtcm_mb": 2 — biz VTCM'yi hic ayarlamiyorduk.
#
# Kullanim:
#   scripts/06_official_pipeline.sh <ckpt> <isim> <work_dir> [min|8gen1|8gen2]
set -euo pipefail

CKPT="${1:?safetensors yolu}"
NAME="${2:?model adi}"
WORK="${3:?work dizini}"
SOC="${4:-min}"

: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali (2.28 olmali)}"

SDIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$WORK/_official/npuconvertv2"
if [ ! -d "$SRC" ]; then
  echo "### resmi scriptler indiriliyor (npuconvertv2)"
  python3 "$SDIR_SELF/fetch_official_scripts.py" --dest "$WORK/_official" \
      ${OFFICIAL_SCRIPTS_URL:+--url "$OFFICIAL_SCRIPTS_URL"} --no-dump
fi
[ -d "$SRC" ] || { echo "HATA: resmi scriptler alinamadi -> $SRC"; exit 1; }
# TUM yollar MUTLAK olmali: asagida `cd "$SRC"` yapiyoruz ve goreli yollar
# o andan itibaren yanlis yeri gosteriyor (ilk surumun hatasi buydu).
SRC="$(cd "$SRC" && pwd)"
DIST="$(pwd)/dist"

CLIP_SKIP="${CLIP_SKIP:-2}"
REALISTIC="${REALISTIC:-1}"      # CyberRealistic gibi foto modeller icin 1
CALIB_LIMIT="${CALIB_LIMIT:-0}"  # 0 = kirpma yok (resmi: 400 ornek)

echo "=========================================================="
echo " RESMI HAT: $NAME  (soc=$SOC, clip_skip=$CLIP_SKIP)"
echo " SDK: $QNN_SDK_ROOT"
echo "=========================================================="

# 2.28 kontrolu — yanlis surumle kosmak saatleri bosa harcar
case "$QNN_SDK_ROOT" in
  *2.28*) ;;
  *) echo "  [!] UYARI: SDK 2.28 degil. Rehber 2.28 sart kosuyor."
     echo "      config.env -> OVERRIDE_QAIRT_ASSET_URL (2.28 satiri)" ;;
esac

# Bizim scriptler goreli yol kullaniyor; resmi scriptler de oyle -> cd sart.
if [ ! -s "$CKPT" ]; then
  echo "HATA: model dosyasi yok/bos -> $CKPT"
  echo "      Resmi hat safetensors'i DOGRUDAN kullaniyor (pipeline/ degil);"
  echo "      not defterinin 5. adimini (Modeli indir) calistirin."
  exit 1
fi
ABS_CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
ABS_SDK="$(cd "$QNN_SDK_ROOT" && pwd)"
cd "$SRC"

# ZIP calistirma bitlerini korumuyor -> paketle gelen MNNConvert ikilisi ve
# .sh scriptleri "Permission denied" veriyor (convert_clip.sh line 6).
chmod +x MNNConvert scripts/*.sh 2>/dev/null || true

# ---- Python ortami --------------------------------------------------------
# Resmi pyproject.toml diffusers==0.31.0 / transformers==4.46.1 / numpy 1.26.4
# istiyor; Colab'in kendi surumleri bunlarla uyusmuyor ve redefined_modules
# eski API'lere dayaniyor. Ayrica QNN 2.28 araclari da Python 3.10 istiyor —
# ikisini TEK venv'de topluyoruz (pyproject zaten onnx/pandas/pyyaml iceriyor).
# CUDA_TORCH: resmi pyproject torch'un CPU surumunu SABITLIYOR
#     torch==2.5.1+cpu   +   index https://download.pytorch.org/whl/cpu
# Bu yuzden GPU'lu bir calisma zamaninda bile difuzyon CPU'da kosuyor
# (~3.85 sn/adim). Rehber de bunu soyluyor: "If you have a CUDA-capable GPU,
# you can edit pyproject.toml to use the GPU build of torch."
# NOT: GPU yalnizca prepare_data.py'yi hizlandirir. Kuantizasyon
# (qnn-onnx-converter), model-lib-generator ve context-binary-generator
# tamamen CPU'dur ve bundan etkilenmez.
CUDA_TORCH="${CUDA_TORCH:-auto}"
if [ "$CUDA_TORCH" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    CUDA_TORCH=1
  else
    CUDA_TORCH=0
  fi
fi
CU="${CUDA_WHL:-cu121}"

# uv.lock RESMI TARIFIN TEST EDILMIS SURUMLERINI tutuyor. Onceki surumde CUDA
# icin pyproject'i duzenleyip uv.lock'u SILMISTIK; uv o zaman en guncelleri
# cekti (onnx 1.22, protobuf 7.35) ve QNN 2.28 onlarla calismiyor:
#   AttributeError: 'NoneType' object has no attribute 'AttributeProto'
# (SDK'nin onnx shim'i desteklenmeyen surumde None donuyor.)
# Artik pyproject/uv.lock'a DOKUNMUYORUZ; CUDA torch kilitli kurulumun
# USTUNE ayrica yukleniyor.
ZIP_PATH="$(dirname "$SRC")/npuconvert.zip"
if [ -f "$ZIP_PATH" ]; then
  ( cd "$(dirname "$SRC")" && unzip -o -q "$ZIP_PATH" \
      "npuconvertv2/pyproject.toml" "npuconvertv2/uv.lock" 2>/dev/null ) || true
fi

# Ortam damgasi: kurulum sekli degistiginde venv yeniden kurulsun.
ENV_STAMP="$SRC/.venv/.setup_version"
ENV_WANT="v2 cuda=$CUDA_TORCH $CU"
VENV_PY="$SRC/.venv/bin/python"
if [ ! -x "$VENV_PY" ] || [ "$(cat "$ENV_STAMP" 2>/dev/null)" != "$ENV_WANT" ]; then
  echo "### resmi Python ortami kuruluyor ($ENV_WANT)"
  command -v uv >/dev/null 2>&1 || pip install -q uv
  command -v uv >/dev/null 2>&1 || { echo "HATA: uv kurulamadi"; exit 1; }
  uv venv -p 3.10 --clear
  uv sync                       # KILITLI surumler — QNN 2.28 ile uyumlu
  if [ "$CUDA_TORCH" = "1" ]; then
    echo "  [cuda] CUDA torch kilitli kurulumun ustune ekleniyor ($CU)"
    nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/         /'
    uv pip install --python "$SRC/.venv/bin/python" "torch==2.5.1" \
        --index-url "https://download.pytorch.org/whl/$CU" \
      || echo "  [!] CUDA torch kurulamadi — CPU torch ile devam"
  else
    echo "  [cuda] GPU yok/kapali -> CPU torch (prepare_data yavas olacak)"
  fi
  echo "$ENV_WANT" > "$ENV_STAMP"
fi
if [ ! -x "$VENV_PY" ]; then
  echo "HATA: resmi Python ortami olusmadi -> $VENV_PY"
  echo "      (uv venv/uv sync ciktisina bakin)"
  exit 1
fi
export PATH="$SRC/.venv/bin:$PATH"
export VIRTUAL_ENV="$SRC/.venv"
echo "  [python] $("$VENV_PY" -V)  ($VENV_PY)"
"$VENV_PY" - <<'PYV' || true
import importlib
for m in ("onnx", "protobuf", "numpy", "torch"):
    try:
        mod = importlib.import_module("google.protobuf" if m == "protobuf" else m)
        print(f"  [surum] {m:9s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [surum] {m:9s} YOK ({type(e).__name__})")
PYV

# ---- Modeli yerine koy ----------------------------------------------------
# prepare_data.py/export_onnx.py --model_path bekliyor; mutlak yol veriyoruz.
REAL_FLAG=""
[ "$REALISTIC" = "1" ] && REAL_FLAG="--realistic"

# ---- 1) Kalibrasyon verisi (gercek difuzyon kosusu) -----------------------
if [ ! -f "data.pkl" ]; then
  echo "### 1) prepare_data.py (20 prompt x difuzyon — EN UZUN ADIM)"
  "$VENV_PY" -c "import torch;print('    [torch]', torch.__version__,
        'cuda:', torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
  "$VENV_PY" prepare_data.py --model_path "$ABS_CKPT" --clip_skip "$CLIP_SKIP" $REAL_FLAG
else
  echo "### 1) prepare_data.py [ATLANDI - data.pkl var]"
fi

if [ ! -f "input_list_unet.txt" ]; then
  echo "### 2) gen_quant_data.py"
  "$VENV_PY" gen_quant_data.py
else
  echo "### 2) gen_quant_data.py [ATLANDI]"
fi

# Hizli deneme: kalibrasyon listesini kirp. Resmi hat 400 ornek kullaniyor ve
# kuantizasyon saatler suruyor. Once BORU HATTININ calistigini dogrulamak icin
# kucuk bir sayi, sonra CALIB_LIMIT=0 ile tam kalite.
if [ "$CALIB_LIMIT" -gt 0 ] 2>/dev/null; then
  for f in input_list_unet.txt input_list_vae_decoder.txt input_list_vae_encoder.txt; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    if [ "$n" -gt "$CALIB_LIMIT" ]; then
      head -n "$CALIB_LIMIT" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
      echo "  [hizli] $f: $n -> $CALIB_LIMIT satir"
    fi
  done
fi

# ---- 3) ONNX export (redefined_modules ile) -------------------------------
if [ ! -f "unet/model.onnx" ]; then
  echo "### 3) export_onnx.py (redefined_modules: MHA->SHA, Linear->Conv)"
  "$VENV_PY" export_onnx.py --model_path "$ABS_CKPT" --clip_skip "$CLIP_SKIP"
else
  echo "### 3) export_onnx.py [ATLANDI - unet/model.onnx var]"
fi

# ---- 3b) Sistem bagimliliklari --------------------------------------------
# Colab imajinda iki sey eksik:
#   1) libc++  — QNN 2.28'in Python baglantilari (libPyIrGraph) LLVM libc++'a
#      bagli: "ImportError: libc++.so.1: cannot open shared object file"
#   2) clang++ — qnn-model-lib-generator uretilen model.cpp'yi clang ile
#      derliyor: "Could not find compiler: clang++"
_need_libcxx=0; _need_clang=0
ldconfig -p 2>/dev/null | grep -q 'libc++\.so\.1' || _need_libcxx=1
command -v clang++ >/dev/null 2>&1 || _need_clang=1
if [ "$_need_libcxx" = "1" ] || [ "$_need_clang" = "1" ]; then
  echo "### 3b) sistem bagimliliklari kuruluyor" \
       "$([ "$_need_libcxx" = 1 ] && echo libc++)" \
       "$([ "$_need_clang" = 1 ] && echo clang)"
  (apt-get -qq update -y >/dev/null 2>&1 || true)
  if [ "$_need_libcxx" = "1" ]; then
    apt-get -qq install -y libc++1 libc++abi1 >/dev/null 2>&1 \
      || apt-get -qq install -y libc++1-14 libc++abi1-14 >/dev/null 2>&1 || true
  fi
  if [ "$_need_clang" = "1" ]; then
    apt-get -qq install -y clang >/dev/null 2>&1 \
      || apt-get -qq install -y clang-14 >/dev/null 2>&1 || true
    # bazi paketler yalnizca clang++-14 birakiyor -> genel adi baglayalim
    if ! command -v clang++ >/dev/null 2>&1; then
      for v in 18 17 16 15 14; do
        if command -v "clang++-$v" >/dev/null 2>&1; then
          ln -sf "$(command -v "clang++-$v")" /usr/local/bin/clang++
          ln -sf "$(command -v "clang-$v")" /usr/local/bin/clang 2>/dev/null || true
          break
        fi
      done
    fi
  fi
  ldconfig 2>/dev/null || true
fi
ldconfig -p 2>/dev/null | grep -q 'libc++\.so\.1' \
  && echo "  [deps] libc++ hazir" \
  || echo "  [!] libc++ YOK — qnn-onnx-converter calismayabilir"
if command -v clang++ >/dev/null 2>&1; then
  echo "  [deps] $(clang++ --version 2>/dev/null | head -1)"
else
  echo "  [!] clang++ YOK — qnn-model-lib-generator derleyemez"
  echo "      Elle: apt-get install -y clang"
fi

# ---- 4) QNN donusumu ------------------------------------------------------
# Resmi convert_all.sh SDK yolunu SABIT kodluyor (/data/qairt/2.28.0.241029);
# bizimkine cevirmek icin gecici bir kopya uretiyoruz.
echo "### 4) QNN donusumu (qnn-onnx-converter -> model-lib -> context-bin)"
for f in scripts/convert_all.sh scripts/convert_all_unet_only.sh; do
  [ -f "$f" ] || continue
  sed -i "s|^QNN_SDK_ROOT=.*|QNN_SDK_ROOT=$ABS_SDK|" "$f"
done
echo "  [sdk] convert_all.sh -> QNN_SDK_ROOT=$ABS_SDK"

# envsetup.sh 'source' edilmesi gerekiyor; resmi script bunu kendisi yapiyor.
bash scripts/convert_all.sh --min_soc "$SOC"

OUT="output/qnn_models_$SOC"
echo
echo "### 5) cikti"
ls -la "$OUT" | sed 's/^/    /'

# ---- 6) Paket -------------------------------------------------------------
# Referans paketlerde dosyalar ZIP KOKUNDE (klasor yok) — resmi export.sh
# 'zip -r ... output_512/qnn_models_min' yaptigi icin orada klasorlu; biz
# uygulamanin bekledigi duz yapiyi uretiyoruz.
mkdir -p "$DIST"
ZIP="$DIST/${NAME}_qnn2.28_${SOC}.zip"
rm -f "$ZIP"
( cd "$OUT" && zip -q -r "$ZIP" . )
echo
echo "########################################################"
echo " BITTI. Cikti: $ZIP"
unzip -l "$ZIP" | sed 's/^/    /'
echo "########################################################"
