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
# ---------------------------------------------------------------------------
# COZUNURLUK
#
# QNN'de cozunurluk istek parametresi DEGIL, derleme zamani ozelligidir:
# unet.bin sabit tensor sekilleriyle derlenir. Uygulama (Ruya / Local Dream)
# taban 512x512 binary'sini yukler ve baska bir boyut istendiginde model
# klasorundeki zstd YAMASINI acilista unet.bin'e uygular:
#
#     768.patch        -> 768x768   (kare yamalar tek sayiyla adlandirilir)
#     512x768.patch    -> 512x768   (dikdortgen yamalar WxH)
#
# Resmi export.sh de tam olarak bunu yapar: her ek cozunurluk icin
# prepare_data -> gen_quant_data -> export_onnx_unet_only -> convert_all_unet_only
# kosulur, cikan unet.bin taban unet.bin'e karsi 'zstd --patch-from' ile
# farklanir ve yama paketin icine konur. VAE/CLIP yamalanmaz; yalnizca UNet.
#
#   RESOLUTIONS="512x768,768x512,768x768" scripts/06_official_pipeline.sh ...
#
# 512x512 her zaman uretilir (yamalarin taban aldigi binary odur), listede
# yazmaya gerek yoktur. Her ek cozunurluk TAM bir kalibrasyon + kuantizasyon
# turudur: sure ve RAM taban kosunun aynisi kadar artar.
# ---------------------------------------------------------------------------
#
# Kullanim:
#   scripts/06_official_pipeline.sh <ckpt> <isim> <work_dir> [min|8gen1|8gen2]
set -euo pipefail

CKPT="${1:?safetensors yolu}"
NAME="${2:?model adi}"
WORK="${3:?work dizini}"
SOC="${4:-min}"

# Paket adina surum+SOC ekini asagida BIZ ekliyoruz. Kullanici modeli hedef
# dosya adiyla ("CyberRealistic_qnn2.28_8gen2") adlandirdiginda ek iki kez
# cikiyordu -> CyberRealistic_qnn2.28_8gen2_qnn2.28_8gen2.zip. Varsa kirp.
NAME_GIRILEN="$NAME"
NAME="$(printf '%s' "$NAME" | sed -E 's/_qnn[0-9]+\.[0-9]+(_(min|8gen[0-9]+))?$//')"
[ -z "$NAME" ] && { echo "HATA: model adi yalnizca surum ekinden olusuyor: $NAME_GIRILEN"; exit 1; }
if [ "$NAME" != "$NAME_GIRILEN" ]; then
  echo "  [ad] '$NAME_GIRILEN' -> '$NAME' (surum/SOC eki paket adina zaten eklenir)"
fi
# Dosya sistemi / HF yolu icin guvenli ad. Bosluksuz adlarda NAME ile ayni.
SLUG="$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9._-' '_')"

: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali (2.28 olmali)}"

# ---- Cozunurluk listesi ---------------------------------------------------
# Taban her zaman 512x512. RESOLUTIONS yalnizca EK boyutlari sayar; 512x512
# yazilirsa sessizce cikarilir (zaten uretiliyor).
BASE_RES="512x512"
EXTRA_RES=""
for _item in $(printf '%s' "${RESOLUTIONS:-}" | tr ',;Xx' '  xx' | tr -s ' '); do
  case "$_item" in
    *x*) ;;
    *) echo "HATA: cozunurluk 'GENISLIKxYUKSEKLIK' olmali (or. 768x512): $_item"; exit 1 ;;
  esac
  _w="${_item%%x*}"; _h="${_item##*x}"
  case "$_w$_h" in
    ""|*[!0-9]*) echo "HATA: cozunurluk sayisal degil: $_item"; exit 1 ;;
  esac
  # SD1.5 UNet 3 kez yari boyuta iner -> latent 8'in, piksel 64'un kati olmali.
  if [ $((_w % 64)) -ne 0 ] || [ $((_h % 64)) -ne 0 ]; then
    echo "HATA: $_item — kenarlar 64'un kati olmali (512, 576, 640, 704, 768, 1024 ...)"
    exit 1
  fi
  if [ "$_w" = 512 ] && [ "$_h" = 512 ]; then
    echo "  [cozunurluk] 512x512 taban — ek listeden cikarildi"
    continue
  fi
  case " $EXTRA_RES " in *" ${_w}x${_h} "*) continue ;; esac
  EXTRA_RES="$EXTRA_RES ${_w}x${_h}"
done
EXTRA_RES="$(printf '%s' "$EXTRA_RES" | sed 's/^ *//')"

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
OFF_DIR="$(cd "$WORK/_official" && pwd)"

# YOLDA BOSLUK: model adi "epiCRealism Natural Sin" gibi bosluk iceriyorsa
# calisma dizini de bosluklu oluyor ve resmi scriptler bunu kaldirmiyor:
#   convert_all.sh:38  cd ${current_pwd}   -> "cd: too many arguments"
# (Alt scriptler goreli yol kullandigi icin hata yalnizca burada patliyor;
# ayrica qnn-model-lib-generator'in urettigi Makefile de bosluklu yolda
# derlenmiyor.) Cozum: bosluksuz bir symlink uzerinden calis — venv ve
# uretilmis dosyalar YERINDE kalir, tasima ya da yeniden kurulum gerekmez.
case "$OFF_DIR" in
  *[[:space:]]*)
    _link="${TMPDIR:-/tmp}/sdqnn/$SLUG"
    mkdir -p "$(dirname "$_link")"
    ln -sfn "$OFF_DIR" "$_link"
    OFF_DIR="$_link"
    echo "  [yol] calisma dizininde bosluk var -> bosluksuz baglanti: $OFF_DIR"
    ;;
esac

SRC="$OFF_DIR/npuconvertv2"
DIST="$(pwd)/dist"

CLIP_SKIP="${CLIP_SKIP:-2}"
REALISTIC="${REALISTIC:-1}"      # CyberRealistic gibi foto modeller icin 1
CALIB_LIMIT="${CALIB_LIMIT:-0}"  # 0 = kirpma yok (resmi: 400 ornek)

echo "=========================================================="
echo " RESMI HAT: $NAME  (soc=$SOC, clip_skip=$CLIP_SKIP)"
echo " SDK: $QNN_SDK_ROOT"
echo " Cozunurluk: $BASE_RES (taban)${EXTRA_RES:+ + yama: $EXTRA_RES}"
echo "=========================================================="

if [ -n "$EXTRA_RES" ]; then
  # Resmi export.sh ek cozunurlukleri YALNIZCA 8gen1/8gen2 icin uretir:
  # "Non-flagship SOC versions can't run higher resolutions".
  if [ "$SOC" = "min" ]; then
    echo "  [!] SOC=min ile ek cozunurluk isteniyor. Resmi tarif bunu yapmiyor:"
    echo "      dusuk HTP (v68) kusaklari yuksek cozunurlugu kaldiramiyor."
    echo "      Yama uretilir ama cihazda yuklenmeyebilir; 8gen1/8gen2 onerilir."
  fi
  case " $EXTRA_RES " in
    *1024*)
      echo "  [!] 1024 kenarli cozunurluk var. Kuantizasyon RAM'i 512'ye gore"
      echo "      ~4x artar (20 GB alt sinir -> cok daha fazlasi) ve cihaz"
      echo "      tarafinda VTCM'ye sigmayabilir. Once 768 ile dogrulayin."
      ;;
  esac
fi

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
# NOT: damga CUDA'yi ICERMEZ. CUDA torch asagida ayri, fikirsiz (idempotent)
# bir adim olarak kuruluyor; boylece kurulum tutmadiginda bir sonraki kosu
# venv'i BASTAN kurmadan yalnizca torch'u tekrar deniyor.
ENV_STAMP="$SRC/.venv/.setup_version"
ENV_WANT="v4"
VENV_PY="$SRC/.venv/bin/python"
if [ ! -x "$VENV_PY" ] || [ "$(cat "$ENV_STAMP" 2>/dev/null)" != "$ENV_WANT" ]; then
  echo "### resmi Python ortami kuruluyor ($ENV_WANT)"
  command -v uv >/dev/null 2>&1 || pip install -q uv
  command -v uv >/dev/null 2>&1 || { echo "HATA: uv kurulamadi"; exit 1; }
  # YORUMLAYICI SISTEMDEN GELMELI. `uv venv -p 3.10` sistemde 3.10 bulamazsa
  # kendi python-build-standalone yapisini indirir; o yapi paylasimli
  # libpython3.10.so.1.0 sunmaz. QNN'in libPyIrGraph.so'su ise sistem
  # python3.10'una karsi derlenmis, DT_NEEDED listesinde libpython3.10.so.1.0
  # var ve import sirasinda onu arar. Sonuc, 4. adimda:
  #     ImportError: cannot import name 'libPyIrGraph' ...
  #     ImportError: libpython3.10.so.1.0: cannot open shared object file
  # (Dosya uv'nin dizininde bulunsa bile is gormez: dlopen edilen .so, DT_NEEDED
  # cozerken yorumlayicinin RUNPATH'ini miras almaz.) Ubuntu 22.04'te python3.10
  # dagitim varsayilanidir ve libpython3.10'a dinamik baglidir.
  if [ ! -x /usr/bin/python3.10 ] \
     || ! ldconfig -p 2>/dev/null | grep -q 'libpython3\.10\.so\.1\.0'; then
    echo "  [python] sistem python3.10 + libpython3.10 kuruluyor"
    DEBIAN_FRONTEND=noninteractive apt-get -qq update -y >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
        python3.10 python3.10-venv libpython3.10 >/dev/null 2>&1 || true
    ldconfig 2>/dev/null || true
  fi
  if [ -x /usr/bin/python3.10 ]; then
    uv venv -p /usr/bin/python3.10 --python-preference only-system --clear
  else
    echo "  [!] sistem python3.10 kurulamadi — uv kendi 3.10'unu kullanacak."
    echo "      qnn-onnx-converter libpython3.10.so.1.0 bulamayabilir;"
    echo "      asagidaki on kontrol bunu prepare_data'dan ONCE soyleyecek."
    uv venv -p 3.10 --clear
  fi
  uv sync                       # KILITLI surumler — QNN 2.28 ile uyumlu
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

# Yorumlayici paylasimli libpython'u nerede tutuyorsa yukleyici yoluna ekle.
# Sistem python'unda gereksiz (ldconfig zaten biliyor); uv'nin kendi python'una
# dusuldugu durumda ise tek sanstir.
PY_LIBDIR="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")' 2>/dev/null || true)"
if [ -n "$PY_LIBDIR" ] && [ -e "$PY_LIBDIR/libpython3.10.so.1.0" ]; then
  export LD_LIBRARY_PATH="$PY_LIBDIR:${LD_LIBRARY_PATH:-}"
fi

# ---- CUDA torch -----------------------------------------------------------
# Resmi pyproject torch'un CPU surumunu SABITLIYOR (torch==2.5.1+cpu, index
# .../whl/cpu). GPU'lu bir calisma zamaninda bunu asmak gerekiyor.
#
# TUZAK (uzun sure fark edilmedi): kurulu surum 2.5.1+cpu iken
#     uv pip install "torch==2.5.1" --index-url .../whl/cu121
# HICBIR SEY KURMAZ. PEP 440'a gore yerel etiketsiz bir '==2.5.1' istegi
# '2.5.1+cpu' tarafindan KARSILANIR; uv/pip "already satisfied" deyip gecer.
# Log'da "CUDA torch ekleniyor" yaziyor ama torch CPU kaliyor ve prepare_data
# ~3 dk yerine ~35 dk suruyor (cozunurluk basina!). Cozum iki parcali:
#   1) --reinstall-package torch  -> istek karsilansa da yeniden kur
#   2) sonucu OLC (torch.cuda.is_available), yaziya degil olcume guven
torch_durumu() {   # "<surum> <0|1>"
  "$VENV_PY" -c 'import torch;print(torch.__version__, int(torch.cuda.is_available()))' \
      2>/dev/null || echo "yok 0"
}
GPU_VAR=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU_VAR=1
  nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/  [gpu] /'
else
  echo "  [gpu] GPU yok"
fi
if [ "$CUDA_TORCH" = "1" ] && [ "$(torch_durumu | awk '{print $2}')" != "1" ]; then
  echo "  [cuda] CUDA torch kuruluyor ($CU) — kilitli 2.5.1+cpu'nun YERINE"
  echo "         (~2.5 GB iner; prepare_data ~35 dk yerine ~3 dk surer)"
  uv pip install --python "$VENV_PY" --reinstall-package torch \
      --index-url "https://download.pytorch.org/whl/$CU" "torch==2.5.1" \
    || echo "  [!] CUDA torch kurulamadi — CPU torch ile devam"
fi
TORCH_DURUM="$(torch_durumu)"
TORCH_VER="${TORCH_DURUM%% *}"; TORCH_CUDA="${TORCH_DURUM##* }"
if [ "$TORCH_CUDA" = "1" ]; then
  echo "  [cuda] torch $TORCH_VER — CUDA ETKIN"
elif [ "$GPU_VAR" = "1" ]; then
  echo "  [!] GPU VAR ama torch $TORCH_VER CUDA goremiyor."
  echo "      prepare_data CPU'da kosacak: cozunurluk basina ~35 dk."
  echo "      Farkli bir CUDA tekerlegi denemek icin: CUDA_WHL=cu124 (ya da cu118)"
else
  echo "  [cuda] torch $TORCH_VER — CPU (prepare_data yavas olacak)"
fi

"$VENV_PY" - <<'PYV' || true
import importlib
for m in ("onnx", "protobuf", "numpy", "torch"):
    try:
        mod = importlib.import_module("google.protobuf" if m == "protobuf" else m)
        print(f"  [surum] {m:9s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [surum] {m:9s} YOK ({type(e).__name__})")
PYV

REAL_FLAG=""
[ "$REALISTIC" = "1" ] && REAL_FLAG="--realistic"

# ---- Sistem bagimliliklari ------------------------------------------------
# Colab imajinda uc sey eksik:
#   1) libc++  — QNN 2.28'in Python baglantilari (libPyIrGraph) LLVM libc++'a
#      bagli: "ImportError: libc++.so.1: cannot open shared object file"
#   2) clang++ — qnn-model-lib-generator uretilen model.cpp'yi clang ile
#      derliyor: "Could not find compiler: clang++"
#   3) zstd    — ek cozunurluk yamalari 'zstd --patch-from' ile uretiliyor
# Dongunun ONUNDE kuruluyor: saatler suren kuantizasyondan sonra eksik bir
# arac yuzunden durmak en pahali hata.
_need_libcxx=0; _need_clang=0; _need_zstd=0
ldconfig -p 2>/dev/null | grep -q 'libc++\.so\.1' || _need_libcxx=1
command -v clang++ >/dev/null 2>&1 || _need_clang=1
if [ -n "$EXTRA_RES" ]; then
  command -v zstd >/dev/null 2>&1 || _need_zstd=1
fi
if [ "$_need_libcxx" = "1" ] || [ "$_need_clang" = "1" ] || [ "$_need_zstd" = "1" ]; then
  echo "### sistem bagimliliklari kuruluyor" \
       "$([ "$_need_libcxx" = 1 ] && echo libc++)" \
       "$([ "$_need_clang" = 1 ] && echo clang)" \
       "$([ "$_need_zstd" = 1 ] && echo zstd)"
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
  if [ "$_need_zstd" = "1" ]; then
    apt-get -qq install -y zstd >/dev/null 2>&1 || true
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
if [ -n "$EXTRA_RES" ]; then
  if command -v zstd >/dev/null 2>&1; then
    echo "  [deps] $(zstd --version 2>/dev/null | head -1)"
  else
    echo "HATA: zstd YOK — ek cozunurluk yamasi uretilemez."
    echo "      Elle: apt-get install -y zstd"
    exit 1
  fi
fi

# ---- Resmi convert scriptlerini bizim SDK'ya bagla ------------------------
# Resmi convert_all.sh SDK yolunu SABIT kodluyor (/data/qairt/2.28.0.241029).
# Ayrica iki `cd` satiri tirnaksiz yazilmis; bosluklu yolda scripti kiriyor.
# Yukaridaki symlink bunu zaten onluyor, bu sed ikinci emniyet kemeri.
for f in scripts/convert_all.sh scripts/convert_all_unet_only.sh; do
  [ -f "$f" ] || continue
  sed -i -e "s|^QNN_SDK_ROOT=.*|QNN_SDK_ROOT=\"$ABS_SDK\"|" \
         -e 's|^cd \$QNN_SDK_ROOT/bin$|cd "$QNN_SDK_ROOT/bin"|' \
         -e 's|^cd \${current_pwd}$|cd "${current_pwd}"|' "$f"
done
echo "  [sdk] convert_all.sh -> QNN_SDK_ROOT=$ABS_SDK"

# ---- On kontrol: qnn-onnx-converter gercekten kosuyor mu? -----------------
# 4. adim, cozunurluk basina ~40 dk suren prepare_data + ONNX disa aktarmanin
# ARDINDAN geliyor. Ortam bozuksa bunu orada ogrenmek bir oturumu yakiyor;
# burada bos bir --help ile saniyeler icinde ogreniliyor.
QNN_BIN="$ABS_SDK/bin/x86_64-linux-clang"
if [ -x "$QNN_BIN/qnn-onnx-converter" ]; then
  QNN_ERR="$WORK/.qnn_onnx_converter.err"
  if PYTHONPATH="$ABS_SDK/lib/python:${PYTHONPATH:-}" \
     LD_LIBRARY_PATH="$ABS_SDK/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}" \
     "$QNN_BIN/qnn-onnx-converter" --help >/dev/null 2>"$QNN_ERR"; then
    echo "  [on kontrol] qnn-onnx-converter calisiyor"
  elif grep -qE 'ImportError|ModuleNotFoundError|Traceback' "$QNN_ERR" 2>/dev/null; then
    # Yalnizca ortam bozuklugunda dur. --help'in kendi cikis kodu SDK surumune
    # gore degisebiliyor; hatti bos yere oldurmemek icin olcut cikis kodu degil,
    # stderr'de bir import hatasi olmasi.
    echo "HATA: qnn-onnx-converter calismiyor — 4. adim kesinlikle duser."
    tail -5 "$QNN_ERR" | sed 's/^/      /'
    if grep -q 'libpython3\.10\.so' "$QNN_ERR" 2>/dev/null; then
      echo "      Neden: yorumlayici paylasimli libpython3.10.so.1.0 sunmuyor"
      echo "             (uv'nin indirdigi python-build-standalone yapisi boyle)."
      echo "      Cozum: apt-get install -y python3.10 libpython3.10"
      echo "             rm -rf \"$SRC/.venv\"   (sonra bu scripti yeniden calistirin)"
    fi
    exit 1
  else
    echo "  [!] qnn-onnx-converter --help sifir disi dondu ama import hatasi yok;"
    echo "      devam ediliyor. (ayrinti: $QNN_ERR)"
  fi
else
  echo "  [!] $QNN_BIN/qnn-onnx-converter yok — on kontrol atlandi"
fi

# ---- Onbellek yardimcilari ------------------------------------------------
# prepare_data.py ~35 dk suruyor ve mobilde sekme arka plana atilinca calisma
# zamani kapaniyor -> her sey bastan. CACHE_REPO verilirse bu asama HER
# COZUNURLUK ICIN TEK SEFER odenir; sonraki oturumlar indirip atlar.
#
# Anahtar duzeni:  <slug>/res_<WxH>   -> kalibrasyon verisi (cozunurluge ozel)
#                  <slug>/out_<soc>   -> birikmis cikti (taban binary + yamalar)
# 512x512 icin cok cozunurluk oncesi kosulardan kalan DUZ <slug> anahtari da
# yedek olarak denenir, boylece eski onbellekler bosa gitmez.
CACHE_REPO="${CACHE_REPO:-}"
CACHE_OUTPUT="${CACHE_OUTPUT:-1}"
CACHE_FILES="data.pkl images input_list_unet.full.txt \
input_list_vae_decoder.full.txt input_list_vae_encoder.full.txt"
CACHE_ON=0
if [ -n "$CACHE_REPO" ] && [ -n "${HF_TOKEN:-}" ]; then
  CACHE_ON=1
  # SISTEM python'u: kilitli venv'e huggingface_hub eklemiyoruz.
  python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
elif [ -n "$CACHE_REPO" ]; then
  echo "  [!] CACHE_REPO verildi ama HF_TOKEN yok — onbellek kapali"
fi

cache_pull() {  # <anahtar> <dosyalar...>
  [ "$CACHE_ON" = "1" ] || return 0
  local key="$1"; shift
  python3 "$SDIR_SELF/stage_cache.py" pull --repo "$CACHE_REPO" \
      --key "$key" --dir "$SRC" --files "$@" || true
}
cache_push() {  # <anahtar> <dosyalar...>
  [ "$CACHE_ON" = "1" ] || return 0
  local key="$1"; shift
  python3 "$SDIR_SELF/stage_cache.py" push --repo "$CACHE_REPO" \
      --key "$key" --dir "$SRC" --files "$@" \
    || echo "  [!] onbellege yazilamadi — devam ediliyor"
}

# ---- Cozunurluk yardimcilari ----------------------------------------------
OUT_BASE="output_512"        # resmi export.sh ile ayni ad
OUT_DIR="$OUT_BASE/qnn_models_$SOC"

# Ruya/Local Dream'in yama tarayicisi kare boyutlari TEK sayiyla, dikdortgen
# boyutlari WxH ile adlandiriyor.
patch_name() {  # <W> <H>
  if [ "$1" = "$2" ]; then printf '%s.patch' "$1"; else printf '%sx%s.patch' "$1" "$2"; fi
}

# Bir cozunurluge gecmeden once o cozunurluge ozel her sey silinir. Resmi
# scriptler "varsa atla" mantigiyla calisiyor (data.pkl, unet/model.onnx,
# qnn_unet/.../libmodel.so); temizlenmezse ikinci cozunurluk sessizce
# BIRINCININ ciktisini yeniden paketler.
reset_stage() {
  rm -rf data.pkl images output unet qnn_unet \
         unet_input_raw vae_decoder_input_raw vae_encoder_input_raw
  rm -f input_list_unet.txt input_list_vae_decoder.txt input_list_vae_encoder.txt \
        input_list_unet.full.txt input_list_vae_decoder.full.txt \
        input_list_vae_encoder.full.txt
}

# prepare_data + gen_quant_data + kalibrasyon listesi kirpma.
stage_data() {  # <W> <H>
  local w="$1" h="$2" res="$1x$2" key="$SLUG/res_$1x$2"

  echo "### [$res] 0) onbellek kontrolu"
  cache_pull "$key" $CACHE_FILES
  if [ "$res" = "$BASE_RES" ] && [ ! -f "data.pkl" ]; then
    # cok cozunurluk oncesi duz anahtar
    cache_pull "$SLUG" $CACHE_FILES
  fi

  if [ ! -f "data.pkl" ]; then
    echo "### [$res] 1) prepare_data.py (20 prompt x difuzyon — EN UZUN ADIM)"
    "$VENV_PY" -c "import torch;print('    [torch]', torch.__version__,
          'cuda:', torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
    "$VENV_PY" prepare_data.py --model_path "$ABS_CKPT" --clip_skip "$CLIP_SKIP" \
        --width "$w" --height "$h" $REAL_FLAG
  else
    echo "### [$res] 1) prepare_data.py [ATLANDI - data.pkl var]"
  fi

  if [ ! -f "input_list_unet.full.txt" ]; then
    echo "### [$res] 2) gen_quant_data.py"
    "$VENV_PY" gen_quant_data.py
    # Kirpilmamis listeleri sakla: CALIB_LIMIT'i sonra BUYUTEBILMEK icin.
    # (Onceki surum listeyi yerinde kirpiyordu; geri cikmak icin gen_quant_data
    # tekrar kosmak gerekiyordu.)
    for f in unet vae_decoder vae_encoder; do
      [ -f "input_list_$f.txt" ] && cp "input_list_$f.txt" "input_list_$f.full.txt"
    done
  else
    echo "### [$res] 2) gen_quant_data.py [ATLANDI]"
  fi

  # En pahali asama bitti -> onbellege yaz.
  if [ -f "data.pkl" ]; then
    echo "### [$res] 2b) onbellege yaziliyor — sonraki oturum atlar"
    cache_push "$key" $CACHE_FILES
  fi

  # Kalibrasyon listesi: HER ZAMAN kirpilmamis .full kopyadan uretilir, boylece
  # CALIB_LIMIT hem asagi hem YUKARI degistirilebilir. Resmi hat 400 ornek
  # kullaniyor ve kuantizasyon buna dogru orantili (saatler).
  for f in unet vae_decoder vae_encoder; do
    local full="input_list_$f.full.txt"
    [ -f "$full" ] || continue
    local n; n=$(wc -l < "$full")
    if [ "$CALIB_LIMIT" -gt 0 ] 2>/dev/null && [ "$n" -gt "$CALIB_LIMIT" ]; then
      head -n "$CALIB_LIMIT" "$full" > "input_list_$f.txt"
      echo "  [kalib] input_list_$f.txt: $n -> $CALIB_LIMIT satir"
    else
      cp "$full" "input_list_$f.txt"
      echo "  [kalib] input_list_$f.txt: $n satir (tam)"
    fi
  done
}

# export_onnx*.py'ye verilecek model yolu. Ilk kosuda safetensors'tan
# diffusers dizini ('./model') dokuluyor (~4 GB, dakikalar); sonraki
# cozunurluklerde ayni dokumu tekrar yapmanin anlami yok.
model_path_arg() {
  if [ -f "$SRC/model/model_index.json" ] && [ -f "$SRC/model/unet/config.json" ]; then
    printf './model'
  else
    printf '%s' "$ABS_CKPT"
  fi
}

echo
echo "=========================================================="
echo " TABAN: $BASE_RES"
echo "=========================================================="

# Onceki surum ciktiyi 'output/' altinda birakiyordu; devam eden kosular
# bastan baslamasin diye tasiyoruz.
if [ ! -d "$OUT_BASE" ] && [ -f "output/qnn_models_$SOC/unet.bin" ]; then
  echo "  [gec] output/ -> $OUT_BASE (onceki surumden devam)"
  mv output "$OUT_BASE"
fi
# Birikmis ciktiyi (taban binary + o ana kadarki yamalar) onbellekten al.
if [ "$CACHE_ON" = "1" ] && [ "$CACHE_OUTPUT" = "1" ] && [ ! -d "$OUT_BASE" ]; then
  cache_pull "$SLUG/out_$SOC" "$OUT_BASE"
fi

if [ -f "$OUT_DIR/unet.bin" ]; then
  echo "### [$BASE_RES] [ATLANDI - $OUT_DIR/unet.bin var]"
else
  stage_data 512 512

  if [ ! -f "unet/model.onnx" ]; then
    echo "### [$BASE_RES] 3) export_onnx.py (redefined_modules: MHA->SHA, Linear->Conv)"
    "$VENV_PY" export_onnx.py --model_path "$(model_path_arg)" --clip_skip "$CLIP_SKIP"
  else
    echo "### [$BASE_RES] 3) export_onnx.py [ATLANDI - unet/model.onnx var]"
  fi

  echo "### [$BASE_RES] 4) QNN donusumu (clip + vae + unet)"
  bash scripts/convert_all.sh --min_soc "$SOC"

  mkdir -p "$OUT_BASE"
  cp -a output/. "$OUT_BASE"/
  rm -rf output
  [ -f "$OUT_DIR/unet.bin" ] || { echo "HATA: taban unet.bin uretilemedi"; exit 1; }
  if [ "$CACHE_ON" = "1" ] && [ "$CACHE_OUTPUT" = "1" ]; then
    echo "### [$BASE_RES] 4b) cikti onbellege yaziliyor"
    cache_push "$SLUG/out_$SOC" "$OUT_BASE"
  fi
fi

# ---- Ek cozunurlukler -----------------------------------------------------
# Her biri: kalibrasyon + ONNX + kuantizasyon (yalnizca UNet), sonra taban
# unet.bin'e karsi zstd farki. VAE/CLIP tekrar donusturulmez.
for res in $EXTRA_RES; do
  w="${res%%x*}"; h="${res##*x}"
  pname="$(patch_name "$w" "$h")"
  if [ -f "$OUT_DIR/$pname" ]; then
    echo
    echo "### [$res] [ATLANDI - $pname var]"
    continue
  fi
  echo
  echo "=========================================================="
  echo " EK COZUNURLUK: $res  ->  $pname"
  echo "=========================================================="
  reset_stage
  stage_data "$w" "$h"

  echo "### [$res] 3) export_onnx_unet_only.py"
  "$VENV_PY" export_onnx_unet_only.py --model_path "$(model_path_arg)" \
      --clip_skip "$CLIP_SKIP" --width "$w" --height "$h"

  echo "### [$res] 4) QNN donusumu (yalnizca UNet)"
  bash scripts/convert_all_unet_only.sh --min_soc "$SOC"

  NEW_UNET="output/qnn_models_$SOC/unet.bin"
  [ -f "$NEW_UNET" ] || { echo "HATA: $res icin unet.bin uretilemedi"; exit 1; }

  echo "### [$res] 5) zstd yamasi: $pname"
  # Resmi export.sh ile ayni cagri (varsayilan sikistirma). --patch-from
  # pencere boyutunu sozluge gore kendisi buyutuyor; ekstra bayrak vermiyoruz
  # cunku cihazdaki cozucu resmi paketlerdekiyle ayni varsayimla calisiyor.
  zstd -f --patch-from "$OUT_DIR/unet.bin" "$NEW_UNET" -o "$OUT_DIR/$pname"
  ls -la "$OUT_DIR/$pname" | sed 's/^/    /'

  # Disk: her tur ~7 GB ara dosya birakiyor, Colab'da yer dar.
  rm -rf output unet qnn_unet

  if [ "$CACHE_ON" = "1" ] && [ "$CACHE_OUTPUT" = "1" ]; then
    echo "### [$res] 6) cikti onbellege yaziliyor"
    cache_push "$SLUG/out_$SOC" "$OUT_BASE"
  fi
done

echo
echo "### cikti"
ls -la "$OUT_DIR" | sed 's/^/    /'

# ---- Paket ----------------------------------------------------------------
# Referans paketlerde dosyalar ZIP KOKUNDE (klasor yok) — resmi export.sh
# 'zip -r ... output_512/qnn_models_min' yaptigi icin orada klasorlu; biz
# uygulamanin bekledigi duz yapiyi uretiyoruz. Yamalar da kokte durur:
# uygulama unet.bin'in yanindaki *.patch dosyalarini tarayarak hangi
# cozunurluklerin secilebilecegini buluyor.
mkdir -p "$DIST"
ZIP="$DIST/${NAME}_qnn2.28_${SOC}.zip"
rm -f "$ZIP"
( cd "$OUT_DIR" && zip -q -r "$ZIP" . )
echo
echo "########################################################"
echo " BITTI. Cikti: $ZIP"
echo " Cozunurlukler: $BASE_RES${EXTRA_RES:+ $EXTRA_RES}"
unzip -l "$ZIP" | sed 's/^/    /'
echo "########################################################"
