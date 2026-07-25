#!/usr/bin/env bash
#
# TESHIS (hizli, ~30 sn) — QAIRT araclarinin TAM arayuzunu ve SDK icindeki
# karma hassasiyet (mixed precision) / backend-aware kuantizasyon
# dokumanlarini doker.
#
# Neden: v68'de 16-bit MatMul reddediliyor ("expected >= 73"). Cozum ya
#   (a) --use_per_row_quantization  (MatMul/FC icin rowwise; --help'e gore
#       restrict_quantization_steps'in kabul ettigi sema),
#   (b) --target_backend / --target_soc_model (backend-aware kuantizasyon —
#       quantizer op bazinda hedefin destekledigi hassasiyeti secer), ya da
#   (c) --config <json> (op/tensor bazinda bitwidth = gercek karma hassasiyet)
# ile saglanir. Hangisinin bu SDK'da nasil kullanildigini TAHMIN ETMEDEN
# gormek icin bu dokumu kullaniyoruz.
#
# Kullanim:  export QNN_SDK_ROOT=...; bash scripts/dump_sdk_help.sh
#
set -uo pipefail

: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali}"
BIN="$QNN_SDK_ROOT/bin/x86_64-linux-clang"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}"
chmod -R +x "$QNN_SDK_ROOT/bin" 2>/dev/null || true

QNN_PY="${QNN_PYTHON:-}"
[ -z "$QNN_PY" ] && [ -f /content/qnn_py.path ] && QNN_PY="$(cat /content/qnn_py.path)"
[ -z "$QNN_PY" ] && QNN_PY="python3"

hdr() { echo; echo "################################################################"; \
        echo "# $*"; echo "################################################################"; }

hdr "1) qairt-quantizer --help (TAM)"
"$QNN_PY" "$BIN/qairt-quantizer" --help 2>&1 || true

hdr "2) qairt-converter --help (yalniz kuantizasyon/precision ile ilgili)"
"$QNN_PY" "$BIN/qairt-converter" --help 2>&1 \
  | grep -iE -- "--[a-z_]+|bitwidth|quant|precision|encoding|dtype|backend|soc" \
  | head -120 || true

hdr "3) qnn-context-binary-generator --help"
"$BIN/qnn-context-binary-generator" --help 2>&1 | head -80 || true

hdr "4) SDK icinde 'mixed precision' / quantizer config sema-ornekleri"
for pat in "mixed_precision" "mixedPrecision" "quantizer_config" "quantization_config"; do
  echo "--- $pat ---"
  grep -rl --include='*.json' --include='*.py' --include='*.md' --include='*.html' \
       -i "$pat" "$QNN_SDK_ROOT" 2>/dev/null | head -8
done

hdr "5) --config icin sema/ornek dosyalari"
find "$QNN_SDK_ROOT" \( -iname '*config*schema*' -o -iname '*quant*config*' \) \
     -size -2M 2>/dev/null | head -20

hdr "6) HTP backend: hangi op'lar 16-bit icin v73+ istiyor?"
grep -rn -i "expected >= 73\|16.*matmul\|matmul.*16" \
     "$QNN_SDK_ROOT/docs" 2>/dev/null | head -20

hdr "7) Desteklenen --target_soc_model degerleri (SDK'dan okunur)"
"$QNN_PY" "$(dirname "${BASH_SOURCE[0]}")/list_soc_models.py" 2>&1 | head -80 || true

# ---------------------------------------------------------------------------
# ASIL ARANAN: graf I/O'sunu 16-bit yapip ic hesabi 8-bit birakmanin resmi yolu.
# qairt-converter --dump_config_template <yaml> ile SEMA dokuluyor; duzenlenip
# --config ile geri veriliyor.
# ---------------------------------------------------------------------------
hdr "8) qairt-converter I/O config YAML SABLONU (--dump_config_template)"
TPL=/tmp/io_config_template.yaml
"$QNN_PY" "$BIN/qairt-converter" --dump_config_template "$TPL" 2>&1 | tail -5 || true
if [ -f "$TPL" ]; then cat "$TPL"; else echo "(sablon uretilemedi)"; fi

hdr "9) qairt-quantizer --config YAML sablonu (varsa)"
for f in $(find "$QNN_SDK_ROOT" \( -iname '*quantizer*config*' -o -iname '*config*template*' \) \
           2>/dev/null | grep -iE '\.(yaml|yml|json)$' | head -5); do
  echo "--- $f ---"; head -60 "$f"
done

hdr "10) HTP opdef surum gecmisi — 16-bit hangi surumde acildi?"
H="$QNN_SDK_ROOT/docs/QNN/general/htp/htp_opdef_version_history.html"
if [ -f "$H" ]; then
  python3 - "$H" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
txt = re.sub(r"<[^>]+>", " ", txt)
txt = re.sub(r"[ \t]+", " ", txt)
lines = [l.strip() for l in txt.splitlines() if l.strip()]
keys = ("layernorm", "layer_norm", "matmul", "int16", "fixed_point_16",
        "v68", "v69", "v73", "version")
for i, l in enumerate(lines):
    low = l.lower()
    if any(k in low for k in keys):
        print(l[:200])
PY
else
  echo "(bulunamadi: $H)"
fi

echo
echo ">>> BITTI. Bu ciktinin TAMAMINI paylas."
