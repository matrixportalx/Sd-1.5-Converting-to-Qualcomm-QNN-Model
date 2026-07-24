#!/usr/bin/env bash
#
# TESHIS — Calisan bir referans modelin context binary metadata'sini doker.
#
# Neden: Uygulama (local-dream QnnModel.hpp) UNet girdilerine
#   inputs[0] latents        -> uint16_t (16-bit kuantize)
#   inputs[1] timestep       -> int32_t
#   inputs[2] text_embedding -> uint16_t (16-bit kuantize)
# yaziyor. Bizim a8w8 binary'mizde bunlar uint8/float32 -> uyumsuzluk -> cokme.
#
# Bu script calisan bir .bin'in GERCEK tensor dtype'larini, graf adlarini ve
# hedef dsp_arch'ini gosterir; boylece birebir eslesecek sekilde uretiriz.
#
# Kullanim:
#   export QNN_SDK_ROOT=...
#   ./inspect_reference.sh /yol/unet.bin
#   ./inspect_reference.sh   # varsayilan: HF'den referans indirir
#
set -euo pipefail

BIN="${1:-}"
: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali}"
TOOL="$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility"
chmod +x "$QNN_SDK_ROOT/bin" -R 2>/dev/null || true
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

if [ ! -x "$TOOL" ]; then
  echo "HATA: qnn-context-binary-utility bulunamadi: $TOOL"
  echo "SDK bin icerigi:"; ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang" | head -30
  exit 1
fi

# Referans yoksa HF'den indir
if [ -z "$BIN" ]; then
  REF_REPO="${REF_REPO:-Mr-J-369/AbyssOrangeMix3-SD1.5-qnn2.28}"
  echo "==> Referans indiriliyor: $REF_REPO"
  python3 - "$REF_REPO" <<'PY'
import sys, os, zipfile
from huggingface_hub import HfApi, hf_hub_download
repo = sys.argv[1]; tok = os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
files = api.list_repo_files(repo)
print("repo dosyalari:", files)
zips = [f for f in files if f.endswith(".zip")]
target = next((z for z in zips if "min" in z.lower()), zips[0] if zips else None)
if target:
    p = hf_hub_download(repo, target, token=tok)
    os.makedirs("/tmp/ref", exist_ok=True)
    with zipfile.ZipFile(p) as zf: zf.extractall("/tmp/ref")
    print("acildi -> /tmp/ref")
else:
    # zip yoksa unet.bin dogrudan olabilir
    for f in files:
        if f.endswith("unet.bin"):
            p = hf_hub_download(repo, f, token=tok)
            os.makedirs("/tmp/ref", exist_ok=True)
            os.replace(p, "/tmp/ref/unet.bin"); print("indirildi -> /tmp/ref/unet.bin")
            break
PY
  BIN="$(find /tmp/ref -name 'unet.bin' | head -1)"
  [ -z "$BIN" ] && { echo "HATA: referans unet.bin bulunamadi"; exit 1; }
fi

echo ""
echo "############################################################"
echo "# REFERANS: $BIN  ($(du -h "$BIN" | cut -f1))"
echo "############################################################"
OUT=/tmp/ref_meta.json
"$TOOL" --context_binary "$BIN" --json_file "$OUT" >/dev/null 2>&1 || \
  "$TOOL" --context_binary "$BIN" || true

if [ -f "$OUT" ]; then
  python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
def walk(o, depth=0):
    pass
info = d.get("info", d)
graphs = info.get("graphs") or d.get("graphs") or []
print(f"GRAF SAYISI: {len(graphs)}")
for g in graphs:
    gi = g.get("info", g)
    print(f"\n--- GRAF: {gi.get('graphName')} ---")
    for key, label in (("graphInputs","GIRDI"), ("graphOutputs","CIKTI")):
        for t in gi.get(key, []):
            ti = t.get("info", t)
            print(f"  {label}: name={ti.get('name')} "
                  f"dtype={ti.get('dataType')} "
                  f"dims={ti.get('dimensions')} "
                  f"quant={ti.get('quantizeParams',{}).get('scaleOffsetEncoding')}")
meta = {k: v for k, v in (info.items() if isinstance(info, dict) else []) if k != "graphs"}
if meta: print("\nMETA:", json.dumps(meta, indent=2)[:1200])
PY
  echo ""
  echo ">>> Bu ciktiyi paylas: dtype'lar (UFIXED_POINT_8/16, INT_32) ve"
  echo ">>> graf adlari bizim uretimimizle eslesmeli."
else
  echo "[!] JSON uretilemedi; yukaridaki duz cikti kullanilacak."
fi
