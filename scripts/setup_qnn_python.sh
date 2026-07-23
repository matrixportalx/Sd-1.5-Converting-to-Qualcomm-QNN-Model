#!/usr/bin/env bash
#
# QAIRT/QNN Python konvertorleri (qairt-converter, qairt-quantizer) icin izole
# bir Python 3.10 ortami kurar. QAIRT 2.39 Python 3.10 + libc++ ister; Colab
# ise Python 3.12 kullanir, bu yuzden ayri bir venv sart.
#
# Ciktisi: venv python yolunu /content/qnn_py.path (veya $2) dosyasina yazar.
# 03_convert_unet_qnn.sh bu yolu QNN_PYTHON olarak kullanir.
#
# Kullanim:
#   export QNN_SDK_ROOT=/content/qairt/qairt/2.39.0.250926
#   ./setup_qnn_python.sh
#
set -euo pipefail

: "${QNN_SDK_ROOT:?QNN_SDK_ROOT ayarli olmali}"
VENV="${1:-/content/qairt-venv}"
OUT_PATH="${2:-/content/qnn_py.path}"

echo "==> Sistem paketleri (python3.10 + libc++)"
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
# Ubuntu 22.04'te python3.10 dagitim varsayilanidir; libc++ QAIRT pybind icin gerekli
apt-get -qq install -y python3.10 python3.10-venv python3.10-dev \
                       libc++1 libc++abi1 >/dev/null
echo "    python3.10: $(python3.10 --version)"

echo "==> Python 3.10 venv: $VENV"
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "==> QAIRT python bagimliliklari"
# Once SDK kendi requirements dosyasini sunuyorsa onu dene
REQ="$(find "$QNN_SDK_ROOT" -name 'requirements.txt' -path '*python*' 2>/dev/null | head -1 || true)"
if [ -n "$REQ" ]; then
  echo "    SDK requirements: $REQ"
  "$VENV/bin/pip" install -q -r "$REQ" || echo "    (bazi paketler atlandi)"
fi
# KRITIK: onnx + UYUMLU protobuf sabitle. protobuf 6.x, PyPI onnx'in C uzantisini
# bozar ve qairt-converter 'onnx=None -> AttributeProto' hatasi verir.
# onnx 1.16 + protobuf 4.25 QAIRT 2.39 ile bilinen calisan kombinasyondur.
"$VENV/bin/pip" install -q "numpy==1.26.4" "protobuf==4.25.5" "onnx==1.16.1" \
                          onnxruntime pyyaml packaging || true
# onnx gercekten yuklenebiliyor mu? (AttributeProto erisimi)
"$VENV/bin/python" -c "import onnx; assert onnx.AttributeProto.INT is not None; \
  print('    onnx', onnx.__version__, 'OK')" \
  || echo "    [uyari] onnx hala import edilemiyor; ciktidaki hataya bakin"

# SDK'nin kendi bagimlilik denetimi (bilgi amacli)
if [ -x "$QNN_SDK_ROOT/bin/check-python-dependency" ]; then
  PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}" \
    "$VENV/bin/python" "$QNN_SDK_ROOT/bin/check-python-dependency" 2>/dev/null || true
fi

# Dogrulama: pybind yuklenebiliyor mu?
echo "==> Dogrulama (qti.aisw.dlc_utils)"
PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}" \
  "$VENV/bin/python" -c "from qti.aisw.dlc_utils import modeltools; print('    pybind OK')" \
  || echo "    [uyari] pybind hala yuklenemedi; ciktidaki hataya bakin"

echo "$VENV/bin/python" > "$OUT_PATH"
echo "[+] QNN Python hazir -> $(cat "$OUT_PATH")"
echo "    (03_convert_unet_qnn.sh bunu QNN_PYTHON olarak otomatik kullanir)"
