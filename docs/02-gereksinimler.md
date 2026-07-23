# 02 — Gereksinimler ve Kurulum

## 1. İşletim sistemi

Qualcomm QNN ve MNN dönüştürücüleri **x86-64 Linux** araçlarıdır.

- **Linux (Ubuntu 20.04/22.04 önerilir):** doğrudan çalışır.
- **Windows:** **WSL2 + Ubuntu** kurun. Saf Windows (native) desteklenmez.
- **macOS / ARM:** desteklenmez (x86-64 gerekir).

## 2. Donanım

| Çözünürlük | Önerilen RAM |
|---|---|
| 512×512 | 20 GB+ |
| 512×768 / 768×512 | 32 GB+ |
| Daha yüksek | 64 GB+ ve bol swap |

Kuantizasyon CPU üzerinde çok yavaştır — bir çözünürlük × bir tier **saatler**
sürebilir. Bu normaldir. GPU zorunlu değildir (yalnızca ONNX export'u hızlandırır).

RAM yetmezse geçici swap ekleyin:

```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

## 3. Python ortamı

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. QAIRT / QNN SDK — **sürüm 2.39** (release'ten otomatik)

Ruya derlemesinde kullanılan SDK, bir GitHub release'i olarak yayında:
**`matrixportalx/qairt-sdk` → `v2.39.0.250926`**. Bu yüzden Qualcomm'dan manuel
indirmeye gerek yok — dahili yardımcı script otomatik indirir:

```bash
python scripts/setup_qnn_sdk.py --dest ./qairt
# çıktının son satırındaki yolu kullanın:
export QNN_SDK_ROOT=$(python scripts/setup_qnn_sdk.py --dest ./qairt | sed -n 's/^QNN_SDK_ROOT=//p' | tail -1)
ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang/"   # qairt-converter / qnn-onnx-converter
```

- Release **public** ise token gerekmez. Private ise: `export GH_TOKEN=ghp_...`
- Farklı sürüm/asset için: `--repo <owner/repo> --tag <tag>` veya `--asset-url <url>`.

> **Sürüm notu:** Çıktı ZIP'i varsayılan olarak `_qnn2.39_min` etiketlenir
> (`QNN_VERSION` env ile değiştirilebilir). QAIRT 2.39, yeni `qairt-converter` +
> `qairt-quantizer` araç zincirini kullanır; dönüşüm scripti bunu otomatik
> algılar, yoksa eski `qnn-onnx-converter`'a düşer.

SDK'nın kendi Python bağımlılıkları varsa:

```bash
source "$QNN_SDK_ROOT/bin/envsetup.sh" 2>/dev/null || true   # PATH/PYTHONPATH
"$QNN_SDK_ROOT/bin/check-python-dependency" 2>/dev/null || true
```

## 5. MNN + MNNConvert

text_encoder ve VAE'yi `.mnn`'e çevirmek için MNNConvert gerekir:

```bash
git clone https://github.com/alibaba/MNN
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j"$(nproc)"
export MNNCONVERT="$PWD/MNNConvert"
```

## 6. Kontrol listesi

```bash
python -c "import torch, diffusers, transformers, onnx; print('py ok')"
echo "$QNN_SDK_ROOT" && ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang" | head
"$MNNCONVERT" --version || echo "MNNCONVERT yolunu kontrol edin"
python scripts/soc_targets.py
```
