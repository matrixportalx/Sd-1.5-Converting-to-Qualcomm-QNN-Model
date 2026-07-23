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

## 4. Qualcomm AI Engine Direct (QNN) SDK — **sürüm 2.28**

1. Qualcomm hesabıyla **Qualcomm AI Hub** veya **Qualcomm Package Manager (QPM)**
   üzerinden **Qualcomm AI Engine Direct SDK 2.28** (`v2.28.0.241029`) indirin.
2. Bir yere açın ve ortam değişkenini ayarlayın:

```bash
export QNN_SDK_ROOT=/opt/qairt/2.28.0.241029
# doğrulama:
ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter"
```

> **Neden tam olarak 2.28?** Paket adındaki `qnn2.28`, uygulamanın beklediği
> runtime sürümüdür. Başka bir SDK sürümüyle üretilen binary uyumsuz olabilir.

QNN converter'ın kendi Python bağımlılıkları vardır; SDK içindeki
`$QNN_SDK_ROOT/bin/check-python-dependency` veya
`$QNN_SDK_ROOT/bin/envsetup.sh` ile kurun:

```bash
source "$QNN_SDK_ROOT/bin/envsetup.sh"   # PATH/PYTHONPATH ayarlar
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
