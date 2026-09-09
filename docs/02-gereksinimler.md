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

Dönüşümün Python ortamını `scripts/06_official_pipeline.sh` **kendisi kurar**:
sistem python3.10 üzerinde bir `uv` venv'i açar ve resmi tarifin kilitli
sürümlerini (torch, diffusers, onnx ...) oraya yükler. Sizin kurmanız gereken
tek şey `uv`:

```bash
pip install uv
```

Yardımcı scriptler (model indirme, HF yükleme, önbellek) sistem python'unda
koşar ve iki paket ister:

```bash
pip install -r requirements.txt
```

> Sistemde **python3.10** yoksa hat onu apt'tan kurar (`python3.10`,
> `python3.10-venv`, `libpython3.10`). Bu şart: QNN'in converter eklentisi
> paylaşımlı `libpython3.10.so.1.0` arar; uv'nin indirdiği python bunu sunmaz.
> Ayrıntı: [`05-sorun-giderme.md`](05-sorun-giderme.md).

## 4. QNN SDK — **sürüm 2.28 şart**

Hat, Local Dream'in resmi `npuconvertv2` scriptlerini koşar ve o tarif **QNN
2.28** araç zinciriyle (`qnn-onnx-converter` → `qnn-model-lib-generator` →
`qnn-context-binary-generator`) test edilmiştir. 2.39 ile koşmak saatleri boşa
harcar ve çıkan paket cihazda yüklenmez; pipeline sürümü başta doğrular.

```bash
export QNN_SDK_ROOT=$(python scripts/setup_qnn_sdk.py --dest ./qairt | sed -n 's/^QNN_SDK_ROOT=//p' | tail -1)
ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang/"   # qnn-onnx-converter burada olmalı
```

İndirme linki `config.env` içindeki `OVERRIDE_QAIRT_ASSET_URL` satırından
gelir; Qualcomm URL'yi değiştirirse tek satırlık düzeltme yeter. Alternatif
olarak bir GitHub release'inden de indirilebilir
(`--repo <owner/repo> --tag <tag>`; private ise `export GH_TOKEN=ghp_...`).

Çıktı ZIP'i `_qnn2.28_<soc>` etiketlenir.

## 5. MNN + MNNConvert

Resmi scriptler kendi `MNNConvert` ikilisini paketle birlikte getirir — ayrıca
kurmanız **gerekmez**. Elle bir dönüşüm yapacaksanız:

```bash
git clone https://github.com/alibaba/MNN
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j"$(nproc)"
export MNNCONVERT="$PWD/MNNConvert"
```

## 6. Kontrol listesi

```bash
echo "$QNN_SDK_ROOT" && ls "$QNN_SDK_ROOT/bin/x86_64-linux-clang" | head
python3 -c "import requests, huggingface_hub; print('yardimci paketler ok')"
command -v uv && uv --version
python scripts/soc_targets.py
```
