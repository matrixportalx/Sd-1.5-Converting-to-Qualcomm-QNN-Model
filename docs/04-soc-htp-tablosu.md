# 04 — Snapdragon SoC ↔ HTP Mimarisi ve Tier Seçimi

## Neden mimari önemli?

QNN "context binary", **belirli bir Hexagon HTP (DSP) mimari sürümüne** göre
derlenir. Kural:

> Düşük mimariye derlenen binary, o sürümden **yukarı** olan cihazlarda çalışır.
> Yüksek mimariye derlenen binary, **aşağı** cihazlarda **çalışmaz**.

Bu yüzden geniş uyumluluk isteyen (Snapdragon 7) **en düşük** mimariyi hedefler.

## SoC → HTP tablosu

`python scripts/soc_targets.py` çıktısıyla aynıdır:

| Anahtar | SoC | soc_model | HTP (dsp_arch) |
|---|---|---|---|
| sm7450 | Snapdragon 7 Gen 1 | SM7450 | v69 |
| sm7435 | Snapdragon 7s Gen 2 | SM7435 | v69 |
| sm7475 | Snapdragon 7+ Gen 2 | SM7475 | v73 |
| sm7550 | Snapdragon 7 Gen 3 | SM7550 | v73 |
| sm7675 | Snapdragon 7+ Gen 3 | SM7675 | v73 |
| sm8450 | Snapdragon 8 Gen 1 | SM8450 | v69 |
| sm8550 | Snapdragon 8 Gen 2 | SM8550 | v73 |
| sm8635 | Snapdragon 8s Gen 3 | SM8635 | v73 |
| sm8650 | Snapdragon 8 Gen 3 | SM8650 | v75 |
| sm8750 | Snapdragon 8 Elite | SM8750 | v79 |

> **Uyarı:** `soc_id` (koddaki sayısal kimlik) QNN SDK sürümüne göre
> değişebilir. Kesin değer için `$QNN_SDK_ROOT` içindeki HTP SoC belgesine veya
> `qnn-platform-validator` çıktısına bakın. `dsp_arch` değeri asıl belirleyicidir.

## Tier'ler (Local Dream "chip level")

| Tier | dsp_arch | ZIP eki | Kapsam |
|---|---|---|---|
| **min** | v68 | `_qnn2.39_min` | SD1.5 destekleyen **tüm** cihazlar (V68+). **Snapdragon 7 buradadır.** |
| mid | v73 | `_qnn2.39` | 8 Gen 2 / 7+ Gen 2 / 8s Gen 3 ve üzeri |
| high | v75 | `_qnn2.39_8gen3` | 8 Gen 3 ve üzeri (en yüksek performans) |

> Sürüm etiketi (`2.39`) `QNN_VERSION` env / `--qnn-version` ile değiştirilebilir.

### Hangi tier'i seçmeliyim?

- **Snapdragon 7 (herhangi bir sürüm): `min`.** — Sizin durumunuz.
- Sadece kendi 8 Gen 2/3 cihazınız için en iyi performans: `mid` / `high`.
- Emin değilseniz: `min` her yerde çalışır, güvenli seçimdir.

Tier seçimi `convert_all.sh`'in 3. argümanı veya `03_convert_unet_qnn.sh`'in
3. argümanıdır:

```bash
./convert_all.sh model.safetensors MyModel min   # <- min
```

## SD1.5 alt sınırı

Local Dream SD1.5 için **Hexagon V68+** gerektirir. V68 altı (çok eski) cihazlar
NPU modunu çalıştıramaz; onlarda uygulamanın CPU modu kullanılır.
