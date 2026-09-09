# 04 — Snapdragon SoC ↔ HTP Mimarisi ve Tier Seçimi

## Neden mimari önemli?

QNN "context binary", **belirli bir Hexagon HTP (DSP) mimari sürümüne** göre
derlenir. Kural:

> Düşük mimariye derlenen binary, o sürümden **yukarı** olan cihazlarda çalışır.
> Yüksek mimariye derlenen binary, **aşağı** cihazlarda **çalışmaz**.

Bu yüzden geniş uyumluluk isteyen (Snapdragon 7) **en düşük** mimariyi hedefler.
Ama uyumluluğun bir bedeli var — aşağıdaki [Hangi SOC'u seçmeliyim?](#hangi-socu-seçmeliyim)
bölümüne bakın.

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

## SOC seçenekleri (Local Dream "chip level")

Resmi hattın (`npuconvertv2`) sunduğu üç seçenek — her biri kendi
`htp_config_<soc>.json` dosyasıyla derlenir:

| SOC | dsp_arch | ZIP eki | Kapsam |
|---|---|---|---|
| **min** | v68 | `_qnn2.28_min` | SD1.5 destekleyen **tüm** cihazlar (V68+) |
| **8gen1** | v69 | `_qnn2.28_8gen1` | 8 Gen 1, 7 Gen 1, 7s Gen 2 |
| **8gen2** | v73 | `_qnn2.28_8gen2` | 8 Gen 2, 8s Gen 3, 7+ Gen 2, 7 Gen 3 |

> Yukarıdaki `min`/`8gen1`/`8gen2` isimleri **resmi hattın** değerleridir.
> Depodan silinen eski hat `min`/`mid`/`high` kullanıyordu; eski notlarda o
> isimlere rastlarsanız kullanmayın.

### Hangi SOC'u seçmeliyim?

**Kendi cihazınızın mimarisini seçin.** Düşük mimariye derlenmiş binary yukarı
cihazlarda *çalışır* ama native derlenmişten belirgin biçimde **yavaştır**.

Ölçüm (OnePlus 12R / Snapdragon 8 Gen 2 = v73, aynı model, aynı prompt ve seed,
20 adım · CFG 7 · 512×512):

| | `min` paketi (v68) | `8gen2` paketi (v73) |
|---|---|---|
| Görsel üretimi | 13,7 sn | **5,7 sn** (2,4× hızlı) |
| Model yükleme + graf hazırlığı | ~27,6 sn | **~4,2 sn** (6,5× hızlı) |
| Görsel kalitesi | — | **aynı** |

- Kendi cihazınız için üretiyorsanız: cihazınıza karşılık gelen SOC.
- Paketi **paylaşacaksanız** ya da farklı nesil cihazlarda kullanacaksanız: `min`.
- 8 Gen 3 (v75) / 8 Elite (v79) için ayrı bir seçenek yok; `8gen2` (v73) geriye
  dönük uyumlu çalışır.

**Kalite SOC'tan bağımsızdır.** Kuantizasyon ayarları (`--act_bitwidth 16`,
`--use_per_channel_quantization`, `redefined_modules/`, `vtcm_mb: 2`) resmi
hatta sabittir; SOC yalnızca hedef mimariyi seçer.

SOC seçimi `06_official_pipeline.sh`'in **4. argümanıdır**:

```bash
bash scripts/06_official_pipeline.sh model.safetensors MyModel work/my 8gen2
```

## SD1.5 alt sınırı

Local Dream SD1.5 için **Hexagon V68+** gerektirir. V68 altı (çok eski) cihazlar
NPU modunu çalıştıramaz; onlarda uygulamanın CPU modu kullanılır.
