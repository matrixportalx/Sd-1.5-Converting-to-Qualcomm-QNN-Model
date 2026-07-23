"""
Snapdragon SoC -> Hexagon HTP (DSP) mimari eslestirme tablosu.

Local Dream (Ruya) uygulamasi UNet'i Qualcomm QNN "context binary" olarak
NPU'da calistirir. Bir QNN context binary'si HER ZAMAN belirli bir HTP (DSP)
mimari surumune gore derlenir. Dusuk bir mimariye (or. v68/v69) derlenen bir
binary, o surumden YUKARI olan tum cihazlarda geriye-donuk uyumlu calisir;
bu yuzden "_min" varyanti en dusuk mimariyi hedefler ve en genis cihaz
yelpazesini (Snapdragon 7 serisi dahil) kapsar.

ONEMLI: Asagidaki soc_id degerleri Qualcomm QNN SDK'sinin dahili SoC kimlik
numaralaridir ve SDK surumune gore degisebilir. Kesin degerler icin kullandiginiz
QNN SDK icindeki:
    $QNN_SDK_ROOT/docs/QNN/general/htp/htp_backend_soc.html
belgesine veya "qnn-platform-validator" ciktisina bakin. Buradaki tablo QNN
SDK 2.28 icin genel kabul goren degerlerdir; uyusmazlik olursa SDK belgesi
esastir.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SocTarget:
    key: str          # dahili anahtar (CLI'de kullanilir)
    name: str         # okunabilir isim
    soc_model: str    # QNN --soc_model / dsp config icin SoC ismi (SMxxxx)
    soc_id: int       # QNN HTP config icindeki sayisal soc_id
    dsp_arch: str     # Hexagon HTP mimari surumu (v68, v69, v73, v75, v79)


# --- Bilinen Snapdragon SoC'lari (SD1.5 = Hexagon V68 ve uzeri gerekir) --------
KNOWN_SOCS = [
    # Snapdragon 7 serisi (kullanicinin hedefi)
    SocTarget("sm7450", "Snapdragon 7 Gen 1",      "SM7450", 57, "v69"),
    SocTarget("sm7435", "Snapdragon 7s Gen 2",     "SM7435", 57, "v69"),
    SocTarget("sm7475", "Snapdragon 7+ Gen 2",     "SM7475", 43, "v73"),
    SocTarget("sm7550", "Snapdragon 7 Gen 3",      "SM7550", 43, "v73"),
    SocTarget("sm7675", "Snapdragon 7+ Gen 3",     "SM7675", 43, "v73"),
    # Snapdragon 8 serisi (referans / karsilastirma)
    SocTarget("sm8450", "Snapdragon 8 Gen 1",      "SM8450", 36, "v69"),
    SocTarget("sm8550", "Snapdragon 8 Gen 2",      "SM8550", 43, "v73"),
    SocTarget("sm8635", "Snapdragon 8s Gen 3",     "SM8635", 43, "v73"),
    SocTarget("sm8650", "Snapdragon 8 Gen 3",      "SM8650", 57, "v75"),
    SocTarget("sm8750", "Snapdragon 8 Elite",      "SM8750", 69, "v79"),
]


# --- Cihaz siniflari (Local Dream'in "chip level" mantigini yansitir) ----------
# Bir context binary'si hedeflenen dsp_arch'tan YUKARI cihazlarda da calisir.
# Bu yuzden her "tier" tek bir taban mimariyi hedefler:
#
#   min  -> v68 tabanli. SD1.5 destekleyen (V68+) TUM cihazlar. -> Snapdragon 7
#   mid  -> v73 tabanli. 8 Gen 2 / 7+ Gen 2 / 8s Gen 3 ve uzeri.
#   high -> v75 tabanli. 8 Gen 3 ve uzeri (en iyi performans).
#
# Ruya uygulamaniz "..._qnn2.28_min.zip" istedigi icin sizin tier'iniz "min".
# ZIP eki: _qnn<surum><tail>  (surum --qnn-version ile, varsayilan 2.39)
TIERS = {
    "min":  {"dsp_arch": "v68", "soc_model": "SM8450", "soc_id": 36,
             "tail": "_min",
             "desc": "En genis uyumluluk (V68+). Snapdragon 7 Gen 1, 7s Gen 2, "
                     "8 Gen 1 ve tum ust cihazlar. ZIP eki: _qnn<surum>_min"},
    "mid":  {"dsp_arch": "v73", "soc_model": "SM8550", "soc_id": 43,
             "tail": "",
             "desc": "Snapdragon 8 Gen 2 / 7+ Gen 2 / 8s Gen 3 ve uzeri. "
                     "ZIP eki: _qnn<surum>"},
    "high": {"dsp_arch": "v75", "soc_model": "SM8650", "soc_id": 57,
             "tail": "_8gen3",
             "desc": "Snapdragon 8 Gen 3 ve uzeri (en yuksek performans). "
                     "ZIP eki: _qnn<surum>_8gen3"},
}


def get_tier(tier: str) -> dict:
    if tier not in TIERS:
        raise SystemExit(
            f"Bilinmeyen tier '{tier}'. Secenekler: {', '.join(TIERS)}")
    return TIERS[tier]


def print_table() -> None:
    print("Snapdragon SoC -> HTP mimari tablosu")
    print("-" * 64)
    print(f"{'Anahtar':<10}{'SoC':<24}{'soc_model':<10}{'dsp_arch'}")
    print("-" * 64)
    for s in KNOWN_SOCS:
        print(f"{s.key:<10}{s.name:<24}{s.soc_model:<10}{s.dsp_arch}")
    print("-" * 64)
    print("\nTier (Local Dream chip level):")
    for name, t in TIERS.items():
        print(f"  {name:<5} dsp_arch={t['dsp_arch']:<5} -> {t['desc']}")


if __name__ == "__main__":
    print_table()
