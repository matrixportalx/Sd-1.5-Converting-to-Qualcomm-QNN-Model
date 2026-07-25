#!/usr/bin/env python3
"""
Qualcomm Software Center'da HANGI QAIRT surumleri hala indirilebiliyor?

Software Center'in ARAYUZU yalnizca guncel surumleri listeliyor, ama dogrudan
API yolu genelde eski surumleri de sunmaya devam ediyor (Radxa/sherpa
dokumanlari bu URL'yi girissiz duz `wget` ile kullaniyor):

    https://softwarecenter.qualcomm.com/api/download/software/sdks/
        Qualcomm_AI_Runtime_Community/All/<SURUM>/v<SURUM>.zip

Neden onemli: referans `_min` modeller QNN 2.28 ile uretilmis. QAIRT 2.39
v68'de ayni grafi uretemiyor (16-bit LayerNorm yok, 16-bit etiketi MatMul
agirliklarina siziyor). 2.28 indirilebiliyorsa dogrudan onunla uretiriz.

Kullanim:
    python probe_qairt_versions.py            # varsayilan aday listesi
    python probe_qairt_versions.py 2.28.0.241029 2.29.0.241129
"""
import sys
import urllib.request

BASE = ("https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        "Qualcomm_AI_Runtime_Community/All/{v}/v{v}.zip")

# 2.28 hedef; komsulari da deniyoruz — 2.28 yoksa en yakin eski surum de ise
# yarayabilir (16-bit MatMul/LayerNorm kisitlari 2.3x'te sikilasmis olabilir).
CANDIDATES = [
    "2.24.0.240626", "2.25.0.240728", "2.26.0.240828", "2.26.2.240911",
    "2.27.0.240926", "2.28.0.241029", "2.28.2.241116", "2.29.0.241129",
    "2.30.0.250109", "2.31.0.250130", "2.32.0.250228",
]


def probe(ver: str, timeout: int = 25):
    url = BASE.format(v=ver)
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            size = int(r.headers.get("content-length", 0))
            return r.status, size, url
    except Exception as e:
        code = getattr(e, "code", None)
        return (code or type(e).__name__), 0, url


def main() -> None:
    vers = sys.argv[1:] or CANDIDATES
    print("=" * 66)
    print("QAIRT surum yoklama — Qualcomm Software Center dogrudan API")
    print("=" * 66)
    ok = []
    for v in vers:
        status, size, url = probe(v)
        if status == 200 and size > 100 * 1024 * 1024:
            print(f"  [VAR]  {v:16s} {size/1e9:5.2f} GB")
            ok.append((v, url))
        else:
            print(f"  [yok]  {v:16s} ({status})")
    print()
    if ok:
        print(">>> INDIRILEBILIR SURUMLER:")
        for v, url in ok:
            print(f"    {v}")
            print(f"      {url}")
        print()
        print(">>> Kullanmak icin config.env'e ekleyin (not defterine dokunmadan):")
        print(f'    OVERRIDE_QAIRT_ASSET_URL={ok[0][1]}')
        print( '    OVERRIDE_QNN_VERSION=<or. 2.28>')
        print(">>> Sonra 4. ve 6. adimlari calistirin.")
    else:
        print(">>> Hicbir aday indirilemedi. Software Center API'si eski")
        print("    surumleri de kapatmis demektir.")


if __name__ == "__main__":
    main()
