#!/usr/bin/env python3
"""
Belirli bir tier (min/mid/high) icin QNN HTP backend-extensions config JSON'u
uretir. qnn-context-binary-generator bu dosyayi --config_file ile alir ve
context binary'yi dogru DSP mimarisine gore derler.

Kullanim:
    python gen_htp_config.py --tier min --output work/htp_min.json

NOT: HTP config semasi QNN SDK surumune gore ufak farkliliklar gosterebilir.
Asagidaki alanlar QNN/QAIRT 2.x icin gecerlidir. Uyumsuzluk olursa
$QNN_SDK_ROOT/examples icindeki htp config orneklerini referans alin.
"""
import argparse
import json

from soc_targets import get_tier


def build_config(tier_name: str) -> dict:
    t = get_tier(tier_name)
    return {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
        },
        "context": {
            # Agirliklari context binary icine gom (cihazda tek dosya)
            "weight_sharing_enabled": False,
        },
        "graphs": [
            {
                # HTP optimizasyon seviyesi: 3 = en agresif (offline hazirlik)
                "O": 3,
                "vtcm_mb": 8,
                "fp16_relaxed_precision": True,
            }
        ],
        "devices": [
            {
                "dsp_arch": t["dsp_arch"],
                "soc_id": t["soc_id"],
                "soc_model": t["soc_model"],
                "pd_session": "unsigned",
                "cores": [{"core_id": 0, "perf_profile": "burst"}],
            }
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="min", choices=["min", "mid", "high"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = build_config(args.tier)
    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[+] HTP config yazildi (tier={args.tier}, "
          f"dsp_arch={cfg['devices'][0]['dsp_arch']}) -> {args.output}")


if __name__ == "__main__":
    main()
