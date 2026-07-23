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
import os

from soc_targets import get_tier


def build_ext_config(tier_name: str) -> dict:
    """HTP backend-uzantisi (netrun extensions) config'i: graph/device ayarlari.
    Bu dosya ANA config'ten degil, backend_extensions.config_file_path ile
    referans edilir. graphs/devices anahtarlari BURADA gecerlidir."""
    t = get_tier(tier_name)
    return {
        "graphs": [
            {"vtcm_mb": 8, "O": 3}
        ],
        "devices": [
            # dsp_arch belirleyici: v68 = en genis uyumluluk (Snapdragon 7 dahil)
            {"dsp_arch": t["dsp_arch"]}
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="min", choices=["min", "mid", "high"])
    ap.add_argument("--output", required=True,
                    help="Ana config yolu (qnn-context-binary-generator --config_file)")
    args = ap.parse_args()

    t = get_tier(args.tier)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    ext_path = os.path.join(out_dir, f"htp_ext_{args.tier}.json")

    # 1) Backend-uzantisi (graph/device) dosyasi
    with open(ext_path, "w") as f:
        json.dump(build_ext_config(args.tier), f, indent=2)

    # 2) Ana config: yalnizca backend_extensions (uzanti .so + ext config yolu)
    main_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": ext_path,
        }
    }
    with open(args.output, "w") as f:
        json.dump(main_cfg, f, indent=2)

    print(f"[+] HTP config yazildi (tier={args.tier}, dsp_arch={t['dsp_arch']})")
    print(f"    ana : {args.output}")
    print(f"    ext : {ext_path}")


if __name__ == "__main__":
    main()
