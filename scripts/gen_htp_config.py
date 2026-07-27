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


def build_ext_config(tier_name: str, graph: str = "") -> dict:
    """HTP backend-uzantisi (netrun extensions) config'i: graph/device ayarlari.
    Bu dosya ANA config'ten degil, backend_extensions.config_file_path ile
    referans edilir. graphs/devices anahtarlari BURADA gecerlidir."""
    t = get_tier(tier_name)
    # DSP_ARCH env ile override edilebilir (or. v68 denemek icin: DSP_ARCH=v68).
    arch = os.environ.get("DSP_ARCH") or t["dsp_arch"]
    cfg = {
        "devices": [
            # dsp_arch belirleyici: v68 = en genis uyumluluk (Snapdragon 7 dahil)
            {"dsp_arch": arch}
        ],
    }
    # Optimizasyon seviyesi. Referans binary ile karsilastirmada tek gercek
    # ayarsal fark buydu:
    #   info.graphs[0].info.graphBlobInfo.info.optimizationLevel
    #     referans: 3   bizim: 0
    # 'graphs' blogu graph_names ister; graf adini artik biliyoruz (DLC adi).
    if graph:
        cfg["graphs"] = [{
            "graph_names": [graph],
            "O": int(os.environ.get("HTP_O", "3")),
        }]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="min", choices=["min", "mid", "high"])
    ap.add_argument("--output", required=True,
                    help="Ana config yolu (qnn-context-binary-generator --config_file)")
    ap.add_argument("--graph", default="",
                    help="graf adi (DLC adi). Verilirse optimizasyon seviyesi "
                         "(O) bu graf icin ayarlanir — referans binary O=3.")
    args = ap.parse_args()

    t = dict(get_tier(args.tier))
    if os.environ.get("DSP_ARCH"):
        t["dsp_arch"] = os.environ["DSP_ARCH"]
    out_dir = os.path.dirname(os.path.abspath(args.output))
    ext_path = os.path.join(out_dir, f"htp_ext_{args.tier}.json")

    # 1) Backend-uzantisi (graph/device) dosyasi
    with open(ext_path, "w") as f:
        json.dump(build_ext_config(args.tier, args.graph), f, indent=2)

    # 2) Ana config: yalnizca backend_extensions (uzanti .so + ext config yolu)
    main_cfg = {
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": ext_path,
        }
    }
    with open(args.output, "w") as f:
        json.dump(main_cfg, f, indent=2)

    _o = os.environ.get("HTP_O", "3")
    print(f"[+] HTP config yazildi (tier={args.tier}, dsp_arch={t['dsp_arch']}"
          + (f", graf={args.graph}, O={_o}" if args.graph else "") + ")")
    print(f"    ana : {args.output}")
    print(f"    ext : {ext_path}")


if __name__ == "__main__":
    main()
