#!/usr/bin/env python3
"""
Uretilen context binary'nin GIRDI/CIKTI tiplerini dogrular.

Neden: Local Dream / Ruya motoru UNet tamponlarina SABIT tiple yazar
(QnnModel.hpp):

    inputs[0] sample         -> uint16_t   => UFIXED_POINT_16
    inputs[1] timestamp      -> int32_t    => INT_32
    inputs[2] text_embedding -> uint16_t   => UFIXED_POINT_16
    outputs[0] output        -> uint16_t   => UFIXED_POINT_16

Binary UFIXED_POINT_8 beklerse motor eleman basina 2 bayt yazar, tampon tasar
ve surec "kod 1" / "Could not free context" ile coker. Yani a8w8 UNet bu
uygulamada ASLA calismaz — paketi telefona atmadan once burada yakalayalim.

Kullanim:
    python check_bin_io.py --bin work/X/qnn/unet.bin --expect unet
    python check_bin_io.py --bin work/X/qnn/vae_decoder.bin   # sadece dokum
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

# Motorun bekledigi tipler (referans calisan binary'den dogrulandi)
EXPECT = {
    "unet": {
        "inputs": {
            "sample": "UFIXED_POINT_16",
            "timestamp": "INT_32",
            "text_embedding": "UFIXED_POINT_16",
        },
        "outputs": {"output": "UFIXED_POINT_16"},
    },
}


def dump_meta(bin_path: str) -> dict:
    root = os.environ.get("QNN_SDK_ROOT")
    if not root:
        raise SystemExit("QNN_SDK_ROOT ayarli degil")
    tool = os.path.join(root, "bin", "x86_64-linux-clang",
                        "qnn-context-binary-utility")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(root, "lib", "x86_64-linux-clang") \
        + ":" + env.get("LD_LIBRARY_PATH", "")
    out = os.path.join(tempfile.mkdtemp(), "meta.json")
    subprocess.run([tool, "--context_binary", bin_path, "--json_file", out],
                   env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(out):
        raise SystemExit(f"metadata uretilemedi: {bin_path}")
    with open(out) as f:
        return json.load(f)


def graphs_of(meta: dict):
    info = meta.get("info", meta)
    return info.get("graphs") or meta.get("graphs") or []


def tensors(gi: dict, key: str):
    for t in gi.get(key, []):
        ti = t.get("info", t)
        yield ti.get("name"), ti.get("dataType"), ti.get("dimensions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--expect", choices=sorted(EXPECT), default=None,
                    help="verilirse tipler dogrulanir; yoksa sadece dokum")
    ap.add_argument("--strict", action="store_true",
                    help="uyusmazlikta cikis kodu 1")
    args = ap.parse_args()

    meta = dump_meta(args.bin)
    size_mb = os.path.getsize(args.bin) / 1e6
    print(f"  [dogrulama] {os.path.basename(args.bin)} ({size_mb:.0f} MB)")

    problems = []
    for g in graphs_of(meta):
        gi = g.get("info", g)
        print(f"    graf: {gi.get('graphName')}")
        # SIRA da tipler kadar kritik: motor tensorleri isimle degil INDEKSLE
        # yaziyor (QnnModel.hpp). Sira bozuksa 59136 elemanlik text_embedding
        # 16384 elemanlik sample tamponuna yazilir -> tampon tasar -> kod 1.
        if args.expect == "unet":
            got_order = [n for n, _, _ in tensors(gi, "graphInputs")]
            want_order = list(EXPECT["unet"]["inputs"])
            if got_order != want_order:
                problems.append(
                    f"GIRDI SIRASI: {got_order} (beklenen {want_order})")
        for key, label in (("graphInputs", "girdi"), ("graphOutputs", "cikti")):
            for name, dtype, dims in tensors(gi, key):
                print(f"      {label:5s} {name:16s} {dtype}  {dims}")
                if not args.expect:
                    continue
                want = EXPECT[args.expect][
                    "inputs" if key == "graphInputs" else "outputs"].get(name)
                # Arac tipleri "QNN_DATATYPE_UFIXED_POINT_16" diye yaziyor,
                # EXPECT ise on eksiz tutuluyor -> karsilastirmadan once kirp.
                got = (dtype or "").replace("QNN_DATATYPE_", "")
                if want and got != want:
                    problems.append(f"{name}: {got} (beklenen {want})")

    if args.expect and problems:
        print()
        print("  !!! TIP UYUSMAZLIGI — bu paket cihazda YUKLENMEZ:")
        for p in problems:
            print(f"        - {p}")
        print("      Motor sample/text_embedding'e uint16, timestamp'e int32")
        print("      yazar (QnnModel.hpp). Duz UNET_MODE=a8w8 bu uygulamada")
        print("      calismaz; a8w8_io16cfg kullanin (config.env).")
        if args.strict:
            sys.exit(1)
    elif args.expect:
        print("  [dogrulama] OK — tipler motorun bekledigiyle esitleniyor.")


if __name__ == "__main__":
    main()
