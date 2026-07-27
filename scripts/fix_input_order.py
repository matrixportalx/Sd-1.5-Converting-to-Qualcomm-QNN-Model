#!/usr/bin/env python3
"""
Context binary'nin GIRDI SIRASINI olcer ve gerekirse ONNX'i duzelterek
yeniden uretilmesini ister.

NEDEN KRITIK
    Motor (local-dream QnnModel.hpp::executeUnetGraphs) tensorleri ISIMLE
    DEGIL INDEKSLE yaziyor:
        inputs[0] -> latents        (uint16, 1*4*64*64  = 16384 eleman)
        inputs[1] -> timestep       (int32)
        inputs[2] -> text_embedding (uint16, 1*77*768   = 59136 eleman)
    Sira bozuksa 59136 elemanlik metin gomme 16384 elemanlik tampona yazilir,
    tampon tasar ve surec "kod 1" ile oluyor. Cihazda gordugumuz
    "Could not free context" bunun kuyruguydu.

    Referans (calisan) binary : sample, timestamp, text_embedding
    Bizim uretilen binary     : text_embedding, timestamp, sample

    ONNX'te sira DOGRU oldugu halde binary ters cikiyor -> sirayi degistiren
    qairt araclari. Hangi kurala gore degistirdigini tahmin etmiyoruz: olculen
    permutasyonun TERSINI ONNX'e uygulayip bir kez yeniden urettiriyoruz.

CIKIS KODLARI
    0 : sira dogru, yapacak bir sey yok
    2 : ONNX duzeltildi -> cagiran taraf donusumu YENIDEN calistirmali
    1 : hata
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def onnx_input_order(path: str):
    import onnx
    m = onnx.load(path, load_external_data=False)
    return [i.name for i in m.graph.input]


def set_onnx_input_order(path: str, order):
    import onnx
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    by_name = {i.name: i for i in g.input}
    missing = [n for n in order if n not in by_name]
    if missing:
        raise SystemExit(f"ONNX'te bulunmayan girdi: {missing}")
    rest = [i.name for i in g.input if i.name not in order]
    new = [by_name[n] for n in list(order) + rest]
    del g.input[:]
    g.input.extend(new)
    onnx.save(m, path)          # agirliklar harici: yalnizca protobuf yazilir


def bin_input_order(bin_path: str):
    root = os.environ.get("QNN_SDK_ROOT")
    if not root:
        raise SystemExit("QNN_SDK_ROOT ayarli degil")
    tool = os.path.join(root, "bin", "x86_64-linux-clang",
                        "qnn-context-binary-utility")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (os.path.join(root, "lib", "x86_64-linux-clang")
                              + ":" + env.get("LD_LIBRARY_PATH", ""))
    out = os.path.join(tempfile.mkdtemp(), "meta.json")
    subprocess.run([tool, "--context_binary", bin_path, "--json_file", out],
                   env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(out):
        raise SystemExit(f"metadata uretilemedi: {bin_path}")
    with open(out) as f:
        meta = json.load(f)
    info = meta.get("info", meta)
    graphs = info.get("graphs") or []
    if not graphs:
        raise SystemExit("binary'de graf yok")
    gi = graphs[0].get("info", graphs[0])
    names = []
    for t in gi.get("graphInputs", []):
        ti = t.get("info", t)
        names.append(ti.get("name"))
    return names


def corrected_onnx_order(onnx_order, bin_order, want):
    """Araclarin uyguladigi POZISYON permutasyonunu olcup tersini dondurur.

    Olculen:  bin_order[i] = onnx_order[p[i]]
    Istenen:  want[i]      = yeni_onnx[p[i]]   ->  yeni_onnx[p[i]] = want[i]
    """
    pos = {n: k for k, n in enumerate(onnx_order)}
    if set(bin_order) != set(onnx_order):
        raise SystemExit(f"isimler ortusmuyor: onnx={onnx_order} bin={bin_order}")
    new = [None] * len(onnx_order)
    for i, name in enumerate(bin_order):
        new[pos[name]] = want[i]
    if any(x is None for x in new):
        raise SystemExit("permutasyon cozulemedi")
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--want", default="sample,timestamp,text_embedding")
    ap.add_argument("--fix-onnx", action="store_true",
                    help="ONNX girdi sirasini permute etmeyi dene. OLCULDU: "
                         "binary sirasini DEGISTIRMIYOR, o yuzden varsayilan "
                         "kapali. Sira artik --source_model_input_shape ile "
                         "veriliyor (bkz. 03_convert_qnn.sh IO_ORDER).")
    args = ap.parse_args()

    want = [w.strip() for w in args.want.split(",") if w.strip()]
    cur_bin = bin_input_order(args.bin)
    print(f"  [sira] binary   : {cur_bin}")
    print(f"  [sira] beklenen : {want}")

    if cur_bin == want:
        print("  [sira] OK — motorun indeks sirasiyla ayni.")
        return

    cur_onnx = onnx_input_order(args.onnx)
    print(f"  [sira] onnx     : {cur_onnx}")
    if not args.fix_onnx:
        print("  [sira] UYUSMAZLIK — motor tensorleri indeksle yazdigi icin bu")
        print("         paket cihazda yuklenmez (kod 1 / Could not free context).")
        sys.exit(3)
    new_onnx = corrected_onnx_order(cur_onnx, cur_bin, want)
    if new_onnx == cur_onnx:
        raise SystemExit(
            "  [sira] HATA: ONNX sirasi degistirilemedi ama binary hala yanlis.\n"
            "         Sirayi belirleyen ONNX degil; baska bir mekanizma var.")
    set_onnx_input_order(args.onnx, new_onnx)
    print(f"  [sira] ONNX yeni sira: {new_onnx}")
    print("  [sira] -> UNet yeniden uretilecek")
    sys.exit(2)


if __name__ == "__main__":
    main()
