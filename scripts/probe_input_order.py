#!/usr/bin/env python3
"""
qairt araclarinin GRAF GIRDI SIRASINI hangi kurala gore belirledigini
KUCUK sahte modellerle olcer.

Neden: gercek UNet ile her deneme ~4 dakika. Ayni girdi imzasina sahip
oyuncak bir model saniyeler surer, dolayisiyla hipotezleri hizlica eleyebiliriz.

Olculen gercekler (UNet):
    ONNX bildirim sirasi     -> ETKISIZ
    --config YAML sirasi     -> ETKISIZ
    --source_model_input_shape sirasi -> ETKISIZ
    sonuc her zaman: text_embedding, timestamp, sample

Test edilen hipotezler:
    A) taban            : bildirim sirasi sample,timestamp,text_embedding
    B) bildirim ters     : sirayi bildirimin belirleyip belirlemedigi
    C) isim uzunlugu     : gozlenen sira azalan isim uzunluguyla ortusuyor
                           (text_embedding=14 > timestamp=9 > sample=6)
    D) ilk kullanim yeri : grafta once tuketilen girdi sona mi gidiyor

MOTOR ISIMLERE BAKMIYOR (QnnModel.hpp yalnizca inputs[0..2] indeksleri
kullaniyor), o yuzden C dogruysa girdileri YENIDEN ADLANDIRARAK cozebiliriz.

Kullanim:
    python probe_input_order.py                 # tum testler
    python probe_input_order.py --keep          # ara dosyalari birakma
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SDK = os.environ.get("QNN_SDK_ROOT", "")
BIN = os.path.join(SDK, "bin", "x86_64-linux-clang")
LIB = os.path.join(SDK, "lib", "x86_64-linux-clang")
PY = os.environ.get("QNN_PYTHON") or "python3"

# UNet ile ayni imza
SHAPES = {"sample": [1, 4, 64, 64], "timestamp": [1], "text_embedding": [1, 77, 768]}


def _env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = LIB + ":" + e.get("LD_LIBRARY_PATH", "")
    e["PATH"] = BIN + ":" + e.get("PATH", "")
    e["PYTHONPATH"] = os.path.join(SDK, "lib", "python") + ":" + e.get("PYTHONPATH", "")
    return e


def build_onnx(path, decl_order, names, consume_order):
    """Uc girdili kucuk model. names: {rol: gercek_ad}, roller
    sample/timestamp/text_embedding. consume_order roller listesi."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    n_s, n_t, n_e = names["sample"], names["timestamp"], names["text_embedding"]
    vi = {
        "sample": helper.make_tensor_value_info(n_s, TensorProto.FLOAT, SHAPES["sample"]),
        "timestamp": helper.make_tensor_value_info(n_t, TensorProto.INT32, SHAPES["timestamp"]),
        "text_embedding": helper.make_tensor_value_info(n_e, TensorProto.FLOAT, SHAPES["text_embedding"]),
    }

    table = numpy_helper.from_array(
        np.random.randn(1000, 4).astype(np.float32), "tp_table")
    shp_t = numpy_helper.from_array(np.array([1, 4, 1, 1], dtype=np.int64), "shp_t")
    shp_e = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), "shp_e")

    # rol -> o rolu tuketen dugum(ler); consume_order'a gore sirala
    nodes = {
        "timestamp": [
            helper.make_node("Gather", ["tp_table", n_t], ["t_g"], axis=0),
            helper.make_node("Reshape", ["t_g", "shp_t"], ["t_r"]),
        ],
        "text_embedding": [
            helper.make_node("ReduceMean", [n_e], ["e_m"], axes=[1, 2], keepdims=1),
            helper.make_node("Reshape", ["e_m", "shp_e"], ["e_r"]),
        ],
        "sample": [
            helper.make_node("Identity", [n_s], ["s_i"]),
        ],
    }
    graph_nodes = []
    for role in consume_order:
        graph_nodes += nodes[role]
    graph_nodes += [
        helper.make_node("Add", ["s_i", "t_r"], ["a1"]),
        helper.make_node("Add", ["a1", "e_r"], ["output"]),
    ]

    g = helper.make_graph(
        graph_nodes, "probe",
        [vi[r] for r in decl_order],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPES["sample"])],
        [table, shp_t, shp_e])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def convert_and_read_order(onnx_path, work):
    """ONNX -> DLC -> context binary -> graphInputs sirasi."""
    dlc = os.path.join(work, "probe.dlc")
    r = subprocess.run([PY, os.path.join(BIN, "qairt-converter"),
                        "--input_network", onnx_path, "--output_path", dlc,
                        "--target_backend", "HTP"],
                       env=_env(), capture_output=True, text=True)
    if not os.path.exists(dlc):
        return None, f"converter basarisiz:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"

    cfg = os.path.join(work, "htp.json")
    ext = os.path.join(work, "htp_ext.json")
    with open(ext, "w") as f:
        json.dump({"devices": [{"dsp_arch": os.environ.get("DSP_ARCH", "v68")}]}, f)
    with open(cfg, "w") as f:
        json.dump({"backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": ext}}, f)

    r = subprocess.run([os.path.join(BIN, "qnn-context-binary-generator"),
                        "--dlc_path", dlc,
                        "--backend", os.path.join(LIB, "libQnnHtp.so"),
                        "--config_file", cfg,
                        "--binary_file", "probe",
                        "--output_dir", work],
                       env=_env(), capture_output=True, text=True)
    binp = os.path.join(work, "probe.bin")
    if not os.path.exists(binp):
        return None, f"context-bin basarisiz:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"

    meta = os.path.join(work, "meta.json")
    subprocess.run([os.path.join(BIN, "qnn-context-binary-utility"),
                    "--context_binary", binp, "--json_file", meta],
                   env=_env(), capture_output=True, text=True)
    if not os.path.exists(meta):
        return None, "metadata okunamadi"
    with open(meta) as f:
        j = json.load(f)
    info = j.get("info", j)
    gi = info["graphs"][0].get("info", info["graphs"][0])
    return [t.get("info", t).get("name") for t in gi.get("graphInputs", [])], None


CASES = [
    ("A taban",
     ["sample", "timestamp", "text_embedding"],
     {"sample": "sample", "timestamp": "timestamp", "text_embedding": "text_embedding"},
     ["sample", "timestamp", "text_embedding"]),
    ("B bildirim ters",
     ["text_embedding", "timestamp", "sample"],
     {"sample": "sample", "timestamp": "timestamp", "text_embedding": "text_embedding"},
     ["sample", "timestamp", "text_embedding"]),
    ("C isim uzunlugu (latent en uzun ad)",
     ["sample", "timestamp", "text_embedding"],
     {"sample": "aaaaaaaaaaaaaa", "timestamp": "bbbbbbbbb", "text_embedding": "cc"},
     ["sample", "timestamp", "text_embedding"]),
    ("D tuketim sirasi ters",
     ["sample", "timestamp", "text_embedding"],
     {"sample": "sample", "timestamp": "timestamp", "text_embedding": "text_embedding"},
     ["text_embedding", "timestamp", "sample"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workdir", default="")
    args = ap.parse_args()

    if not SDK:
        raise SystemExit("QNN_SDK_ROOT ayarli degil")

    root = args.workdir or tempfile.mkdtemp(prefix="probe_order_")
    os.makedirs(root, exist_ok=True)
    print(f"[*] calisma klasoru: {root}")
    print(f"[*] hedef sira (motorun bekledigi): sample, timestamp, text_embedding")
    print()

    results = []
    for label, decl, names, consume in CASES:
        work = os.path.join(root, label.split()[0])
        os.makedirs(work, exist_ok=True)
        onnx_path = os.path.join(work, "probe.onnx")
        build_onnx(onnx_path, decl, names, consume)
        order, err = convert_and_read_order(onnx_path, work)
        rev = {v: k for k, v in names.items()}
        roles = [rev.get(n, n) for n in order] if order else None
        print(f"--- {label}")
        print(f"    bildirim : {decl}")
        print(f"    tuketim  : {consume}")
        print(f"    adlar    : {names}")
        if err:
            print(f"    SONUC    : HATA — {err}")
        else:
            print(f"    binary   : {order}")
            print(f"    roller   : {roles}")
            print(f"    ISTENEN? : "
                  f"{'EVET' if roles == ['sample','timestamp','text_embedding'] else 'hayir'}")
        print()
        results.append((label, roles, err))

    print("=" * 70)
    ok = [l for l, r, e in results if r == ["sample", "timestamp", "text_embedding"]]
    if ok:
        print("ISE YARAYAN YAPILANDIRMA(LAR):")
        for l in ok:
            print(f"  * {l}")
    else:
        print("Hicbiri istenen sirayi vermedi -> sira DLC yolunda sabit.")
        print("Kalan yol: qnn-onnx-converter -> qnn-model-lib-generator ->")
        print("           qnn-context-binary-generator --model <.so>")
        print("(referans paketin uretildigi eski yol; girdi sirasi model.cpp'deki")
        print(" bildirim sirasidir)")
    print("=" * 70)

    if not args.keep and not args.workdir:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
