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
    c_half = numpy_helper.from_array(np.float32([0.5]), "c_half")

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
        # DIKKAT: burada onceden Identity vardi ve donusturucu onu ELIYOR;
        # sample'in ilk tuketimi en sondaki Add'e kayiyordu (A senaryosunun
        # [t,e,s] cikmasinin sebebi buydu). Elenmeyecek bir op kullaniyoruz.
        "sample": [
            helper.make_node("Mul", [n_s, "c_half"], ["s_i"]),
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
        [table, shp_t, shp_e, c_half])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def _write_calib(work, names):
    """Kucuk kalibrasyon seti. Graf KUANTIZE olmali: HTP float Gather'i kabul
    etmiyor (ilk yoklamada dordu de burada oldu), gercek UNet ise 8-bit."""
    lines = []
    for i in range(2):
        toks = []
        for role, nm in (("sample", names["sample"]),
                         ("timestamp", names["timestamp"]),
                         ("text_embedding", names["text_embedding"])):
            pth = os.path.join(work, f"{role}_{i}.raw")
            if role == "timestamp":
                np.array([i * 100], dtype=np.int32).tofile(pth)
            else:
                np.random.randn(*SHAPES[role]).astype(np.float32).tofile(pth)
            toks.append(f"{nm}:={pth}")
        lines.append(" ".join(toks))
    lst = os.path.join(work, "input_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lst


def _write_io_config(work, names):
    """Gercek hattaki --config YAML'inin kucuk kopyasi: sinir tensorleri
    uint16, timestamp int32."""
    b = []
    b.append("Converted Graph:")
    b.append("  - Input Tensors:")
    b.append("  - Output Tensors:")
    b.append("")
    b.append("Input Tensor Configuration:")
    for i, (role, dt, src) in enumerate(
            (("sample", "uint16", "float32"),
             ("timestamp", "int32", "int32"),
             ("text_embedding", "uint16", "float32")), 1):
        b += [f"  # Input {i}", f"  - Name: {names[role]}",
              "    Src Model Parameters:", f"        DataType: {src}",
              "    Desired Model Parameters:", f"        DataType: {dt}", ""]
    b += ["", "Output Tensor Configuration:", "  # Output 1",
          "  - Name: output", "    Src Model Parameters:",
          "        DataType: float32", "    Desired Model Parameters:",
          "        DataType: uint16", ""]
    pth = os.path.join(work, "io.yaml")
    with open(pth, "w") as f:
        f.write("\n".join(b) + "\n")
    return pth


def convert_and_read_order(onnx_path, work, names, io_config=False):
    """ONNX -> DLC -> kuantize DLC -> context binary -> graphInputs sirasi."""
    dlc = os.path.join(work, "probe.dlc")
    cargs = [PY, os.path.join(BIN, "qairt-converter"),
             "--input_network", onnx_path, "--output_path", dlc,
             "--target_backend", "HTP"]
    if io_config:
        cargs += ["--config", _write_io_config(work, names)]
    r = subprocess.run(cargs, env=_env(), capture_output=True, text=True)
    if not os.path.exists(dlc):
        return None, f"converter basarisiz:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"

    qdlc = os.path.join(work, "probe_q.dlc")
    lst = _write_calib(work, names)
    r = subprocess.run([PY, os.path.join(BIN, "qairt-quantizer"),
                        "--input_dlc", dlc, "--input_list", lst,
                        "--act_bitwidth", "8", "--weights_bitwidth", "8",
                        "--bias_bitwidth", "32", "--target_backend", "HTP",
                        "--output_dlc", qdlc],
                       env=_env(), capture_output=True, text=True)
    if os.path.exists(qdlc):
        dlc = qdlc
    else:
        return None, f"quantizer basarisiz:\n{r.stdout[-1500:]}{r.stderr[-1500:]}"

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


NAMES = {"sample": "sample", "timestamp": "timestamp",
         "text_embedding": "text_embedding"}

# (etiket, bildirim sirasi, adlar, TUKETIM sirasi, io_config)
# Olculen kural: sira = optimize grafta ILK TUKETIM sirasi.
# Burada asil soru: --config (uint16 sinir) bu kurali bozuyor mu? Gercek UNet'te
# sira [text_embedding, timestamp, sample] cikiyor, oysa diffusers once
# time_proj, sonra conv_in, en son attention calistiriyor.
CASES = [
    ("A2 tuketim s,t,e  (config YOK)",
     ["sample", "timestamp", "text_embedding"], NAMES,
     ["sample", "timestamp", "text_embedding"], False),
    ("E  tuketim s,t,e  (config VAR)",
     ["sample", "timestamp", "text_embedding"], NAMES,
     ["sample", "timestamp", "text_embedding"], True),
    ("F  tuketim t,s,e  (config VAR)  kontrol",
     ["sample", "timestamp", "text_embedding"], NAMES,
     ["timestamp", "sample", "text_embedding"], True),
    ("G  tuketim e,t,s  (config VAR)  gercek UNet'e benzer",
     ["sample", "timestamp", "text_embedding"], NAMES,
     ["text_embedding", "timestamp", "sample"], True),
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
    for label, decl, names, consume, iocfg in CASES:
        work = os.path.join(root, label.split()[0])
        os.makedirs(work, exist_ok=True)
        onnx_path = os.path.join(work, "probe.onnx")
        build_onnx(onnx_path, decl, names, consume)
        order, err = convert_and_read_order(onnx_path, work, names, iocfg)
        rev = {v: k for k, v in names.items()}
        roles = [rev.get(n, n) for n in order] if order else None
        print(f"--- {label}")
        print(f"    bildirim : {decl}")
        print(f"    tuketim  : {consume}")
        print(f"    io-config: {'VAR' if iocfg else 'yok'}")
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

    if not ok:
        # Yedek plan icin gereken arayuzleri simdi dokuyoruz ki bir tur daha
        # kaybetmeyelim: eski qnn-onnx-converter -> qnn-model-lib-generator yolu.
        for tool in ("qnn-onnx-converter", "qnn-model-lib-generator"):
            path = os.path.join(BIN, tool)
            if not os.path.exists(path):
                print(f"\n--- {tool}: YOK")
                continue
            print(f"\n--- {tool} --help (yedek plan icin) ---")
            r = subprocess.run([PY, path, "--help"], env=_env(),
                               capture_output=True, text=True)
            txt = (r.stdout or "") + (r.stderr or "")
            for line in txt.splitlines()[:120]:
                print("    " + line)

    if not args.keep and not args.workdir:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
