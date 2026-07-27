#!/usr/bin/env python3
"""
ONNX girdilerinin adlarini/sekillerini ISTENEN SIRADA yazar.

qairt-converter'in girdi listesini `--source_model_input_shape` (-s)
argumanlarindan kurdugu varsayimini denemek icin: bu bayrak "the name and
dimension of ALL the input buffers to the network" diyor, yani sirayi da
belirlemesi bekleniyor.

Neden gerekli: ONNX'teki graph.input sirasini degistirmek binary'nin
graphInputs sirasini DEGISTIRMIYOR (denendi, olculdu). Motor ise tensorleri
indeksle yaziyor, dolayisiyla sira kritik.

Cikti (satir basina, sekmeyle ayrilmis):
    sample<TAB>1,4,64,64
    timestamp<TAB>1
    text_embedding<TAB>1,77,768
"""
import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--order", default="sample,timestamp,text_embedding",
                    help="virgulle ayrilmis istenen sira")
    args = ap.parse_args()

    import onnx
    m = onnx.load(args.onnx, load_external_data=False)
    dims = {}
    for i in m.graph.input:
        shp = []
        for d in i.type.tensor_type.shape.dim:
            # dinamik boyut varsa bos birak -> cagiran taraf atlar
            shp.append(str(d.dim_value) if d.dim_value else "?")
        dims[i.name] = shp

    want = [w.strip() for w in args.order.split(",") if w.strip()]
    order = [n for n in want if n in dims] + [n for n in dims if n not in want]
    for name in order:
        shp = dims[name]
        if "?" in shp:
            print(f"# atlandi (dinamik boyut): {name}", file=sys.stderr)
            continue
        print(f"{name}\t{','.join(shp)}")


if __name__ == "__main__":
    main()
