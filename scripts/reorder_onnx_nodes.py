#!/usr/bin/env python3
"""
ONNX dugumlerini, ISTENEN GIRDI SIRASI olusacak sekilde topolojik olarak
yeniden siralar.

OLCULEN KURAL (probe_input_order.py):
    QNN graf girdi sirasi = optimize grafta ILK TUKETIM sirasi.
    Bildirim sirasi, tensor adlari ve -s argumanlari ETKISIZ.

Motor (QnnModel.hpp) tensorleri indeksle yaziyor:
    inputs[0]=latents  inputs[1]=timestep  inputs[2]=text_embedding
Yani `sample` once, sonra `timestamp`, en son `text_embedding` tuketilmeli.

Nasil: Kahn topolojik siralamasi + oncelik. Her dugume, transitif olarak
bagli oldugu graf girdilerinin en kucuk oncelik degeri atanir; hazir
dugumler arasindan en dusuk oncelikli (esitlikte en eski) secilir. Boylece
sample'a bagli dugumler once, timestamp'e bagli olanlar sonra, yalnizca
text_embedding'e bagli olanlar en son cikar — bagimliliklar bozulmadan.

Dugum sirasini degistirmek ONNX semantigini DEGISTIRMEZ; graph.node yalnizca
topolojik olarak gecerli olmak zorundadir.

Kullanim:
    python reorder_onnx_nodes.py --onnx work/X/onnx/unet_512x512.onnx \
        --order sample,timestamp,text_embedding
"""
import argparse
import heapq
import os


def reorder(path: str, order):
    import onnx
    m = onnx.load(path, load_external_data=False)
    g = m.graph

    prio = {name: i for i, name in enumerate(order)}
    LOW = len(order) + 1

    initializers = {i.name for i in g.initializer}
    graph_inputs = {i.name for i in g.input}

    # tensor -> uretici dugum indeksi
    producer = {}
    for idx, n in enumerate(g.node):
        for o in n.output:
            if o:
                producer[o] = idx

    n_nodes = len(g.node)
    deps = [set() for _ in range(n_nodes)]      # dugum -> bagli oldugu dugumler
    dependents = [[] for _ in range(n_nodes)]
    node_prio = [LOW] * n_nodes

    for idx, n in enumerate(g.node):
        for t in n.input:
            if not t or t in initializers:
                continue
            if t in producer:
                deps[idx].add(producer[t])
            elif t in graph_inputs:
                node_prio[idx] = min(node_prio[idx], prio.get(t, LOW))
    for idx in range(n_nodes):
        for d in deps[idx]:
            dependents[d].append(idx)

    # oncelik yayilimi: bir dugumun onceligi, bagli oldugu dugumlerinkinden
    # kucuk olamaz demiyoruz — MIN aliyoruz ki sample'a bagli zincir once gelsin.
    # Topolojik sirayla ilerleyerek min'i asagi tasiriz.
    indeg = [len(deps[i]) for i in range(n_nodes)]
    stack = [i for i in range(n_nodes) if indeg[i] == 0]
    topo = []
    tmp_indeg = list(indeg)
    while stack:
        i = stack.pop()
        topo.append(i)
        for j in dependents[i]:
            tmp_indeg[j] -= 1
            if tmp_indeg[j] == 0:
                stack.append(j)
    for i in topo:
        for j in dependents[i]:
            node_prio[j] = min(node_prio[j], node_prio[i])

    # Kahn + (oncelik, orijinal sira) yigini
    heap = []
    remaining = list(indeg)
    for i in range(n_nodes):
        if remaining[i] == 0:
            heapq.heappush(heap, (node_prio[i], i))
    new_order = []
    while heap:
        _, i = heapq.heappop(heap)
        new_order.append(i)
        for j in dependents[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                heapq.heappush(heap, (node_prio[j], j))

    if len(new_order) != n_nodes:
        raise SystemExit(f"topolojik siralama tamamlanamadi "
                         f"({len(new_order)}/{n_nodes}) — dongu olabilir")

    old_nodes = list(g.node)
    first_before = _first_use(old_nodes, graph_inputs, initializers)
    new_nodes = [old_nodes[i] for i in new_order]
    first_after = _first_use(new_nodes, graph_inputs, initializers)

    print(f"  [dugum-sirasi] ilk tuketim ONCE : {first_before}")
    print(f"  [dugum-sirasi] ilk tuketim SONRA: {first_after}")

    if first_before == first_after:
        print("  [dugum-sirasi] degisiklik yok")
        return first_after

    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, path)          # agirliklar harici: yalnizca protobuf yazilir
    print(f"  [dugum-sirasi] yazildi -> {os.path.basename(path)}")
    return first_after


def _first_use(nodes, graph_inputs, initializers):
    """Graf girdilerinin ilk tuketim sirasi."""
    seen, out = set(), []
    for n in nodes:
        for t in n.input:
            if t in graph_inputs and t not in seen and t not in initializers:
                seen.add(t)
                out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--order", default="sample,timestamp,text_embedding")
    args = ap.parse_args()
    order = [x.strip() for x in args.order.split(",") if x.strip()]
    reorder(args.onnx, order)


if __name__ == "__main__":
    main()
