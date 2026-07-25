#!/usr/bin/env python3
"""
QAIRT'in DESTEKLEDIGI SoC modellerini (ve varsa HTP mimarilerini) SDK'nin
kendisinden okur.

Neden: `--target_soc_model` sabit kodlanamaz — QAIRT 2.39
"SOC model SM8350 is not supported" diyor, yani surumden surume liste
degisiyor. Bu script listeyi calisma aninda cikarir; 03_convert_qnn.sh de
hedef HTP mimarisine (v68/v69/...) uyan DESTEKLENEN bir SoC secer.

Kullanim:
    python list_soc_models.py                # tum liste: "SOC<TAB>arch"
    python list_soc_models.py --arch v68     # o mimariye uyan ilk SoC (veya bos)
"""
import argparse
import os
import re
import sys

SOC_RE = re.compile(r"\b((?:SM|SDM|QCS|IPQ|SA|QC|SC|SXR|SSG|STP|QRB|AIC)"
                    r"[0-9][0-9A-Za-z_-]*)\b")
ARCH_RE = re.compile(r"\bv(6[0-9]|7[0-9]|8[0-9])\b", re.I)


def _from_module():
    """backend_awareness modulunu introspect et: SoC -> bilgi eslemesi ara."""
    try:
        import importlib
        mod = importlib.import_module(
            "qti.aisw.converters.common.backend_awareness")
    except Exception:
        return {}

    found = {}

    def harvest(d):
        for k, v in d.items():
            if isinstance(k, str) and SOC_RE.fullmatch(k):
                m = ARCH_RE.search(str(v))
                found.setdefault(k, ("v" + m.group(1)) if m else "")

    for name in dir(mod):
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if isinstance(obj, dict):
            harvest(obj)
        else:  # sinif icindeki sozlukler
            for attr in dir(obj):
                try:
                    sub = getattr(obj, attr)
                except Exception:
                    continue
                if isinstance(sub, dict):
                    harvest(sub)
    return found


def _from_files():
    """Modul introspect'i tutmazsa SDK python kaynaklarini tara."""
    root = os.environ.get("QNN_SDK_ROOT", "")
    base = os.path.join(root, "lib", "python", "qti", "aisw", "converters",
                        "common")
    found = {}
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if not fn.endswith((".py", ".json")):
                continue
            try:
                txt = open(os.path.join(dirpath, fn), errors="ignore").read()
            except Exception:
                continue
            if "soc" not in txt.lower():
                continue
            for line in txt.splitlines():
                socs = SOC_RE.findall(line)
                if not socs:
                    continue
                m = ARCH_RE.search(line)
                for s in socs:
                    found.setdefault(s, ("v" + m.group(1)) if m else "")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="", help="or. v68 — uyan ilk SoC'yi bas")
    args = ap.parse_args()

    socs = _from_module()
    for k, v in _from_files().items():
        if k not in socs or (not socs[k] and v):
            socs[k] = v

    if not socs:
        sys.exit(0)  # sessizce bos — cagiran taraf backend-aware'i atlar

    if args.arch:
        want = args.arch.lower()
        for s in sorted(socs):
            if socs[s].lower() == want:
                print(s)
                return
        return  # uyan yok -> bos cikti

    for s in sorted(socs):
        print(f"{s}\t{socs[s]}")


if __name__ == "__main__":
    main()
