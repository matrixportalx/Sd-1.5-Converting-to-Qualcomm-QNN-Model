#!/usr/bin/env python3
"""
Pahali ara ciktilari Hugging Face'e (ozel depo) yedekler / geri alir.

NEDEN: Colab calisma zamani mobilde sekme arka plana atilinca kapaniyor.
prepare_data.py ~35 dk suruyor ve ciktisi (data.pkl + images/) kaybolunca
her sey bastan basliyor. Bu script o asamayi TEK SEFER odenir hale getirir:

    pull  -> depoda varsa indirir; prepare_data ATLANIR
    push  -> prepare_data bittiginde yukler

Drive'a alternatif: hesap kotasi yemez, ayni token'i (HF_TOKEN) kullanir ve
yeni oturumda dogrudan kaynaktan iner.

Kullanim:
    python stage_cache.py pull --repo user/sd-qnn-cache --key CyberRealistic \
        --dir /path/npuconvertv2 --files data.pkl images input_list_unet.txt
    python stage_cache.py push --repo ... (ayni argumanlar)

Cikis kodu: 0 = is yapildi, 3 = onbellekte yok (pull), 1 = hata.
"""
import argparse
import os
import sys

DEFAULT_FILES = ["data.pkl", "images"]


def _api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _repo_id(api, repo):
    if "/" in repo:
        return repo
    who = api.whoami()
    user = who.get("name") or who.get("email", "").split("@")[0]
    if not user:
        sys.exit("HATA: HF kullanici adi alinamadi (token gecerli mi?)")
    return f"{user}/{repo}"


def push(args):
    api = _api(args.token)
    repo = _repo_id(api, args.repo)
    api.create_repo(repo_id=repo, repo_type="dataset", private=True,
                    exist_ok=True)

    sent = 0
    for rel in args.files:
        p = os.path.join(args.dir, rel)
        if not os.path.exists(p):
            print(f"  [atla] yok: {rel}")
            continue
        dest = f"{args.key}/{rel}"
        if os.path.isdir(p):
            print(f"  [gonder] {rel}/ -> {repo}:{dest}/")
            api.upload_folder(folder_path=p, path_in_repo=dest,
                              repo_id=repo, repo_type="dataset",
                              commit_message=f"cache {args.key}/{rel}")
        else:
            mb = os.path.getsize(p) / 1e6
            print(f"  [gonder] {rel} ({mb:.0f} MB) -> {repo}:{dest}")
            api.upload_file(path_or_fileobj=p, path_in_repo=dest,
                            repo_id=repo, repo_type="dataset",
                            commit_message=f"cache {args.key}/{rel}")
        sent += 1
    print(f"[+] onbellege yazildi: {sent} oge -> "
          f"https://huggingface.co/datasets/{repo}/tree/main/{args.key}")


def pull(args):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (EntryNotFoundError,
                                       RepositoryNotFoundError)
    api = _api(args.token)
    try:
        repo = _repo_id(api, args.repo)
    except SystemExit:
        raise
    print(f"[*] onbellek araniyor: {repo}:{args.key}/")
    try:
        local = snapshot_download(
            repo_id=repo, repo_type="dataset", token=args.token,
            allow_patterns=[f"{args.key}/**"],
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        print("[-] onbellek yok (ilk kosu) — asamalar normal kosacak")
        return 3
    except Exception as e:                       # aglar/yetki vb.
        print(f"[-] onbellek okunamadi ({type(e).__name__}: {e})")
        return 3

    src_root = os.path.join(local, args.key)
    if not os.path.isdir(src_root):
        print("[-] onbellekte bu model yok — asamalar normal kosacak")
        return 3

    import shutil
    got = 0
    for rel in args.files:
        s = os.path.join(src_root, rel)
        if not os.path.exists(s):
            continue
        d = os.path.join(args.dir, rel)
        if os.path.exists(d):
            print(f"  [atla] yerelde zaten var: {rel}")
            got += 1
            continue
        os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
        print(f"  [al] {rel}")
        got += 1
    if not got:
        print("[-] onbellekte kullanilabilir oge yok")
        return 3
    print(f"[+] {got} oge onbellekten alindi — o asamalar atlanacak")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--repo", required=True,
                    help="'kullanici/depo' ya da sadece 'depo' (dataset)")
    ap.add_argument("--key", required=True, help="model adi (alt klasor)")
    ap.add_argument("--dir", required=True, help="yerel calisma dizini")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()
    if not a.token:
        sys.exit("HATA: HF_TOKEN yok (yazma izinli olmali).")
    sys.exit(push(a) or 0 if a.action == "push" else pull(a))


if __name__ == "__main__":
    main()
