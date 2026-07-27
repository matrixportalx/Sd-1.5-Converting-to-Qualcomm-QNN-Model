#!/usr/bin/env python3
"""
Adim 1 — diffusers pipeline -> Local Dream (Ruya) formatinda ara ciktilar.

Local Dream'in SD1.5-NPU yukleyicisi (app kaynak kodu) su yapiyi bekler:
  * token_emb.bin   : CLIP token gomme tablosu, HAM fp16 [vocab, 768]
  * pos_emb.bin     : CLIP pozisyon gomme, HAM fp32 [77, 768]
  * clip_v2.mnn     : CLIP transformer; giris 'input_embedding' [1,77,768],
                      cikis 'last_hidden_state' (gomme uygulama tarafinda yapilir)
  * unet.bin        : QNN (int8 kuantize) — graf adi 'unet'
  * vae_decoder.bin : QNN (fp16) — graf adi 'vae_decoder'
  * vae_encoder.bin : QNN (fp16, opsiyonel) — graf adi 'vae_encoder'
  * tokenizer.json  : HF tokenizer

Bu script ONNX/ham ciktilari uretir:
  token_emb.bin, pos_emb.bin, clip_v2.onnx, unet_<WxH>.onnx,
  vae_decoder.onnx, vae_encoder.onnx
Sonraki adimlar bunlari MNN/QNN'e cevirir.

Kullanim:
    python 01_export_onnx.py --pipeline work/pipeline --output work/onnx \
        --resolutions 512x512 --opset 17
"""
import argparse
import os

import numpy as np
import torch

from common import (LATENT_CHANNELS, TEXT_SEQ_LEN, parse_resolutions,
                    DEFAULT_RESOLUTIONS)


def onnx_export(*args, **kwargs):
    try:
        return torch.onnx.export(*args, dynamo=False, **kwargs)
    except TypeError:
        return torch.onnx.export(*args, **kwargs)


def simplify_onnx(path):
    try:
        import onnx
        from onnxslim import slim
        onnx.save(slim(onnx.load(path)), path)
        print("    [onnxslim] sadelestirildi")
    except Exception as e:
        print(f"    [onnxslim] atlandi ({type(e).__name__})")


# --------------------------------------------------------------------------
# CLIP: gomme tablolarini ayir + transformer'i input_embedding alacak sekilde
# --------------------------------------------------------------------------
def _find_clip_embeddings(te):
    """CLIPTextEmbeddings modulunu bul (surumden bagimsiz)."""
    for m in te.modules():
        if m.__class__.__name__ == "CLIPTextEmbeddings":
            return m
    # ad ile ara (yedek)
    for m in te.modules():
        if hasattr(m, "token_embedding") and hasattr(m, "position_embedding"):
            return m
    raise RuntimeError("CLIPTextEmbeddings bulunamadi")


def export_clip_split(pipe, out_dir, opset, pipeline_dir):
    # SDPA dikkat yolu ONNX'e IsNaN (MNN desteklemez) ekler; CLIP'i EAGER
    # dikkat ile yeniden yukle (sonuc sayisal olarak ayni).
    te = pipe.text_encoder.eval()
    try:
        from transformers import CLIPTextModel
        te = CLIPTextModel.from_pretrained(
            os.path.join(pipeline_dir, "text_encoder"),
            attn_implementation="eager").eval()
        print("    [clip] eager attention ile yuklendi (IsNaN onlenir)")
    except Exception as e:
        print(f"    [clip] eager yukleme basarisiz ({e}); mevcut kullaniliyor")
    hidden = te.config.hidden_size
    emb = _find_clip_embeddings(te)

    # 1) token_emb.bin — HAM fp32 [vocab, hidden]
    # Referans model: 49408*768*4 = 151,781,376 bayt -> fp32 (fp16 DEGIL).
    # TextEncoder.hpp 100MB esigine gore fp32 legacy yolunu secer.
    tok_w = emb.token_embedding.weight.detach().cpu().numpy().astype(np.float32)
    tok_path = os.path.join(out_dir, "token_emb.bin")
    tok_w.tofile(tok_path)
    print(f"[*] token_emb.bin  {tok_w.shape} fp32 -> {tok_path} "
          f"({os.path.getsize(tok_path)>>20} MB)")

    # 2) pos_emb.bin — HAM fp32 [77, hidden]
    pos_w = emb.position_embedding.weight.detach().cpu().numpy().astype(np.float32)
    pos_path = os.path.join(out_dir, "pos_emb.bin")
    pos_w.tofile(pos_path)
    print(f"[*] pos_emb.bin    {pos_w.shape} fp32 -> {pos_path} "
          f"({os.path.getsize(pos_path)} B)")

    # 3) clip_v2.onnx — giris input_embedding, cikis last_hidden_state.
    # text_encoder'i input_ids ile cagirir ama embeddings.forward'i gecici olarak
    # disaridan gelen gomme ile degistirir (icteki maskeleme surumden bagimsiz calisir).
    class ClipV2(torch.nn.Module):
        def __init__(self, text_encoder, emb_mod):
            super().__init__()
            self.te = text_encoder
            self.emb_mod = emb_mod

        def forward(self, input_embedding):
            orig = self.emb_mod.forward
            self.emb_mod.forward = lambda *a, **k: input_embedding
            try:
                ids = torch.zeros(input_embedding.shape[:2], dtype=torch.long)
                out = self.te(input_ids=ids)
            finally:
                self.emb_mod.forward = orig
            return out.last_hidden_state

    path = os.path.join(out_dir, "clip_v2.onnx")
    dummy = torch.randn(1, TEXT_SEQ_LEN, hidden)
    print(f"[*] clip_v2 -> {path}")
    onnx_export(
        ClipV2(te, emb).eval(), (dummy,), path,
        input_names=["input_embedding"],
        output_names=["last_hidden_state"],
        opset_version=opset,
        do_constant_folding=True,
    )
    simplify_onnx(path)

    # 4) clip.onnx — TAM CLIP (gomme dahil, giris input_ids). Referans paketlerde
    # clip.mnn hem clip_v2.mnn ile birlikte bulunur.
    class ClipFull(torch.nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.te = text_encoder

        def forward(self, input_ids):
            return self.te(input_ids=input_ids).last_hidden_state

    path_full = os.path.join(out_dir, "clip.onnx")
    dummy_ids = torch.randint(0, 1000, (1, TEXT_SEQ_LEN), dtype=torch.int32)
    print(f"[*] clip (tam) -> {path_full}")
    onnx_export(
        ClipFull(te).eval(), (dummy_ids,), path_full,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        opset_version=opset,
        do_constant_folding=True,
    )
    simplify_onnx(path_full)


# --------------------------------------------------------------------------
# VAE decoder / encoder
# --------------------------------------------------------------------------
def export_vae_decoder(pipe, out_dir, opset, res0):
    vae = pipe.vae.eval()

    class Decoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            # NOT: latent/scaling_factor bolmesi YOK — HTP 'Div' op'unu float
            # grafta olusturamiyor; olcekleme uygulama tarafinda yapilir.
            return self.vae.decode(latent).sample

    path = os.path.join(out_dir, "vae_decoder.onnx")
    dummy = torch.randn(1, LATENT_CHANNELS, res0.latent_h, res0.latent_w)
    print(f"[*] vae_decoder -> {path}")
    onnx_export(
        Decoder(vae), (dummy,), path,
        input_names=["latent"], output_names=["image"],
        opset_version=opset, do_constant_folding=True)
    simplify_onnx(path)


def export_vae_encoder(pipe, out_dir, opset, res0):
    vae = pipe.vae.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            # moments [1, 8, H/8, W/8] (mean+logvar); ornekleme/olcek uygulamada
            h = self.vae.encoder(image)
            return self.vae.quant_conv(h)

    path = os.path.join(out_dir, "vae_encoder.onnx")
    dummy = torch.randn(1, 3, res0.height, res0.width)
    print(f"[*] vae_encoder -> {path}")
    onnx_export(
        Encoder(vae), (dummy,), path,
        input_names=["image"], output_names=["moments"],
        opset_version=opset, do_constant_folding=True)
    simplify_onnx(path)


# --------------------------------------------------------------------------
# UNet
# --------------------------------------------------------------------------
def _strip_timestamp_expand(path):
    """EMNIYET AGI: izleme kancasi tutmazsa, 'timestamp' girisinden beslenen
    KIMLIK Expand dugumunu ONNX duzeyinde kaldirir.

    HTP bu dugumu Reshape'e cevirip  in:INT_32 -> out:UFIXED_POINT_8  istiyor;
    kabul listesinde bu ikili yok. Expand hedef sekli girisin kendi sekline
    esitse islem zaten kimliktir, guvenle silinebilir.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as e:
        print(f"    [expand] ONNX temizligi atlandi ({type(e).__name__})")
        return
    try:
        m = onnx.load(path)
        g = m.graph
        tin = next((i for i in g.input if i.name == "timestamp"), None)
        if tin is None:
            return
        shp = [d.dim_value for d in tin.type.tensor_type.shape.dim]
        inits = {i.name for i in g.initializer}
        init_val = {i.name: numpy_helper.to_array(i) for i in g.initializer
                    if i.name in inits}

        # timestamp'ten baslayarak tip-korur dugumler uzerinden ilerle
        reachable = {"timestamp"}
        rename, keep, removed = {}, [], 0
        for n in g.node:
            if (n.op_type == "Expand" and n.input and n.input[0] in reachable
                    and len(n.input) == 2 and n.input[1] in init_val
                    and [int(v) for v in init_val[n.input[1]].tolist()] == shp):
                rename[n.output[0]] = n.input[0]
                reachable.add(n.output[0])
                removed += 1
                continue
            if n.op_type in ("Identity", "Cast") and n.input and n.input[0] in reachable:
                reachable.add(n.output[0])
            keep.append(n)
        if not removed:
            return
        del g.node[:]
        g.node.extend(keep)
        for n in g.node:
            for i, x in enumerate(n.input):
                while x in rename:
                    x = rename[x]
                n.input[i] = x
        big = os.path.getsize(path) > 1_800_000_000
        if big:
            data = os.path.basename(path) + ".data"
            onnx.save(m, path, save_as_external_data=True,
                      all_tensors_to_one_file=True, location=data,
                      size_threshold=1024)
        else:
            onnx.save(m, path)
        print(f"    [expand] ONNX duzeyinde {removed} kimlik Expand kaldirildi")
    except Exception as e:
        print(f"    [expand] ONNX temizligi basarisiz ({type(e).__name__}: {e})")


def export_unet(pipe, out_dir, opset, res):
    unet = pipe.unet.eval()
    hidden = pipe.text_encoder.config.hidden_size

    # 16-bit graf siniri ile 8-bit ic graf arasina BARIYER (Clip -> ReluMinMax).
    #
    # Neden: motor sample/text_embedding/output'a uint16 yaziyor, ama v68/v69
    # 16-bit LayerNorm'u desteklemiyor -> ic hesap 8-bit olmali. Overrides ile
    # yalnizca sinir tensorlerini 16-bit yapinca QNN etiketi to_k/to_v uzerinden
    # ic grafa tasiyor ve cross-attention MatMul'unun ikinci operandini "agirlik"
    # sayip 16-bit'e cekiyor:
    #   "mixedPrecisionForWeights: attn2/MatMul_1 ... 8 bit activations with
    #    16 bit weights are not supported on backend"
    # Clip'in AGIRLIGI YOK; QNN 16-bit girisle 8-bit cikis arasina Convert
    # ekleyebilir ve etiket daha ileri gitmez.
    #
    # Sinirlar gercek dagilimin cok uzerinde (kirpma yapmaz), yalnizca op'un
    # sadelestirilmemesi icin sonlu.
    # NOT: converter opset-13 Clip'i desteklemiyor ("Operation Clip Not
    # Supported. Expected operator version: [1, 6, 11, 12]"). Ayrica artik
    # gerek yok — 16-bit sinir qairt-converter --config ile veriliyor
    # (bkz. gen_io_config.py). Denemek isteyen UNET_CLIP_BARRIER=1 verebilir.
    CLIP = 1.0e4
    USE_CLIP = os.environ.get("UNET_CLIP_BARRIER", "0") == "1"

    class UNetWrap(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states):
            if USE_CLIP:
                sample = torch.clamp(sample, -CLIP, CLIP)
                encoder_hidden_states = torch.clamp(
                    encoder_hidden_states, -CLIP, CLIP)
            out = self.unet(sample, timestep,
                            encoder_hidden_states=encoder_hidden_states).sample
            return torch.clamp(out, -CLIP, CLIP) if USE_CLIP else out

    # --- timestamp yolu: sinusoidal hesap yerine ONCEDEN HESAPLANMIS TABLO ---
    #
    # Motor (local-dream QnnModel.hpp) timestamp'i HAM INT32 yaziyor:
    #     int32_t *positionData = ...inputs[1]...; positionData[0] = timestep;
    # Yani graf sinirinda tip INT_32 olmak ZORUNDA — kuantize edilemez.
    #
    # Ama int32 bir tensor uzerindeki her sekil islemi (Expand, Unsqueeze,
    # Reshape) HTP'de "in:INT_32 -> out:UFIXED_POINT_8" istiyor ve bu ikili
    # kabul listesinde yok; int32 girdi ancak int32 cikti verebiliyor. Araya
    # Cast koymak da ise yaramiyor, converter Cast'i katlayip atiyor.
    #
    # HTP'nin int32'yi kuantize dunyaya baglamak icin kabul ettigi tek yol
    # Gather: veri kuantize, INDEKS int32, cikti kuantize. time_proj zaten
    # yalnizca t'nin fonksiyonu ve t tamsayi (motor static_cast<int> yapiyor),
    # o yuzden 1000 timestep'in tamami onceden hesaplanip [1000, 320] sabit
    # tabloya konabilir. Sonuc BIREBIR ayni, sadece sin/cos grafta degil.
    class TimeProjTable(torch.nn.Module):
        def __init__(self, time_proj, num_train_timesteps=1000):
            super().__init__()
            ts = torch.arange(num_train_timesteps, dtype=torch.float32)
            with torch.no_grad():
                table = time_proj(ts).float()      # [T, 320]
            self.register_buffer("table", table)

        def forward(self, timesteps):
            # index_select -> ONNX Gather(axis=0); indeks int32 kalir.
            return self.table.index_select(0, timesteps)

    # export_unet her cozunurluk icin bir kez cagriliyor -> tabloyu bir kez sar.
    # (Sinif her cagrida yeniden tanimlandigi icin isinstance ise yaramaz.)
    if type(unet.time_proj).__name__ != "TimeProjTable":
        n_train = getattr(pipe.scheduler.config, "num_train_timesteps", 1000)
        unet.time_proj = TimeProjTable(unet.time_proj, n_train)
        print(f"    [time_proj] {n_train}x{unet.time_proj.table.shape[1]} tablo "
              f"(sin/cos graftan cikti, Gather ile okunuyor)")

    path = os.path.join(out_dir, f"unet_{res.tag}.onnx")
    sample = torch.randn(1, LATENT_CHANNELS, res.latent_h, res.latent_w)
    # Referans binary ile ayni: timestamp = INT_32, dims [1].
    timestep = torch.tensor([1], dtype=torch.int32)
    ehs = torch.randn(1, TEXT_SEQ_LEN, hidden)
    print(f"[*] unet {res.tag} -> {path}")

    # diffusers UNet.forward icinde `timesteps.expand(sample.shape[0])` var.
    # timesteps sekli [1], batch da 1 oldugundan bu KIMLIK islemi — ama izleme
    # sirasinda yine de bir ONNX Expand dugumu yaziliyor. QNN bunu Reshape'e
    # cevirip su kombinasyonu istiyor:
    #   in[0]:INT_32 -> out[0]:UFIXED_POINT_8
    # HTP'nin kabul listesinde bu ikili YOK (int32 girdi ancak int32 cikti
    # verebiliyor). timestamp girisi INT_32 kalmak ZORUNDA (motor boyle
    # yaziyor), o yuzden cozum Expand'i grafa hic yazdirmamak.
    #
    # Girise Cast koymak ise ise yaramiyor: converter Cast'i katlayip atiyor
    # ("The cast op ... will be interpreted at conversion time") ve Expand yine
    # int32 aliyor. Bu yuzden izleme sirasinda kimlik expand'lari eliyoruz.
    _orig_expand = torch.Tensor.expand
    _n_skipped = [0]

    def _expand_identity_skip(self, *sizes):
        # NOT: izleme sirasinda sample.shape[0] duz int OLMAYABILIR (izlenmis
        # deger / SymInt). Bu yuzden tip kontrolu yerine int()'e cevirmeyi
        # deniyoruz — cevrilemezse orijinal expand'a dusuyoruz.
        try:
            tgt = sizes[0] if len(sizes) == 1 else None
            if isinstance(tgt, (list, tuple)) and len(tgt) == 1:
                tgt = tgt[0]
            if tgt is not None and self.dim() == 1 and int(self.shape[0]) == int(tgt):
                _n_skipped[0] += 1
                return self          # kimlik: dugum yazma
        except Exception:
            pass
        return _orig_expand(self, *sizes)

    torch.Tensor.expand = _expand_identity_skip
    try:
        onnx_export(
            UNetWrap(unet), (sample, timestep, ehs), path,
            # Isimler referansla birebir: sample/timestamp/text_embedding/output
            input_names=["sample", "timestamp", "text_embedding"],
            output_names=["output"],
            opset_version=opset, do_constant_folding=True)
    finally:
        torch.Tensor.expand = _orig_expand
    print(f"    [expand] {_n_skipped[0]} kimlik expand elendi")
    _strip_timestamp_expand(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--output", default="work/onnx")
    ap.add_argument("--resolutions", default="")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--unet-only", action="store_true")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    os.makedirs(args.output, exist_ok=True)
    resolutions = (parse_resolutions(args.resolutions)
                   if args.resolutions else DEFAULT_RESOLUTIONS)

    print(f"[*] Pipeline yukleniyor: {args.pipeline}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pipeline, torch_dtype=torch.float32,
        safety_checker=None, load_safety_checker=False)

    with torch.no_grad():
        if not args.unet_only:
            export_clip_split(pipe, args.output, args.opset, args.pipeline)
            export_vae_decoder(pipe, args.output, args.opset, resolutions[0])
            export_vae_encoder(pipe, args.output, args.opset, resolutions[0])
        for res in resolutions:
            export_unet(pipe, args.output, args.opset, res)

    # tokenizer.json (HF hizli tokenizer; tum SD1.5 icin ayni CLIP tokenizer)
    if not args.unet_only:
        try:
            from transformers import CLIPTokenizerFast
            tk = CLIPTokenizerFast.from_pretrained(
                os.path.join(args.pipeline, "tokenizer"))
            tk.save_pretrained(args.output)   # tokenizer.json yazar
            if os.path.exists(os.path.join(args.output, "tokenizer.json")):
                print("[*] tokenizer.json yazildi")
        except Exception as e:
            print("[!] tokenizer.json uretilemedi:", e)

    print("[+] ONNX/emb export tamam ->", args.output)


if __name__ == "__main__":
    main()
