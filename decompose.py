"""Split the IOI self-repair after head ablation into: other heads, MLP layers, and final-norm scale.

Using the clean run's final RMSNorm scale, logit diff is linear in residual components:
  LD = sum_heads DLA + sum_MLP DLA + embed DLA   (exact up to float error; checked below)
So  LD_clean - LD_abl = [direct of ablated heads] - [rise of other heads] - [rise of MLPs]
                        - [norm term: LD_abl(real scale) - LD_abl(clean scale)]
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ioi import HeadAblator, TEMPLATES, encode, head_dim, logit_diff, make_prompts, tok_id

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
p.add_argument("--n", type=int, default=40)
p.add_argument("--sets", default="top_dla:15.28,14.13,12.13,14.15;paper:14.15,14.31,15.20,15.12")
p.add_argument("--out", required=True)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
cfg = model.config
nL, nH, dH = cfg.num_hidden_layers, cfg.num_attention_heads, head_dim(cfg)
norm, W_U = model.model.norm, model.lm_head.weight.float()


@torch.no_grad()
def components(ids):
    """Last-position residual pieces: z per layer, MLP out per layer, embedding, final resid, logits."""
    st = {"z": {}, "mlp": {}}
    hs = []
    for l, layer in enumerate(model.model.layers):
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: st["z"].__setitem__(l, a[0][:, -1].float())))
        hs.append(layer.mlp.register_forward_hook(lambda m, a, o, l=l: st["mlp"].__setitem__(l, o[:, -1].float())))
    hs.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: st.__setitem__("emb", o[:, -1].float())))
    hs.append(norm.register_forward_pre_hook(lambda m, a: st.__setitem__("resid", a[0][:, -1].float())))
    out = model(ids)
    for h in hs:
        h.remove()
    st["logits"] = out.logits
    return st


def scale_of(resid):
    return torch.rsqrt(resid.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)


def dla_parts(st, scale, dirs):
    heads = torch.zeros(nL, nH, dirs.shape[0], device=dirs.device)
    for l, layer in enumerate(model.model.layers):
        Wo = layer.self_attn.o_proj.weight.float()
        for h in range(nH):
            sl = slice(h * dH, (h + 1) * dH)
            heads[l, h] = (((st["z"][l][:, sl] @ Wo[:, sl].T) * scale) * dirs).sum(-1)
    mlps = torch.stack([((st["mlp"][l] * scale) * dirs).sum(-1) for l in range(nL)])
    emb = ((st["emb"] * scale) * dirs).sum(-1)
    return heads, mlps, emb


results = {"model": args.model, "sets": {}}
max_recon_err = 0.0
for spec in args.sets.split(";"):
    name, hl = spec.split(":")
    heads = [tuple(int(x) for x in h.split(".")) for h in hl.split(",")]
    agg = {}
    for mode in ["zero", "mean", "resample"]:
        acc = {k: [] for k in ["total", "direct", "other_heads_rise", "mlp_rise", "norm_term"]}
        mlp_rise_layers = torch.zeros(nL)
        for t in range(len(TEMPLATES)):
            pr = make_prompts(tok, args.n, seed=t, template_idx=t)
            ref = make_prompts(tok, args.n, seed=1000 + t, template_idx=t)
            ids, ref_ids = encode(tok, pr, args.device), encode(tok, ref, args.device)
            io = torch.tensor([tok_id(tok, x["io"]) for x in pr], device=args.device)
            s = torch.tensor([tok_id(tok, x["s"]) for x in pr], device=args.device)
            dirs = (W_U[io] - W_U[s]) * norm.weight.float()[None, :]

            # full-position z for replacement values
            zfull = {}
            hs = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                lambda m, a, l=l: zfull.__setitem__(l, a[0].detach())) for l in {l for l, _ in heads}]
            with torch.no_grad():
                model(ref_ids)
            for h in hs:
                h.remove()
            torch.manual_seed(t)
            perm = torch.randperm(args.n, device=args.device)
            vals = {}
            for (l, h) in heads:
                sl = slice(h * dH, (h + 1) * dH)
                vals[(l, h)] = (zfull[l][:, :, sl].mean(0, keepdim=True) if mode == "mean"
                                else zfull[l][perm][:, :, sl])

            c = components(ids)
            with HeadAblator(model, heads, mode, vals):
                a = components(ids)
            sc_c = scale_of(c["resid"])
            hc, mc, ec = dla_parts(c, sc_c, dirs)
            ha, ma, ea = dla_parts(a, sc_c, dirs)  # ablated run, clean scale
            ld_c = logit_diff(c["logits"], tok, pr)
            ld_a = logit_diff(a["logits"], tok, pr)
            ld_a_cleanscale = ha.sum((0, 1)) + ma.sum(0) + ea
            max_recon_err = max(max_recon_err, (ld_c - (hc.sum((0, 1)) + mc.sum(0) + ec)).abs().max().item())

            mask = torch.zeros(nL, nH, dtype=torch.bool)
            for (l, h) in heads:
                mask[l, h] = True
            mask = mask.to(args.device)
            dh = ha - hc
            acc["total"].append((ld_c - ld_a).cpu())
            acc["direct"].append((-dh[mask].sum(0)).cpu())
            acc["other_heads_rise"].append(dh[~mask].sum(0).cpu())
            acc["mlp_rise"].append((ma - mc).sum(0).cpu())
            acc["norm_term"].append((ld_a - ld_a_cleanscale).cpu())
            mlp_rise_layers += (ma - mc).sum(1).cpu()
        m = {k: torch.cat(v) for k, v in acc.items()}
        n = len(m["total"])
        agg[mode] = {k: {"mean": v.mean().item(), "se": (v.std() / n ** 0.5).item()} for k, v in m.items()}
        agg[mode]["check_total_minus_parts"] = (m["direct"] - m["other_heads_rise"] - m["mlp_rise"] - m["norm_term"] - m["total"]).abs().max().item()
        agg[mode]["mlp_rise_by_layer"] = (mlp_rise_layers / n).tolist()
        print(name, mode, {k: round(v["mean"], 3) for k, v in agg[mode].items() if isinstance(v, dict)},
              "check", round(agg[mode]["check_total_minus_parts"], 5))
    results["sets"][name] = {"heads": hl, "modes": agg}
results["max_clean_reconstruction_error"] = max_recon_err
print("max clean LD reconstruction error", max_recon_err)
os.makedirs(args.out, exist_ok=True)
json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
