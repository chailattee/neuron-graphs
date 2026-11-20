"""Pilot A: does the model self-repair on IOI when its top direct-effect heads are ablated?

For each template: clean logit diff (IO - S), per-head direct logit attribution (DLA), then ablate the
top-k DLA heads (zero / mean / resample) and compare
  total effect  = LD_clean - LD_ablated            (what actually happens)
  direct effect = DLA_H(clean z) - DLA_H(replacement z), using the clean final-norm scale
self-repair fraction = 1 - total / direct. Also lists heads whose DLA rises most under ablation.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ioi import HeadAblator, head_dim, TEMPLATES, collect_z, encode, final_resid, head_dla, logit_diff, make_prompts

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
p.add_argument("--n", type=int, default=40, help="prompts per template")
p.add_argument("--k", type=int, default=4, help="number of top-DLA heads to ablate")
p.add_argument("--paper_heads", default="14.15,14.31,15.20,15.12", help="IO-mover heads from Haider et al. (unverified)")
p.add_argument("--out", required=True)
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
dH = head_dim(model.config)
paper_heads = [tuple(int(x) for x in h.split(".")) for h in args.paper_heads.split(",") if h]

# Pass 1: per-template clean runs, DLA averaged over all prompts -> choose heads
per_t = {}
dla_sum = torch.zeros(nL, nH)
count = 0
for t in range(len(TEMPLATES)):
    prompts = make_prompts(tok, args.n, seed=t, template_idx=t)
    ref = make_prompts(tok, args.n, seed=1000 + t, template_idx=t)  # different names, for mean
    ids, ref_ids = encode(tok, prompts, args.device), encode(tok, ref, args.device)
    zs, logits = collect_z(model, ids)
    resid, _ = final_resid(model, ids)
    ref_zs, _ = collect_z(model, ref_ids)
    ld = logit_diff(logits, tok, prompts)
    dla = head_dla(model, tok, prompts, zs, resid)
    dla_sum += dla.sum(-1)
    count += len(prompts)
    per_t[t] = dict(prompts=prompts, ids=ids, zs=zs, resid=resid, ref_zs=ref_zs, ld=ld, dla=dla)

dla_mean = dla_sum / count
flat = [(dla_mean[l, h].item(), (l, h)) for l in range(nL) for h in range(nH)]
flat.sort(reverse=True)
top_heads = [lh for _, lh in flat[: args.k]]
neg_heads = [lh for _, lh in sorted(flat)[:3]]


def replacement(t, heads, mode, perm):
    d = per_t[t]
    vals = {}
    for (l, h) in heads:
        sl = slice(h * dH, (h + 1) * dH)
        if mode == "mean":
            vals[(l, h)] = d["ref_zs"][l][:, :, sl].mean(0, keepdim=True)  # (1, T, dH)
        elif mode == "resample":
            vals[(l, h)] = d["ref_zs"][l][perm][:, :, sl]  # donor prompts with other names
        else:
            vals[(l, h)] = torch.zeros_like(d["zs"][l][:, :, sl])
    return vals


def run_set(heads, label):
    res = {"heads": [f"{l}.{h}" for l, h in heads], "modes": {}}
    for mode in ["zero", "mean", "resample"]:
        tot, dirs, rise = [], [], torch.zeros(nL, nH)
        for t, d in per_t.items():
            perm = torch.randperm(args.n, device=args.device)
            vals = replacement(t, heads, mode, perm)
            with HeadAblator(model, heads, mode, vals):
                zs_a, logits_a = collect_z(model, d["ids"])
                resid_a, _ = final_resid(model, d["ids"])
            ld_a = logit_diff(logits_a, tok, d["prompts"])
            # direct effect: swap only H's z in the clean run's DLA (clean norm scale)
            zs_rep = {l: z.clone() for l, z in d["zs"].items()}
            for (l, h), v in vals.items():
                sl = slice(h * dH, (h + 1) * dH)
                zs_rep[l][:, :, sl] = v.expand_as(zs_rep[l][:, :, sl])
            dla_rep = head_dla(model, tok, d["prompts"], zs_rep, d["resid"])
            hidx = [(l, h) for l, h in heads]
            direct = sum(d["dla"][l, h] - dla_rep[l, h] for l, h in hidx)
            tot.append((d["ld"].cpu() - ld_a.cpu()))
            dirs.append(direct)
            dla_a = head_dla(model, tok, d["prompts"], zs_a, resid_a)
            rise += (dla_a - d["dla"]).sum(-1)
        tot, dirs = torch.cat(tot), torch.cat(dirs)
        rise = rise / count
        for (l, h) in heads:
            rise[l, h] = float("-inf")
        risers = sorted(((rise[l, h].item(), f"{l}.{h}") for l in range(nL) for h in range(nH)), reverse=True)[:8]
        res["modes"][mode] = {
            "total_effect_mean": tot.mean().item(),
            "total_effect_se": (tot.std() / len(tot) ** 0.5).item(),
            "direct_effect_mean": dirs.mean().item(),
            "direct_effect_se": (dirs.std() / len(dirs) ** 0.5).item(),
            "self_repair_fraction": 1 - tot.mean().item() / dirs.mean().item(),
            "top_dla_risers": [{"head": n, "dla_increase": v} for v, n in risers],
        }
        print(label, mode, {k: v for k, v in res["modes"][mode].items() if k != "top_dla_risers"})
    return res


all_ld = torch.cat([d["ld"].cpu() for d in per_t.values()])
results = {
    "model": args.model,
    "n_prompts": count,
    "clean_logit_diff_mean": all_ld.mean().item(),
    "clean_accuracy": (all_ld > 0).float().mean().item(),
    "top_dla_heads": [{"head": f"{l}.{h}", "dla": v} for v, (l, h) in flat[:15]],
    "most_negative_dla_heads": [{"head": f"{l}.{h}", "dla": dla_mean[l, h].item()} for l, h in neg_heads],
    "paper_heads_dla": {f"{l}.{h}": dla_mean[l, h].item() for l, h in paper_heads},
    "ablate_top_dla": run_set(top_heads, "top_dla"),
}
if paper_heads:
    results["ablate_paper_heads"] = run_set(paper_heads, "paper")

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps({k: results[k] for k in ["clean_logit_diff_mean", "clean_accuracy", "top_dla_heads", "paper_heads_dla"]}, indent=1))
