"""IOI prompts, head ablation hooks, and per-head direct logit attribution for Llama-style HF models."""

import random

import torch

TEMPLATES = [
    "Then, {A} and {B} went to the store. {B} gave a drink to",
    "When {A} and {B} got a snack at the cafe, {B} decided to give it to",
    "Afterwards, {A} and {B} went to the park. {B} gave a ball to",
    "Friends {A} and {B} found a book at the school. {B} gave it to",
]
NAMES = [
    "Mary", "John", "Tom", "James", "Anna", "Paul", "Sarah", "David", "Laura", "Mark",
    "Kate", "Peter", "Emma", "Jack", "Lisa", "Mike", "Susan", "Daniel", "Helen", "Chris",
]


def head_dim(cfg):
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def single_token_names(tok):
    return [n for n in NAMES if len(tok.encode(" " + n, add_special_tokens=False)) == 1]


def make_prompts(tok, n, seed=0, template_idx=None):
    """Return list of dicts: text, io, s (single-token names with leading space), template index."""
    rng = random.Random(seed)
    names = single_token_names(tok)
    out = []
    for i in range(n):
        t = template_idx if template_idx is not None else i % len(TEMPLATES)
        a, b = rng.sample(names, 2)
        # ABB: A and B ... B gave to -> answer A (IO = A, S = B)
        out.append({"text": TEMPLATES[t].format(A=a, B=b), "io": " " + a, "s": " " + b, "t": t})
    return out


def tok_id(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, s
    return ids[0]


def encode(tok, prompts, device):
    """All prompts must share a template so positions align; returns input_ids (B, T)."""
    enc = [tok(p["text"]).input_ids for p in prompts]
    L = {len(e) for e in enc}
    assert len(L) == 1, f"prompts have different lengths {L}; use one template per batch"
    return torch.tensor(enc, device=device)


def logit_diff(logits, tok, prompts):
    io = torch.tensor([tok_id(tok, p["io"]) for p in prompts], device=logits.device)
    s = torch.tensor([tok_id(tok, p["s"]) for p in prompts], device=logits.device)
    last = logits[:, -1, :]
    return last.gather(1, io[:, None])[:, 0] - last.gather(1, s[:, None])[:, 0]


class HeadAblator:
    """Forward pre-hook on each layer's o_proj that replaces chosen heads' z (pre-o_proj) values.

    mode: 'zero' -> 0; 'mean' -> provided mean z (T, d_head) per head; 'resample' -> z from donor run.
    Values are stored per (layer, head) as tensors broadcastable to (B, T, d_head).
    """

    def __init__(self, model, heads, mode, values=None):
        self.model, self.heads, self.mode, self.values = model, heads, mode, values or {}
        self.handles = []
        cfg = model.config
        self.d_head = head_dim(cfg)

    def __enter__(self):
        by_layer = {}
        for (l, h) in self.heads:
            by_layer.setdefault(l, []).append(h)
        for l, hs in by_layer.items():
            o_proj = self.model.model.layers[l].self_attn.o_proj
            self.handles.append(o_proj.register_forward_pre_hook(self._hook(l, hs)))
        return self

    def _hook(self, l, hs):
        def fn(module, args):
            (z,) = args
            z = z.clone()
            for h in hs:
                sl = slice(h * self.d_head, (h + 1) * self.d_head)
                if self.mode == "zero":
                    z[..., sl] = 0
                else:
                    z[..., sl] = self.values[(l, h)].to(z.dtype)
            return (z,)
        return fn

    def __exit__(self, *a):
        for hd in self.handles:
            hd.remove()


@torch.no_grad()
def collect_z(model, input_ids):
    """Return dict layer -> z (B, T, n_heads*d_head) at o_proj input."""
    zs, handles = {}, []
    for l, layer in enumerate(model.model.layers):
        def mk(l):
            def fn(module, args):
                zs[l] = args[0].detach()
            return fn
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk(l)))
    out = model(input_ids)
    for hd in handles:
        hd.remove()
    return zs, out.logits


@torch.no_grad()
def head_dla(model, tok, prompts, zs, final_resid):
    """Direct logit attribution of each head to IO-S logit diff at the last position.

    Uses the final RMSNorm scale from this run held fixed (standard DLA). Returns (n_layers, n_heads, B).
    """
    cfg = model.config
    nH, dH = cfg.num_attention_heads, head_dim(cfg)
    norm = model.model.norm
    x = final_resid[:, -1, :].float()
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)  # (B,1)
    W_U = model.lm_head.weight.float()
    io = torch.tensor([tok_id(tok, p["io"]) for p in prompts], device=x.device)
    s = torch.tensor([tok_id(tok, p["s"]) for p in prompts], device=x.device)
    dirs = (W_U[io] - W_U[s]) * norm.weight.float()[None, :]  # (B, D)
    out = torch.zeros(len(model.model.layers), nH, len(prompts))
    for l, layer in enumerate(model.model.layers):
        Wo = layer.self_attn.o_proj.weight.float()  # (D, nH*dH)
        z = zs[l][:, -1, :].float()  # (B, nH*dH)
        for h in range(nH):
            sl = slice(h * dH, (h + 1) * dH)
            contrib = z[:, sl] @ Wo[:, sl].T  # (B, D)
            out[l, h] = ((contrib * scale) * dirs).sum(-1).cpu()
    return out


@torch.no_grad()
def final_resid(model, input_ids):
    store = {}
    hd = model.model.norm.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach()))
    out = model(input_ids)
    hd.remove()
    return store["x"], out.logits
