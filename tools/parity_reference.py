#!/usr/bin/env python3
"""External reference oracle only (HuggingFace fp32). Not part of the C++ hot path."""
import json, struct, math
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def read_bin(path):
    with open(path, 'rb') as f:
        N, n = struct.unpack('<ii', f.read(8))
        ids = np.frombuffer(f.read(N*n*4), dtype=np.int32).reshape(N, n)
        m = struct.unpack('<I', f.read(4))[0]
        meta = []
        for _ in range(m):
            l = struct.unpack('<I', f.read(4))[0]
            meta.append(f.read(l).decode())
    return ids, meta

ids, meta = read_bin(ROOT / 'data' / 'parity.bin')
N, n = ids.shape
with open(ROOT / 'reports' / 'parity_cpp.bin', 'rb') as f:
    N2, n2, V = struct.unpack('<iii', f.read(12))
    cpp_logits = np.frombuffer(f.read(N2*n2*V*4), dtype=np.float32).reshape(N2, n2, V).copy()
    cpp_nll = np.frombuffer(f.read(N2*4), dtype=np.float32).copy()
    cpp_top1 = np.frombuffer(f.read(N2*n2*4), dtype=np.int32).reshape(N2, n2).copy()
assert (N2, n2) == (N, n)

model = LlamaForCausalLM.from_pretrained(ROOT / 'models', torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
kl = np.zeros((N, n), dtype=np.float64)
hf_nll = np.zeros(N, dtype=np.float64)
top1_match = 0
max_logit_abs = 0.0
with torch.no_grad():
    for i in range(N):
        x = torch.tensor(ids[i:i+1], dtype=torch.long)
        out = model(x).logits[0].float().numpy()
        max_logit_abs = max(max_logit_abs, float(np.max(np.abs(out - cpp_logits[i]))))
        # KL(HF || C++)
        z = out - out.max(axis=-1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
        zc = cpp_logits[i] - cpp_logits[i].max(axis=-1, keepdims=True)
        q = np.exp(zc); q /= q.sum(axis=-1, keepdims=True)
        kl[i] = np.sum(p * (np.log(np.clip(p, 1e-30, None)) - np.log(np.clip(q, 1e-30, None))), axis=-1)
        nll = 0.0
        for t in range(n-1):
            zt = out[t]
            mx = zt.max()
            nll += -(zt[ids[i, t+1]] - mx - math.log(np.exp(zt - mx).sum()))
        hf_nll[i] = nll / (n-1)
        top1_match += int(np.sum(out.argmax(axis=-1) == cpp_top1[i]))

ppl_cpp = float(np.exp(cpp_nll.mean()))
ppl_hf = float(np.exp(hf_nll.mean()))
res = {
    'N': int(N), 'n': int(n), 'V': int(V),
    'max_kl_hf_cpp': float(kl.max()),
    'mean_kl_hf_cpp': float(max(0.0, kl.mean())),
    'top1_agreement': float(top1_match / (N*n)),
    'max_logit_abs_diff': max_logit_abs,
    'ppl_cpp': ppl_cpp, 'ppl_hf': ppl_hf,
    'ppl_relative_diff': abs(ppl_cpp - ppl_hf) / ppl_hf,
    'nll_cpp': cpp_nll.tolist(), 'nll_hf': hf_nll.tolist(),
    'thresholds': {'max_kl': 1e-5, 'mean_kl': 1e-6, 'top1_agreement': 1.0, 'ppl_relative_diff': 1e-3},
}
res['PASS'] = bool(res['max_kl_hf_cpp'] <= 1e-5 and res['mean_kl_hf_cpp'] <= 1e-6 and res['top1_agreement'] == 1.0 and res['ppl_relative_diff'] <= 1e-3)
(ROOT / 'reports' / 'parity.json').write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ['max_kl_hf_cpp','mean_kl_hf_cpp','top1_agreement','max_logit_abs_diff','ppl_cpp','ppl_hf','ppl_relative_diff','PASS']}, indent=2))
