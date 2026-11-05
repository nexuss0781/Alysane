#!/usr/bin/env python3
"""Independent NumPy oracle for the C2+C3 mathematical certificates.

Recomputes, from scratch and without reusing any C++ code:
  * exact full-V total variation from cached probabilities
  * the corrected trace bound |TV - hatTV| <= rho_i + rho_j
  * the landmark psi upper/lower distortion bounds
  * the behavioural routing certificate d(x,y) <= d(x,x*) + diam(D)
  * the subdominant ultrametric property U <= D and strong triangle inequality

Writes reports/certificates.json. This is an ORACLE, not part of the hot path.
"""
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def read_probs(path, N, V):
    with open(path, "rb") as f:
        n, v = struct.unpack("<ii", f.read(8))
        assert (n, v) == (N, V), (n, v, N, V)
        return np.frombuffer(f.read(n * v * 4), dtype=np.float32).reshape(n, v).copy()


def read_traces(path):
    out = []
    with open(path, "rb") as f:
        N, L, d, H = struct.unpack("<iiii", f.read(16))
        for _ in range(N):
            def vec():
                (m,) = struct.unpack("<I", f.read(4))
                return np.frombuffer(f.read(m * 4), dtype=np.float32).copy()

            vec()  # pooled (not needed for distributional certificates)
            vec()  # attn entropy
            (k,) = struct.unpack("<I", f.read(4))
            ids = np.frombuffer(f.read(k * 4), dtype=np.int32).copy()
            probs = vec()
            tail, ent, nll = struct.unpack("<fff", f.read(12))
            out.append(dict(ids=ids, probs=probs, tail=tail, entropy=ent))
    return out, L, d, H


def topk_tv(a, b, V):
    """Tail-uniform reconstruction TV via dense arrays (independent of C++ map logic)."""
    k = len(a["ids"])
    pa = np.full(V, a["tail"] / (V - k), dtype=np.float64)
    pb = np.full(V, b["tail"] / (V - k), dtype=np.float64)
    pa[a["ids"]] = a["probs"]
    pb[b["ids"]] = b["probs"]
    return 0.5 * np.abs(pa - pb).sum()


def prim_mst(D):
    """Independent O(N^2) Prim implementation."""
    n = D.shape[0]
    key = np.full(n, np.inf)
    par = np.full(n, -1, dtype=int)
    inmst = np.zeros(n, dtype=bool)
    key[0] = 0.0
    edges = []
    for _ in range(n):
        u = int(np.argmin(np.where(inmst, np.inf, key)))
        inmst[u] = True
        if par[u] >= 0:
            edges.append((par[u], u, float(key[u])))
        upd = (~inmst) & (D[u] < key)
        key[upd] = D[u][upd]
        par[upd] = u
    return edges


def main():
    gates = json.loads((REP / "gates.json").read_text())
    V = gates["knowledge"]["V"]
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    N = manifest["n_train"]

    P = read_probs(REP / "train_probs.bin", N, V).astype(np.float64)
    traces, L, d, H = read_traces(REP / "train_traces.bin")
    rng = np.random.default_rng(20261105)
    res = {"V": V, "N": N, "checks": {}}

    # ---- exact TV matrix (independent, vectorised) ----
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        D[i] = 0.5 * np.abs(P - P[i]).sum(axis=1)
    res["tv_matrix"] = {
        "min": float(D.min()),
        "median": float(np.median(D[np.triu_indices(N, 1)])),
        "max": float(D.max()),
        "symmetry_max_err": float(np.abs(D - D.T).max()),
    }

    # ---- corrected trace certificate ----
    pairs = 2000
    ii = rng.integers(0, N, size=(pairs * 2, 2))
    ii = ii[ii[:, 0] != ii[:, 1]][:pairs]
    viol_safe = viol_half = 0
    max_err = max_tau = max_half_excess = 0.0
    for i, j in ii:
        exact = D[i, j]
        hat = topk_tv(traces[i], traces[j], V)
        tau = traces[i]["tail"] + traces[j]["tail"]
        err = abs(exact - hat)
        max_err = max(max_err, err)
        max_tau = max(max_tau, tau)
        if err > tau + 1e-6:
            viol_safe += 1
        if err > 0.5 * tau + 1e-6:
            viol_half += 1
            max_half_excess = max(max_half_excess, err - 0.5 * tau)
    res["checks"]["trace_bound"] = {
        "pairs": int(len(ii)),
        "violations_safe_tau": viol_safe,
        "violations_original_half_tau": viol_half,
        "max_abs_error": max_err,
        "max_tau": max_tau,
        "max_half_tau_excess": max_half_excess,
        "PASS": viol_safe == 0,
        "note": "safe tau = rho_i + rho_j; original half-sum reported, not assumed",
    }

    # ---- landmark psi distortion ----
    M = 32
    lm = [0]
    mind = D[0].copy()
    while len(lm) < M:
        nxt = int(np.argmax(mind))
        lm.append(nxt)
        mind = np.minimum(mind, D[nxt])
    psi = D[:, lm]
    cov = float(D[:, lm].min(axis=1).max())
    up_viol = lo_viol = 0
    tested = 0
    for _ in range(3000):
        i, j = rng.integers(0, N, size=2)
        if i == j:
            continue
        tested += 1
        dz = np.linalg.norm(psi[i] - psi[j])
        db = D[i, j]
        if dz > np.sqrt(M) * db + 1e-5:
            up_viol += 1
        if dz < max(0.0, db - 2 * cov) - 1e-5:
            lo_viol += 1
    res["checks"]["psi_distortion"] = {
        "M": M,
        "pairs": tested,
        "coverage_radius": cov,
        "upper_violations": up_viol,
        "lower_violations": lo_viol,
        "PASS": up_viol == 0 and lo_viol == 0,
    }

    # ---- subdominant ultrametric ----
    edges = prim_mst(D)
    adj = [[] for _ in range(N)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    U = np.zeros((N, N), dtype=np.float64)
    for s in range(N):
        seen = np.zeros(N, dtype=bool)
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                U[s, v] = max(U[s, u], w)
                stack.append(v)
    below = bool(np.all(U <= D + 1e-6))
    strong = True
    for i in range(N):
        if np.any(U[i] > np.maximum(U[i][:, None], U).max(axis=0) + 1e-6):
            pass
    # vectorised strong triangle check
    strong = bool(np.all(U[:, None, :] <= np.maximum(U[:, :, None], U[None, :, :]) + 1e-6))
    res["checks"]["ultrametric"] = {
        "subdominant_U_le_D": below,
        "strong_triangle": strong,
        "symmetry_max_err": float(np.abs(U - U.T).max()),
        "PASS": below and strong,
    }

    # ---- routing certificate on the MST-gap diagnostic partition ----
    ws = sorted(w for _, _, w in edges)
    best_q, best_gap = 0, -1.0
    for q in range(2, min(16, N - 1) + 1):
        gap = ws[N - q] - ws[N - q - 1]
        if gap > best_gap:
            best_gap, best_q = gap, q
    cut = 0.5 * (ws[N - best_q - 1] + ws[N - best_q])
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, w in edges:
        if w <= cut:
            a, b = find(u), find(v)
            if a != b:
                parent[b] = a
    lab = np.array([find(i) for i in range(N)])
    uniq = {r: k for k, r in enumerate(sorted(set(lab)))}
    lab = np.array([uniq[r] for r in lab])
    q = int(lab.max() + 1)
    diam = np.zeros(q)
    for c in range(q):
        idx = np.where(lab == c)[0]
        if len(idx) > 1:
            diam[c] = D[np.ix_(idx, idx)].max()
    cert_viol = 0
    for x in range(N):
        xstar = int(np.argmin(D[x]))
        r = D[x, xstar]
        c = lab[xstar]
        bound = r + diam[c]
        same = np.where(lab == c)[0]
        cert_viol += int(np.sum(D[x, same] > bound + 1e-6))
    res["checks"]["routing_certificate"] = {
        "partition_q": q,
        "cut": cut,
        "certificate_violations": cert_viol,
        "PASS": cert_viol == 0,
        "theorem": "d(x,y) <= d(x,x*) + diam(D) for all y in the region of x*",
    }

    res["overall_PASS"] = all(c["PASS"] for c in res["checks"].values())
    REP.mkdir(exist_ok=True)
    (REP / "certificates.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["overall_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
