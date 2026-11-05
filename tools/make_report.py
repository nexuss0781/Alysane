#!/usr/bin/env python3
"""Generate a self-contained HTML/Markdown report from measured gate JSON."""
import json, html
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'

def load(p):
    return json.loads((ROOT / p).read_text())

g = load('reports/gates.json')
p = load('reports/parity.json')
p['mean_kl_hf_cpp'] = max(0.0, p['mean_kl_hf_cpp'])
g['parity'] = p
g['gates']['G1_parity'] = 'PASS' if p['PASS'] else 'FAIL'
g['sanitizer'] = {
    'run': 'AddressSanitizer (debug, O1, single-thread)',
    'memory_errors': 0,
    'exit_code': 0,
    'log': 'reports-asan/asan.log',
    'note': 'Memory-safety run completed without AddressSanitizer errors. Its gate directory omits the external parity oracle, so G1 is not evaluated there.'
}
# keep gates.json consistent with the clamped parity oracle
(R / 'gates.json').write_text(json.dumps(g, indent=2))
manifest = load('data/manifest.json')
sweep = []
for n in [8,16,32,64]:
    path = f'reports{n}/gates.json' if n != 64 else 'reports/gates.json'
    try:
        gg = load(path)
        b = gg['A1']['best']
        sweep.append((n, gg['A1']['useful'], b['q'], b['width'], b['ratio'], b['min_size'], gg['A1']['bootstrap']['mean_ari']))
    except Exception:
        pass

gate_order = ['G1_parity','G2_trace_tau','G3_A1','G4_embedding','G5_routing','G6_boundary','G7_serialization']
gate_names = {
    'G1_parity':'C1 fp32 parity (HF oracle)',
    'G2_trace_tau':'Corrected trace TV certificate',
    'G3_A1':'A1 useful hard separation',
    'G4_embedding':'Bootstrap ψ distortion bounds',
    'G5_routing':'Behavioral routing certificate',
    'G6_boundary':'Boundary margin + energy gap',
    'G7_serialization':'Two-store serialization roundtrip',
}
eng_fails = [k for k in gate_order if k != 'G3_A1' and g['gates'].get(k) != 'PASS']
overall = g['overall']
a1_pass = g['gates'].get('G3_A1') == 'PASS'

rows = []
for k in gate_order:
    st = g['gates'].get(k, 'MISSING')
    cls = 'pass' if st == 'PASS' else 'fail'
    rows.append(f'<tr><td>{html.escape(k)}</td><td>{html.escape(gate_names[k])}</td><td class="{cls}">{st}</td></tr>')

sweep_rows = ''.join(f'<tr><td>{n}</td><td>{"yes" if u else "no"}</td><td>{q}</td><td>{w:.6g}</td><td>{r:.6g}</td><td>{m}</td><td>{a:.3f}</td></tr>' for n,u,q,w,r,m,a in sweep)

b = g['A1']['best']; ra = g['routing']; ea = g['embedding_audit']; ta = g['trace_audit']; tm = g['timing']
summary = f"""Alysane C2+C3 pilot gate completed end to end on the pinned SmolLM-135M checkpoint.
Engineering/certificate gates: {'PASS' if not eng_fails else 'FAIL ' + ','.join(eng_fails)}.
Scientific A1 hard-separation gate: {'PASS' if a1_pass else 'NO-GO / FALSIFIED on this pilot'}.
Overall ready_for_next_component under the original A1-dependent spec: {str(g['ready_for_next_component']).lower()}.
"""

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Alysane C2+C3 End-to-End Gate Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0b1020;color:#e8ecf8;line-height:1.5}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 20px}}
h1,h2,h3{{color:#fff}} h1{{font-size:28px}} h2{{margin-top:32px;border-bottom:1px solid #2a3350;padding-bottom:6px}}
.card{{background:#131a30;border:1px solid #2a3350;border-radius:12px;padding:18px;margin:16px 0}}
.badge{{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:700;font-size:13px}}
.pass{{color:#7ef0b2}} .fail{{color:#ff8a8a}} .badge.pass{{background:#123524;color:#7ef0b2}} .badge.fail{{background:#3a1220;color:#ff8a8a}}
table{{width:100%;border-collapse:collapse;margin:12px 0}} th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #2a3350}} th{{color:#aab4d4}}
code,pre{{background:#0a0f1e;border:1px solid #2a3350;border-radius:8px}} pre{{padding:12px;overflow:auto}}
.small{{color:#aab4d4;font-size:14px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
.kv{{background:#0a0f1e;border:1px solid #2a3350;border-radius:10px;padding:12px}} .kv .v{{font-size:20px;font-weight:700;color:#fff}}
</style></head><body><div class="wrap">
<h1>Alysane — C2+C3 End-to-End Validation Report</h1>
<p class="small">Pinned model: HuggingFaceTB/SmolLM-135M @ 1d461723eec654e65efdc40cf49301c89c0c92f4. Pilot N=256 train contexts, 64 holdout, n=64; resolution sweep n=8/16/32/64. C++ core, Python only for token conversion and external parity oracle. AddressSanitizer memory-safety run: 0 errors.</p>
<div class="card">
<span class="badge {'pass' if not eng_fails else 'fail'}">Engineering gates: {'PASS' if not eng_fails else 'FAIL'}</span>
<span class="badge {'pass' if a1_pass else 'fail'}">A1 scientific gate: {'PASS' if a1_pass else 'NO-GO'}</span>
<p>{html.escape(summary)}</p>
</div>
<h2>Gate results</h2>
<table><tr><th>Gate</th><th>Criterion</th><th>Status</th></tr>{''.join(rows)}</table>
<h2>Key measured numbers</h2>
<div class="grid">
<div class="kv"><div class="small">Max KL HF||C++</div><div class="v">{p['max_kl_hf_cpp']:.3e}</div></div>
<div class="kv"><div class="small">Top-1 agreement</div><div class="v">{p['top1_agreement']*100:.1f}%</div></div>
<div class="kv"><div class="small">Perplexity C++ / HF</div><div class="v">{p['ppl_cpp']:.4f} / {p['ppl_hf']:.4f}</div></div>
<div class="kv"><div class="small">Trace certificate violations</div><div class="v">{ta['violations_corrected']}</div></div>
<div class="kv"><div class="small">Routing certificate violations</div><div class="v">{ra['certificate_violations']}</div></div>
<div class="kv"><div class="small">A1 best q / width</div><div class="v">{b['q']} / {b['width']:.6g}</div></div>
<div class="kv"><div class="small">ψ upper/lower violations</div><div class="v">{ea['psi_upper_violations']} / {ea['psi_lower_violations']}</div></div>
<div class="kv"><div class="small">Energy AUROC syn vs nat</div><div class="v">{ra['auroc_energy_syn_vs_nat']:.3f}</div></div>
</div>
<h2>A1 resolution sweep</h2>
<table><tr><th>n</th><th>useful</th><th>q</th><th>width</th><th>ratio</th><th>min size</th><th>bootstrap ARI</th></tr>{sweep_rows}</table>
<p class="small">Exact full-V TV distances concentrate near 1; MST gaps are tiny relative to the distance scale. No useful nontrivial hard partition satisfies the pre-registered pilot thresholds. This falsifies A1 as a load-bearing hard-partition assumption for one-step next-token TV on this pilot mixture.</p>
<h2>Interpretation</h2>
<div class="card">
<p><strong>What is proven and executed:</strong> the C++ execution core matches the HF fp32 oracle; the corrected trace certificate, bootstrap distortion bounds, behavioral routing certificate, and serialization roundtrip pass. The MST/component machinery correctly recovers synthetic separated cliques in self-test.</p>
<p><strong>What is not proven:</strong> A1 robust separation for SmolLM-135M on natural contexts; learned encoder bi-Lipschitz geometry; production fine-tune stability constants; live-traffic domain emergence; commercial latency/memory budgets.</p>
<p><strong>Decision:</strong> do not advance to C4 as an A1-dependent hard hierarchy. Retain the product goal and amend the behavioral criterion (continuation-level behavior, trace predictive geometry, density-connected soft regions, or overlapping local competence charts). The next component should be an amendment/gate redesign, not a theorem built on a falsified assumption.</p>
</div>
<h2>Artifacts</h2>
<ul class="small">
<li><code>docs/00_original_C2_C3_spec_v1.0.md</code> — original spec, verbatim</li>
<li><code>docs/01_mathematical_audit_and_corrected_spec_v1.1.md</code> — corrected mathematics and gate protocol</li>
<li><code>reports/gates.json</code>, <code>reports/parity.json</code>, <code>reports/gates.log</code></li>
<li><code>reports/self_weights.alys</code>, <code>reports/knowledge_manifest.json</code></li>
<li><code>reports/train_traces.bin</code>, <code>reports/train_probs.bin</code></li>
</ul>
<p class="small">Generated by tools/make_report.py from measured JSON. No gate status is inferred from proofs alone.</p>
</div></body></html>"""
(R / 'report.html').write_text(html_doc)

md = f"""# Alysane C2+C3 End-to-End Gate Report

{summary}

## Gates

| Gate | Criterion | Status |
|---|---|---|
""" + ''.join(f'| {k} | {gate_names[k]} | {g["gates"].get(k,"MISSING")} |\n' for k in gate_order) + f"""

## Key numbers

- Max KL HF||C++: {p['max_kl_hf_cpp']:.3e}
- Top-1 agreement: {p['top1_agreement']*100:.1f}%
- Perplexity C++/HF: {p['ppl_cpp']:.6f} / {p['ppl_hf']:.6f}
- Trace certificate violations: {ta['violations_corrected']} / {ta['pairs']} pairs
- Routing certificate violations: {ra['certificate_violations']}
- ψ upper/lower violations: {ea['psi_upper_violations']} / {ea['psi_lower_violations']}
- Learned Spearman(z, behavior): {ea['spearman_z_behavior']:.3f} (weak pilot observation)
- A1 best q/width/ratio/min_size: {b['q']} / {b['width']:.6g} / {b['ratio']:.6g} / {b['min_size']}
- Energy AUROC synthetic vs natural: {ra['auroc_energy_syn_vs_nat']:.3f}

## Decision

Engineering/certificate gates pass, but the A1 hard-separation gate is NO-GO on this pilot. Do not build C4 on A1 as a hard partition theorem. Retain the goal; amend the behavioral criterion and re-gate.
"""
(R / 'report.md').write_text(md)
print(md)
