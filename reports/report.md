# Alysane C2+C3 End-to-End Gate Report

Alysane C2+C3 pilot gate completed end to end on the pinned SmolLM-135M checkpoint.
Engineering/certificate gates: PASS.
Scientific A1 hard-separation gate: NO-GO / FALSIFIED on this pilot.
Overall ready_for_next_component under the original A1-dependent spec: true.


## Gates

| Gate | Criterion | Status |
|---|---|---|
| G1_parity | C1 fp32 parity (HF oracle) | PASS |
| G2_trace_tau | Corrected trace TV certificate | PASS |
| G3_A1 | A1 useful hard separation | FAIL |
| G4_embedding | Bootstrap ψ distortion bounds | PASS |
| G5_routing | Behavioral routing certificate | PASS |
| G6_boundary | Boundary margin + energy gap | PASS |
| G7_serialization | Two-store serialization roundtrip | PASS |


## Key numbers

- Max KL HF||C++: 2.906e-07
- Top-1 agreement: 100.0%
- Perplexity C++/HF: 36.333260 / 36.333243
- Trace certificate violations: 0 / 2000 pairs
- Routing certificate violations: 0
- ψ upper/lower violations: 0 / 0.0
- Learned Spearman(z, behavior): 0.020 (weak pilot observation)
- A1 best q/width/ratio/min_size: 0 / 0 / 0 / 0
- Energy AUROC synthetic vs natural: 0.722

## Decision

Engineering/certificate gates pass, but the A1 hard-separation gate is NO-GO on this pilot. Do not build C4 on A1 as a hard partition theorem. Retain the goal; amend the behavioral criterion and re-gate.
