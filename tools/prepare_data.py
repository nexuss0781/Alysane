#!/usr/bin/env python3
"""Offline corpus/token conversion only. No training, no labels fed to the core.
Sources: public-domain Gutenberg prose/science, Wikipedia math/science wikitext,
and local Python standard-library source code. Metadata is audit-only.
"""
import json, random, struct, glob, os, re
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data'
N_TRAIN = int(os.environ.get('ALYSANE_N_TRAIN', '256'))
N_HOLD = int(os.environ.get('ALYSANE_N_HOLD', '64'))
N_CTX = int(os.environ.get('ALYSANE_N_CTX', '64'))
SEED = 20261105

tok = Tokenizer.from_file(str(ROOT / 'models' / 'tokenizer.json'))

def clean_gutenberg(s):
    m = re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', s, re.S)
    if m: s = s[m.end():]
    m = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', s)
    if m: s = s[:m.start()]
    return s

def chunks(text, source):
    ids = tok.encode(text).ids
    return [(ids[i:i+N_CTX], f'{source}:{i}') for i in range(0, len(ids)-N_CTX+1, N_CTX)]

prose, science, code = [], [], []
for p in sorted(RAW.glob('gutenberg_*.txt')):
    text = clean_gutenberg(p.read_text(errors='ignore'))
    target = prose if p.stem in ('gutenberg_1342','gutenberg_1661','gutenberg_2701') else science
    target.extend(chunks(text, p.stem))
for p in sorted(RAW.glob('wiki_*.txt')):
    science.extend(chunks(p.read_text(errors='ignore'), p.stem))
import sysconfig
_stdlib = sysconfig.get_paths()['stdlib']
code_files = sorted(glob.glob(os.path.join(_stdlib, '*.py')))
code_files = [p for p in code_files if os.path.getsize(p) > 2000][:80]
for p in code_files:
    code.extend(chunks(Path(p).read_text(errors='ignore'), 'pystdlib/' + Path(p).name))

rng = random.Random(SEED)
def take(pool, n_train, n_hold):
    pool = list(pool); rng.shuffle(pool)
    # deduplicate by token tuple
    seen, uniq = set(), []
    for ids, meta in pool:
        t = tuple(ids)
        if t not in seen:
            seen.add(t); uniq.append((ids, meta))
    need = n_train + n_hold
    if len(uniq) < need: raise RuntimeError(f'not enough chunks: {len(uniq)} < {need}')
    return uniq[:n_train], uniq[n_train:need]

# Balanced natural mixture; source labels are never read by C++ gate logic.
# Balanced thirds across prose / code / science; labels are audit-only.
def split3(n):
    a = n // 3; b = n // 3; return a, b, n - a - b
tp, tc, ts = split3(N_TRAIN); hp, hc, hs = split3(N_HOLD)
tr_p, ho_p = take(prose, tp, hp)
tr_c, ho_c = take(code, tc, hc)
tr_s, ho_s = take(science, ts, hs)
train = tr_p + tr_c + tr_s
hold = ho_p + ho_c + ho_s
rng.shuffle(train); rng.shuffle(hold)
assert len(train) == N_TRAIN and len(hold) == N_HOLD

def write_bin(path, rows):
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', len(rows), N_CTX))
        for ids, meta in rows:
            f.write(struct.pack(f'<{N_CTX}i', *ids))
        f.write(struct.pack('<I', len(rows)))
        for ids, meta in rows:
            b = meta.encode(); f.write(struct.pack('<I', len(b)) + b)

write_bin(OUT / 'train.bin', train)
write_bin(OUT / 'holdout.bin', hold)
write_bin(OUT / 'parity.bin', train[:8])
manifest = {
    'seed': SEED, 'n_ctx': N_CTX, 'n_train': N_TRAIN, 'n_hold': N_HOLD,
    'tokenizer_vocab': tok.get_vocab_size(),
    'train_source_counts': {'prose': len(tr_p), 'code': len(tr_c), 'science_math': len(tr_s)},
    'hold_source_counts': {'prose': len(ho_p), 'code': len(ho_c), 'science_math': len(ho_s)},
    'note': 'Source metadata is audit-only; the C++ gate does not use labels. This is a pilot convenience mixture, not live commercial traffic.'
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
