import json, sys, re
sys.path.insert(0, '.')
HOTCANON = None
src = open('build_g5.py', encoding='utf-8').read()
ns = {}
exec(src.split('# ---------------------------------------------------------------- S1')[0].replace('OUT = sys.argv[1]', 'OUT=None'), ns)
HOT = ns['HOT']
d = json.load(open('../scripts/group_5.json', encoding='utf-8'))
def eff(s): return sum(1 for ch in s if ('一' <= ch <= '鿿') or ch.isalnum())
NUMCH = set('零一二两三四五六七八九十百千万')
for sc in d['scripts']:
    print('=' * 20, sc['id'], sc['title'])
    surf, canon, total = set(), set(), 0
    for i, ln in enumerate(sc['lines'], 1):
        hw = [h for h in ln['hotwords'].split('；') if h]
        total += len(hw)
        surf.update(hw); canon.update(HOT[h] for h in hw)
        mark = '#' if any(c in NUMCH for c in ln['text']) else ' '
        print(f"{i:2d} {mark} [{eff(ln['text']):3d}] {ln['speaker']}|{ln['label']}| {ln['text']}  <N:{ln['numbers']}> <H:{ln['hotwords']}> {ln['direction']}")
    print(f"eff={sum(eff(l['text']) for l in sc['lines'])} lines={len(sc['lines'])} distinct_surface={len(surf)} distinct_canonical={len(canon)} total_occ={total}")
    print('  surfaces:', sorted(surf))
