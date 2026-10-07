import json, sys
def eff(s): return sum(1 for ch in s if ("一" <= ch <= "鿿") or ch.isalnum())
d = json.load(open(sys.argv[1], encoding="utf-8"))
for sc in d["scripts"]:
    nums = [x for ln in sc["lines"] for x in ln["numbers"].split("；") if x.strip()]
    tot = sum(eff(ln["text"]) for ln in sc["lines"])
    print(f'{sc["id"]} {sc["title"]} lines={len(sc["lines"])} chars={tot} number_entries={len(nums)} title_len={len(sc["title"])}')
    if len(sys.argv) > 2:
        for i, ln in enumerate(sc["lines"], 1):
            print(f'  {i:2d} {eff(ln["text"]):3d} {ln["speaker"]}: {ln["text"]}')
