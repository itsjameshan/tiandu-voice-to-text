import json, sys
def eff(s):
    return sum(1 for ch in s if ("一" <= ch <= "鿿") or ch.isalnum())
data = json.load(open(sys.argv[1], encoding="utf-8"))
for sc in data["scripts"]:
    print("=====", sc["id"], sc["title"])
    tot = 0
    for i, ln in enumerate(sc["lines"], 1):
        n = eff(ln["text"]); tot += n
        print(f"{i:>2} {n:>3} [{ln['speaker']}] [{ln['label']}] {ln['text'][:40]}")
    print("TOTAL", tot)
