import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
n = 0
for sc in data["scripts"]:
    for ln in sc["lines"]:
        if ln["hotwords"] == "大理古城":
            ln["hotwords"] = "大理"; n += 1
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("fixed", n)
