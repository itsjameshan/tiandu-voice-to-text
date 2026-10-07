import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
VOCAB = ["拾光滇程旅行社", "陈导", "云上花间", "望湖索道", "环湖电瓶车",
         "昆明", "大理", "丽江", "石林",
         "自费项目", "自费", "团费", "定金", "尾款", "品质团", "经济团", "价目单", "差价",
         "团队价", "门市价", "手续费", "旅拍", "预交", "A区", "B区",
         "合同编号", "合同号", "导游证号", "订单号"]
for sc in d["scripts"]:
    for ln in sc["lines"]:
        t = ln["text"]
        found = []
        for w in VOCAB:
            if w in t:
                if w == "自费" and "自费项目" in t:
                    continue
                found.append((t.index(w), w))
        found.sort()
        ln["hotwords"] = "；".join(w for _, w in found)
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
# report brand / risky words anywhere in the file
raw = open(p, encoding="utf-8").read()
for w in ["微信", "支付宝", "美团", "携程", "抖音", "车牌", "国", "族", "教"]:
    if w in raw:
        print("FOUND", w)
print("done")
