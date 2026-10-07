import json, sys
p = sys.argv[1]
data = json.load(open(p, encoding="utf-8"))
s1, s2, s3 = data["scripts"]

def find(sc, prefix):
    hits = [ln for ln in sc["lines"] if ln["text"].startswith(prefix)]
    assert len(hits) == 1, (sc["id"], prefix, len(hits))
    return hits[0]

# S1
s1["setting"] = "上午的石林景区，星野云途旅行社的何导带一个散客拼团从入口步行游览，一路讲解路线、景观、安全和集合安排，全程没有任何纠纷。"
ln = find(s1, "好，说一下游览路线。")
ln["text"] = ln["text"].replace("最后从另一边的出口出来", "最后走到出口")
ln = find(s1, "何导，前面那里好漂亮")
ln["text"] = "何导，前面那里好漂亮，我们会在那儿停下来拍照吗？"
ln = find(s1, "可以，那里本来就是拍照点")
ln["text"] = ln["text"].replace("可以，那里本来就是拍照点", "会的，那里本来就是拍照点")
ln = find(s1, "走完最后这一段")
ln["hotwords"] = "购物点"

# S2
s2["setting"] = "冬天上午，星野云途旅行社的何导和司机带一个由几个家庭拼成的小团，从昆明市区坐车到滇池边的海埂大坝看红嘴鸥，全程没有任何纠纷。"
for prefix in ["原来面包不能喂啊", "好，那我就站后面一点看", "听到没有，手机拿稳", "好嘞，我们这就过去"]:
    find(s2, prefix)["label"] = "正常讲解"
find(s2, "当然可以，那是大家自由购物")["hotwords"] = "进店；红嘴鸥"

json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("ok")
