import json
import re
import sys

LABELS = {"购物安排", "费用", "行程变更", "服务态度", "威胁消费", "正常讲解", "其他"}
FIVE_PERSON_GROUPS = {3, 8}

BANNED = [
    "滚", "傻", "蠢", "白痴", "穷鬼", "穷人", "没素质", "骗子", "王八", "他妈", "妈的", "混蛋", "废物",
    "民族", "彝族", "白族", "傣族", "纳西", "藏族", "苗族", "哈尼", "撒尼", "回族", "汉族", "壮族",
    "国家", "政治", "共产", "领导人", "外国", "日本", "美国", "韩国", "台湾", "香港",
    "外地人", "乡下人", "农村人",
    "佛教", "宗教", "信仰", "菩萨", "开光", "寺",
    "报警", "警察", "派出所", "打人", "动手", "不让走", "关门", "锁门",
    "治病", "治疗", "降血压", "降血糖", "抗癌", "疗效",
    "FCO9021N", "松某某", "金某某",
]


def effective_len(s):
    return sum(1 for ch in s if ("一" <= ch <= "鿿") or ch.isalnum())


def main(path, group):
    errors = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("group") != group:
        errors.append(f"group should be {group}, got {data.get('group')}")
    scripts = data.get("scripts", [])
    if len(scripts) != 3:
        errors.append(f"need exactly 3 scripts, got {len(scripts)}")
    max_roles = 5 if group in FIVE_PERSON_GROUPS else 4
    summary = []
    for k, sc in enumerate(scripts, start=1):
        sid = sc.get("id")
        if sid != f"G{group}-S{k}":
            errors.append(f"script {k}: id should be G{group}-S{k}, got {sid}")
        for key in ("title", "setting", "focus"):
            if not str(sc.get(key, "")).strip():
                errors.append(f"{sid}: missing {key}")
        roles = sc.get("roles", [])
        if not (2 <= len(roles) <= max_roles):
            errors.append(f"{sid}: roles must be 2..{max_roles}, got {len(roles)}")
        lines = sc.get("lines", [])
        if not (30 <= len(lines) <= 75):
            errors.append(f"{sid}: line count {len(lines)} outside 30..75")
        total = 0
        used = set()
        for i, ln in enumerate(lines, start=1):
            for key in ("speaker", "text", "direction", "label", "numbers", "hotwords"):
                if key not in ln:
                    errors.append(f"{sid} line {i}: missing field {key}")
            sp = ln.get("speaker", "")
            if sp not in roles:
                errors.append(f"{sid} line {i}: speaker '{sp}' not in roles")
            used.add(sp)
            text = ln.get("text", "")
            if not text.strip():
                errors.append(f"{sid} line {i}: empty text")
            if re.search(r"[0-9０-９]", text):
                errors.append(f"{sid} line {i}: Arabic digits in text -> write numbers in Chinese characters: {text}")
            if re.search(r"[（(].*[)）]", text):
                errors.append(f"{sid} line {i}: brackets in text; move stage directions to 'direction': {text}")
            if ln.get("label") not in LABELS:
                errors.append(f"{sid} line {i}: bad label '{ln.get('label')}'")
            blob = text + ln.get("direction", "")
            for w in BANNED:
                if w in blob:
                    errors.append(f"{sid} line {i}: banned term '{w}': {text}")
            total += effective_len(text)
        unused = set(roles) - used
        if unused:
            errors.append(f"{sid}: roles never speak: {sorted(unused)}")
        if not (750 <= total <= 1100):
            errors.append(f"{sid}: effective chars {total} outside 750..1100 (target 800..1000)")
        summary.append(f"{sid} {sc.get('title')}: {len(lines)} lines, {total} chars, roles={roles}")
    for s in summary:
        print(s)
    if errors:
        print(f"\n{len(errors)} PROBLEM(S):")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
