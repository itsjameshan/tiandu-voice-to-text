"""AI 合成测评录音（pipeline/tts_dataset.py）的测试：用假的合成函数（按字数出一段纯音），不需要语音合成模型。"""
import numpy as np
import pytest

from pipeline import tts_dataset as T
from pipeline.data import AI_DISCLAIMER, AI_DISCLAIMER_SPEAKER, FLAG_LABELS, get_script, load_scripts


def fake_tts(text, sid, speed):
    """假的合成函数：每个字 0.12 秒的锯齿波"嗡嗡声"（基音随声音编号变，谐波一直到高频，像人声一样），
    采样率 22050（检查重采样），首尾各有 0.2 秒静音（检查去静音）。"""
    from scipy.signal import sawtooth

    sr = 22050
    seconds = max(0.3, 0.12 * len(text) / speed)
    t = np.arange(int(sr * seconds)) / sr
    tone = 0.3 * sawtooth(2 * np.pi * (120 + 3 * sid) * t)
    pad = np.zeros(int(0.2 * sr))
    return np.concatenate([pad, tone, pad]).astype(np.float32), sr


def _voices(script):
    return {role: {"sid": 10 + k, "speed": 1.0} for k, role in enumerate(script["roles"])}


@pytest.mark.parametrize("direction,check", [
    ("（与上一句部分重叠）", lambda fx: fx["overlap"]),
    ("（打断）", lambda fx: fx["overlap"]),
    ("（停顿约三秒）", lambda fx: fx["pause"] == 3),
    ("（车内安静约十秒，只有发动机声）", lambda fx: fx["pause"] == 10),
    ("（约十分钟后）", lambda fx: fx["pause"] == T.SCENE_JUMP_PAUSE),
    ("（用车载扩音器）", lambda fx: fx["megaphone"]),
    ("（放下扩音器，对司机说）", lambda fx: not fx["megaphone"]),
    ("（电话外放）", lambda fx: fx["phone"]),
    ("（离手机较远）", lambda fx: fx["far"] and not fx["near"]),
    ("（靠近手机小声说）", lambda fx: fx["near"] and not fx["far"] and fx["gain_db"] == -5),
    ("（提高音量）", lambda fx: fx["gain_db"] == 4),
    ("（语速较快）", lambda fx: fx["speed"] == 1.15),
    ("（隧道内，用车载扩音器，有回声）", lambda fx: fx["echo"] and fx["megaphone"]),
    ("（车载导航提示音响起，不是台词）", lambda fx: fx["events"] == ["beep"]),
    ("", lambda fx: not any([fx["overlap"], fx["far"], fx["megaphone"], fx["phone"]]) and fx["pause"] is None),
])
def test_line_effects(direction, check):
    assert check(T.line_effects(direction))


def test_every_direction_in_the_scripts_parses():
    """24 个剧本里所有舞台说明都能解析，停顿不超过 10 秒，语速和音量在合理范围内。"""
    for script in load_scripts():
        for line in script["lines"]:
            fx = T.line_effects(line.get("direction", ""))
            assert fx["pause"] is None or 0 < fx["pause"] <= T.MAX_PAUSE
            assert 0.85 <= fx["speed"] <= 1.2 and -10 <= fx["gain_db"] <= 8


def test_clean_tts_text():
    assert T.clean_tts_text("今晚有一场演出，叫《云上花间》。") == "今晚有一场演出，叫云上花间。"
    assert T.clean_tts_text("所以只能……") == "所以只能"
    assert T.clean_tts_text("不买的话——行程就不好安排了") == "不买的话，行程就不好安排了"
    # 冒号换成逗号（中英混读时模型会把冒号念成英文 colon）；连着的大写字母逐个念（中间加逗号）
    assert T.clean_tts_text("我念啊：DD 零零零零") == "我念啊，D, D 零零零零"
    assert T.clean_tts_text("HT 杠 二零二六") == "H, T 杠 二零二六"
    assert T.clean_tts_text("想换A区吧") == "想换A区吧"


def test_has_latin():
    """有英文字母的台词要用中英混读的模型念（只用中文的会把字母吞掉）。"""
    assert T.has_latin("行话叫A货") and T.has_latin("H, T 杠")
    assert not T.has_latin("熟普每饼三百五十七克") and not T.has_latin("零零零零，三七二一")


def test_plan_timeline_gaps_overlaps_and_pauses():
    rng = np.random.default_rng(0)
    normal, overlap, pause = T.line_effects(""), T.line_effects("（打断）"), T.line_effects("（停顿约两秒）")
    starts = T.plan_timeline([2.0, 1.0, 1.0, 1.0], [normal, normal, overlap, pause], rng, start=0.5)
    assert starts[0] == 0.5
    assert 2.5 + 0.25 <= starts[1] <= 2.5 + 0.8          # 正常停顿
    assert starts[1] + 0.3 <= starts[2] < starts[1] + 1.0  # 打断：上一句还没说完就开始
    assert starts[3] == pytest.approx(starts[2] + 1.0 + 2.0, abs=1e-3)  # 停顿约两秒


def test_clips_from_lines_merges_consecutive_same_label():
    flag = FLAG_LABELS[0]
    timeline = [
        {"index": 0, "start": 0.0, "end": 1.0, "label": "正常讲解"},
        {"index": 1, "start": 1.5, "end": 2.5, "label": flag},
        {"index": 2, "start": 2.8, "end": 4.0, "label": flag},
        {"index": 3, "start": 4.5, "end": 5.0, "label": "其他"},
        {"index": 4, "start": 5.5, "end": 6.0, "label": flag},
    ]
    assert T.clips_from_lines(timeline) == [(1.5, 4.0, flag), (5.5, 6.0, flag)]


@pytest.mark.parametrize("condition", ["Q", "N", "F"])
def test_render_recording(condition):
    script = get_script("G3-S1")
    pool = [fake_tts("嘈杂的背景人声", sid, 1.0)[0] for sid in (40, 70, 90)]
    audio, truth = T.render_recording(script, condition, _voices(script), {"sid": 5}, fake_tts, seed=1,
                                      babble_pool=pool)
    assert audio.dtype == np.float32 and np.isfinite(audio).all() and np.max(np.abs(audio)) <= 0.9
    assert truth["duration"] == pytest.approx(len(audio) / T.SR, abs=0.01)
    # 开头是声明，参考文本第一行也是声明
    assert truth["turns"][0][2] == AI_DISCLAIMER_SPEAKER and truth["turns"][0][0] < 1.0
    assert truth["reference"].splitlines()[0] == AI_DISCLAIMER
    assert len(truth["turns"]) == len(script["lines"]) + 1
    assert all(0 <= s < e <= truth["duration"] for s, e, _ in truth["turns"])
    assert {speaker for _, _, speaker in truth["turns"][1:]} == set(script["roles"])
    assert all(label in FLAG_LABELS for _, _, label in truth["clips"])
    assert truth["f_variant"] in ("pocket", "far") if condition == "F" else truth["f_variant"] == ""


def test_conditions_sound_different():
    """口袋录音的高频比安静录音少得多；嘈杂录音一开始就有别人说话的声音，台词停顿处也不安静。"""
    from scipy.signal import welch

    pocket_script = get_script("G1-S2")  # 组号 + 剧本号是单数 → 口袋
    voices = _voices(pocket_script)
    q, _ = T.render_recording(pocket_script, "Q", voices, {"sid": 5}, fake_tts, seed=2)
    f, truth = T.render_recording(pocket_script, "F", voices, {"sid": 5}, fake_tts, seed=2)
    assert truth["f_variant"] == "pocket"

    def high_share(x):
        freqs, power = welch(x, fs=T.SR, nperseg=2048)
        return power[freqs > 4000].sum() / power.sum()

    assert high_share(f) < high_share(q) * 0.5

    script = get_script("G3-S1")  # 酒店大堂，没有发动机声
    pool = [fake_tts("背景人声嘈杂", sid, 1.0)[0] for sid in (40, 70)]
    n, truth_n = T.render_recording(script, "N", _voices(script), {"sid": 5}, fake_tts, seed=2, babble_pool=pool)
    quiet, truth_q = T.render_recording(script, "Q", _voices(script), {"sid": 5}, fake_tts, seed=2)

    def before_first_line(audio, truth):  # 声明之后、第一句台词之前（对话开头留的 0.5 秒）
        return audio[int((truth["turns"][0][1] + 1.05) * T.SR): int((truth["turns"][1][0] - 0.05) * T.SR)]

    def floor(audio, truth):  # 对话部分最安静的 10% 小段的音量
        part, hop = audio[int(truth["turns"][1][0] * T.SR):], int(0.1 * T.SR)
        blocks = sorted(T.rms(part[i:i + hop]) for i in range(0, len(part) - hop, hop))
        return blocks[len(blocks) // 10]

    assert T.rms(n[: int(0.25 * T.SR)]) == 0  # 声明之前是纯静音
    assert T.rms(before_first_line(n, truth_n)) > 10 * T.rms(before_first_line(quiet, truth_q))
    assert floor(n, truth_n) > 10 * floor(quiet, truth_q)


def test_same_seed_same_audio():
    script = get_script("G4-S1")
    a, _ = T.render_recording(script, "Q", _voices(script), {"sid": 5}, fake_tts, seed=7)
    b, _ = T.render_recording(script, "Q", _voices(script), {"sid": 5}, fake_tts, seed=7)
    c, _ = T.render_recording(script, "Q", _voices(script), {"sid": 5}, fake_tts, seed=8)
    assert np.array_equal(a, b) and not (len(a) == len(c) and np.array_equal(a, c))


def test_cache_synthesizes_each_line_once():
    script = get_script("G2-S1")
    calls = []

    def counting_tts(text, sid, speed):
        calls.append(text)
        return fake_tts(text, sid, speed)

    cache = {}
    for condition in ("Q", "F"):
        T.render_recording(script, condition, _voices(script), {"sid": 5}, counting_tts, seed=3, cache=cache)
    assert len(calls) == len({(T.clean_tts_text(line["text"]), line["speaker"]) for line in script["lines"]}) + 1


def test_make_voice_plan_rules():
    rng = np.random.default_rng(0)
    profiles = {}
    for sid in list(T.FEMALE_SIDS) + list(T.MALE_SIDS):
        emb = rng.standard_normal(16)
        profiles[sid] = {"f0": 220.0 if sid < 58 else 120.0, "emb": (emb / np.linalg.norm(emb)).tolist()}
    profiles[57]["f0"] = 330.0
    scripts = [get_script("G3-S1"), get_script("G7-S3"), get_script("G1-S1")]
    rules = {"G3-S1": {"游客乙": {"similar_to": "游客甲"}},
             "G7-S3": {"游客乙": {"gender": "女", "high": True, "speed": 1.1, "note": "孩子"}}}
    plan = T.make_voice_plan(scripts, profiles, rules)
    by = {(p["script_id"], p["role"]): p for p in plan}
    assert len(plan) == sum(len(s["roles"]) for s in scripts)
    for script in scripts:  # 同一个剧本里每个角色的声音不一样
        sids = [by[(script["id"], role)]["sid"] for role in script["roles"]]
        assert len(set(sids)) == len(sids)
    a, b = by[("G3-S1", "游客甲")], by[("G3-S1", "游客乙")]
    assert a["gender"] == b["gender"]  # 声线相近：同性别、声纹最像的那个
    sims = [float(np.dot(profiles[a["sid"]]["emb"], profiles[s]["emb"]))
            for s in (T.FEMALE_SIDS if a["gender"] == "女" else T.MALE_SIDS) if s != a["sid"]]
    assert float(np.dot(profiles[a["sid"]]["emb"], profiles[b["sid"]]["emb"])) == pytest.approx(max(sims))
    child = by[("G7-S3", "游客乙")]
    assert child["sid"] == 57 and child["speed"] == 1.1 and child["note"] == "孩子"
    assert all(p["gender"] == "男" for p in plan if p["role"] == "司机")


def test_voice_rules_refer_to_real_roles():
    """VOICE_RULES 里写的剧本和角色都存在；similar_to 指向排在前面的角色（先挑好它的声音）。"""
    for script_id, roles in T.VOICE_RULES.items():
        script = get_script(script_id)
        for role, rule in roles.items():
            assert role in script["roles"], (script_id, role)
            assert set(rule) <= {"gender", "similar_to", "speed", "high", "note"}, (script_id, role)
            assert rule.get("gender", "女") in ("女", "男")
            if "similar_to" in rule:
                assert script["roles"].index(rule["similar_to"]) < script["roles"].index(role)


def _profiles(seed=0):
    rng = np.random.default_rng(seed)
    profiles = {}
    for sid in list(T.FEMALE_SIDS) + list(T.MALE_SIDS):
        emb = rng.standard_normal(16)
        profiles[sid] = {"f0": 220.0 if sid < 58 else 120.0, "emb": (emb / np.linalg.norm(emb)).tolist()}
    return profiles


def test_pick_disclaimer_voice():
    """声明用一个女声，和别的声音都不太像（平均声纹相似度最低）。"""
    profiles = _profiles()
    odd = np.zeros(16)
    odd[0] = 1.0
    for sid in profiles:  # 让 20 号和别的都不像：别的声音在第一维上都是负的
        if sid != 20:
            emb = np.array(profiles[sid]["emb"])
            emb[0] = -abs(emb[0]) - 1.0
            profiles[sid]["emb"] = (emb / np.linalg.norm(emb)).tolist()
    profiles[20]["emb"] = odd.tolist()
    assert T.pick_disclaimer_voice(profiles) == 20


def test_voice_plan_file_round_trip(tmp_path):
    scripts = [get_script("G2-S3"), get_script("G5-S1")]
    plan = T.make_voice_plan(scripts, _profiles(), T.VOICE_RULES)
    path = tmp_path / "plan.csv"
    T.write_voice_plan(path, plan, disclaimer_sid=9)
    voices, disclaimer = T.load_voice_plan(path)
    assert disclaimer == {"sid": 9, "speed": 1.0}
    assert set(voices) == {"G2-S3", "G5-S1"}
    for item in plan:
        assert voices[item["script_id"]][item["role"]] == {"sid": item["sid"], "speed": item["speed"]}
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 带 BOM，Excel 直接打开不乱码


def test_load_voice_plan_reports_problems(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text("script_id,role,sid,gender,speed,note\nG1-S1,导游,5,女,1.0,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="声明"):
        T.load_voice_plan(path)
    path.write_text("script_id,role,sid,gender,speed,note\n*,声明,5,女,1.0,\nG1-S1,导游,abc,女,1.0,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 3 行"):
        T.load_voice_plan(path)


def test_committed_voice_plan():
    """data/tts_voice_plan.csv：每个剧本的每个角色都有声音；同一剧本里不重复；性别和编号范围一致；
    声明的声音不给任何角色用；VOICE_RULES 里的要求都做到了。"""
    voices, disclaimer = T.load_voice_plan()
    rows = {(r["script_id"], r["role"]): r for r in T.voice_plan_rows()}
    for script in load_scripts():
        sids = [voices[script["id"]][role]["sid"] for role in script["roles"]]
        assert len(set(sids)) == len(sids), script["id"]
        assert disclaimer["sid"] not in sids
        for role in script["roles"]:
            row = rows[(script["id"], role)]
            assert int(row["sid"]) in (T.FEMALE_SIDS if row["gender"] == "女" else T.MALE_SIDS)
            rule = T.VOICE_RULES.get(script["id"], {}).get(role, {})
            if "gender" in rule:
                assert row["gender"] == rule["gender"], (script["id"], role)
            if "similar_to" in rule:
                assert row["gender"] == rows[(script["id"], rule["similar_to"])]["gender"]
            assert float(row["speed"]) == rule.get("speed", 1.0)
    assert set(voices) == {s["id"] for s in load_scripts()}


def test_voice_plan_balances_genders():
    """没有特别要求的角色补少的那一边：4 个以上角色的剧本里男女声都有。"""
    plan = T.make_voice_plan(load_scripts(), _profiles(), T.VOICE_RULES)
    for script in load_scripts():
        genders = [p["gender"] for p in plan if p["script_id"] == script["id"]]
        if script["id"] not in ("G2-S3",):  # 母女两人 + 店员、经理：规则只定了两个
            assert {"女", "男"} <= set(genders), (script["id"], genders)
