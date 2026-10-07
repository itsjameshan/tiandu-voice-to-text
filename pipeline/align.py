"""字错率的拆分与逐段对照（"数据校对"标签页和测评工具用）。

字错率 =（错字 + 漏字 + 多字）÷ 参考文本字数。
    例：参考"雾隐行舟旅行社"，识别成"雾影行走旅行社"：错 2 字，2 ÷ 7 ≈ 28.6%。

怎么数错字、漏字、多字？
    找出"把参考文本改成识别结果，最少要改几步"（编辑距离，也叫 Levenshtein 距离）。
    每一步只有三种：
        换一个字（replace）  → 错字 sub
        删掉一个字（delete） → 漏字 dele（参考文本里有，识别结果里没有）
        加一个字（insert）   → 多字 ins （参考文本里没有，识别结果里多出来）
    算字错率前，两边都先用 pipeline.text_norm.normalize_for_cer 归一：
    去掉标点和空格、全角转半角、字母大写——写法不同不算错。

两个函数：
    cer_details(参考文本, 识别结果)       一段文字的字错率，并拆成错字、漏字、多字
    align_segments(各段识别文字, 参考文本)  把一段录音各段的识别结果和整篇参考文本逐字对齐，
                                          找出每段对应参考文本里的哪几句，标出不一致的段

基线做法：
    用 rapidfuzz 算编辑距离（rapidfuzz.distance.Levenshtein）。
    逐段对照：把各段识别文字连起来，和整篇参考文本逐字对齐；
    每段识别文字对上的那一截参考文本，就是这一段的"参考文本"；
    两段交界处识别漏掉的字：中间有标点（或换行）就在第一个标点处分开，
    标点前的归前一段（前一句的句尾），标点后的归后一段（后一句的开头）；没有标点就都归前一段。
可改进方向：
    - 交界处漏掉的字跨过好几个标点时，可以优先在句号、问号、换行处分，而不是第一个标点；
    - 同音字（"昆名"和"昆明"）可以另外统计，看看有多少错误是热词纠错能改的。
测评指标：
    cer_details 的结果就是测评用的字错率（tools/evaluate.py cer）；
    tests/test_align.py 用视频里的例子检查错字、漏字、多字的个数。

注意：本文件只依赖 rapidfuzz 和标准库，不加载识别模型。
"""
from pipeline.text_norm import normalize_for_cer


def cer_details(reference: str, hypothesis: str) -> dict:
    """算一段文字的字错率，并拆成错字、漏字、多字。

    reference：参考文本（实际说了什么）；hypothesis：识别结果。
    返回 {"cer": 字错率, "sub": 错字数, "dele": 漏字数, "ins": 多字数, "n_ref": 参考文本字数}。
    参考文本归一后是空的：识别结果也空，字错率算 0.0；识别结果不空，算 1.0。
    """
    from rapidfuzz.distance import Levenshtein

    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)

    sub = dele = ins = 0
    # editops 列出"把 ref 改成 hyp"的每一步
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "replace":
            sub += 1
        elif op.tag == "delete":
            dele += 1
        elif op.tag == "insert":
            ins += 1

    n_ref = len(ref)
    if n_ref == 0:
        cer = 0.0 if not hyp else 1.0
    else:
        cer = (sub + dele + ins) / n_ref
    return {"cer": cer, "sub": sub, "dele": dele, "ins": ins, "n_ref": n_ref}


def _normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """逐字归一，同时记下归一后每个字来自原文的第几个字。

    例："今天，去。" → ("今天去", [0, 1, 3])。这样对齐以后能从原文里截出带标点的那一句。
    """
    chars = []
    positions = []
    for i, ch in enumerate(text):
        for c in normalize_for_cer(ch):  # 极少数字符归一后变成两个字（如"㎏"→"KG"）
            chars.append(c)
            positions.append(i)
    return "".join(chars), positions


def _align_chars(ref: str, hyp: str) -> list[tuple[int, int]]:
    """逐字对齐 ref 和 hyp：识别结果的第 j 个字，对上参考文本的 [开始, 结束) 这一截。

    对的字、错字各对上一个字（结束 = 开始 + 1）；多出来的字对上空的一截（结束 = 开始）。
    两个相邻的识别字对上的位置之间如果有空档，空档里就是识别漏掉的字。
    """
    from rapidfuzz.distance import Levenshtein

    spans = [(0, 0)] * len(hyp)
    # opcodes 把对齐结果分成一块一块：
    #   equal（相同）、replace（错字）、delete（漏字）、insert（多字），
    #   每块给出参考文本里的范围 [src_start, src_end) 和识别结果里的范围 [dest_start, dest_end)
    for block in Levenshtein.opcodes(ref, hyp):
        for j in range(block.dest_start, block.dest_end):
            if block.tag in ("equal", "replace"):  # 一个字对一个字
                i = block.src_start + (j - block.dest_start)
                spans[j] = (i, i + 1)
            else:  # insert：多出来的字，挂在参考文本的这个位置上
                spans[j] = (block.src_start, block.src_start)
        # delete 块在识别结果里没有字，不用记
    return spans


def _choose_cut(lo: int, hi: int, positions: list[int]) -> int:
    """两段交界处，参考文本第 lo 到 hi 个字（不含 hi）是识别漏掉的字，决定从哪里分开。

    从 lo 往后找第一个"前面有标点"的位置（原文里它和前一个字之间隔着标点、空白或换行）；
    找不到就在 hi 处分开（漏掉的字都归前一段）。
    """
    for c in range(lo, hi + 1):
        if 0 < c < len(positions) and positions[c] - positions[c - 1] > 1:
            return c
    return hi


def align_segments(segment_texts: list[str], reference: str) -> list[dict]:
    """把一段录音各段的识别文字，和这段录音的整篇参考文本对照。

    segment_texts：按时间顺序排好的每段识别文字（测评模式的 text_raw）；
    reference：整篇参考文本（可以有标点、换行）。
    返回每段一个字典：
        hyp   这一段的识别文字（原样）
        ref   参考文本里对应的那一截（保留标点，两头去掉空白和换行）
        diff  归一后两者是否不同（True 表示这一段要重点听）
        cer   这一段的字错率
    """
    if not segment_texts:
        return []

    ref_norm, ref_positions = _normalize_with_positions(reference)
    hyp_parts = [normalize_for_cer(t) for t in segment_texts]
    hyp_norm = "".join(hyp_parts)
    spans = _align_chars(ref_norm, hyp_norm)

    # 每两段的交界处，在归一后的参考文本里找一个分开的位置
    cuts = [0]
    b = 0  # 第 k 段第一个字在 hyp_norm 里的位置
    for part in hyp_parts[:-1]:
        b += len(part)
        lo = spans[b - 1][1] if b > 0 else 0                    # 前一个识别字对上的位置之后
        hi = spans[b][0] if b < len(hyp_norm) else len(ref_norm)  # 后一个识别字对上的位置
        cuts.append(_choose_cut(lo, hi, ref_positions))
    cuts.append(len(ref_norm))

    def original_index(norm_index: int) -> int:
        """归一后参考文本的第 n 个字，在原文里是第几个字；n 等于总字数时对应原文末尾。"""
        if norm_index >= len(ref_positions):
            return len(reference)
        return ref_positions[norm_index]

    rows = []
    for k, text in enumerate(segment_texts):
        # 原文里从这段第一个字截到下一段第一个字之前，所以句尾的标点跟着这一段；
        # 第 1 段从原文第 0 个字开始，开头的引号、空白也算进来
        begin = 0 if k == 0 else original_index(cuts[k])
        end = original_index(cuts[k + 1])
        ref_piece = reference[begin:end].strip()
        details = cer_details(ref_piece, text)
        rows.append({
            "hyp": text,
            "ref": ref_piece,
            "diff": normalize_for_cer(ref_piece) != normalize_for_cer(text),
            "cer": details["cer"],
        })
    return rows
