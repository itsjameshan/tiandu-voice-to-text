"""测评用的几个纯计算函数（tools/evaluate.py 的 speakers、classify、clips 子命令和分类训练脚本都用它）。

这里只做"拿标准答案和工具结果比一比、算出一个数"，不读文件、不跑模型。
五个函数：
    speaker_error_rate(标准答案说话时间, 结果说话时间)   说话人标错的时长比例（第 3 组）
    confusion_matrix(标准答案类别, 预测类别, 类别表)     混淆矩阵（第 6、7 组）
    per_class_pr(标准答案类别, 预测类别, 类别表)         各类准确率、召回率（第 6、7 组）
    false_positive_rate(标准答案类别, 预测类别)          误报率（第 6、7、8 组）
    clip_boundary_error(标注片段, 导出片段)              片段起止误差（第 8 组）

说话时间（turn）和片段（clip）可以写成元组，也可以直接用统一中间格式的段落字典：
    说话时间：(开始秒, 结束秒, 说话人) 或 {"start": …, "end": …, "speaker": …}
    片段：    (开始秒, 结束秒, …)       或 {"start": …, "end": …}（第三项以后不看）

注意：分类测评里的"类别"都用类别名（如"威胁消费"），不是显示的标签（如"疑似·消费施压"）。

基线做法：
    只用标准库和 numpy；说话人对应要用匈牙利算法（scipy.optimize.linear_sum_assignment），
    scipy 只在 speaker_error_rate 里面导入，这样只装了 TensorFlow、numpy 的机房电脑
    也能用这个模块跑分类训练脚本。
可改进方向：
    - 说话人：现在只看两边都有人说话的时间；还可以另算"漏掉的说话时间""多出来的说话时间"，
      合起来就是论文里常见的说话人分离错误率（DER）；交界处前后各留 0.25 秒不计分，可以减少
      人工标注起止不准带来的影响；
    - 分类：再算每类的 F1（准确率和召回率的调和平均），或按组分别算误报率；
    - 片段：现在按"重叠最多的先配对"（贪心），也可以像说话人那样用匈牙利算法求总重叠最大的配对。
测评指标：
    本文件本身就是测评指标的定义；每个函数的说明里都写了公式。手算的小例子见 tests/test_metrics.py。
"""
import numpy as np

# 标准答案为这个类别的句子，被分成任一疑似类别就算误报
NORMAL_LABEL = "正常讲解"


# ---------------- 小工具：把元组或段落字典统一成数字 ----------------

def _turn(item) -> tuple[float, float, str]:
    """一条说话时间 → (开始秒, 结束秒, 说话人)。支持元组/列表和段落字典。"""
    if isinstance(item, dict):
        return float(item["start"]), float(item["end"]), str(item["speaker"])
    return float(item[0]), float(item[1]), str(item[2])


def _span(item) -> tuple[float, float]:
    """一个片段 → (开始秒, 结束秒)。支持元组/列表和段落字典，第三项以后不看。"""
    if isinstance(item, dict):
        return float(item["start"]), float(item["end"])
    return float(item[0]), float(item[1])


# ---------------- 说话人标错的时长比例 ----------------

def _activity(turns: list[tuple[float, float, str]], step: float, n_frames: int) -> tuple[list[str], np.ndarray]:
    """把说话时间画到时间格子上。

    返回 (说话人名字列表, 表格)：表格第 i 行第 k 列为 True，表示第 i 个人在第 k 格（k×step 秒起）在说话。
    """
    names = []
    for _, _, name in turns:
        if name not in names:
            names.append(name)
    active = np.zeros((len(names), n_frames), dtype=bool)
    for start, end, name in turns:
        first = max(0, int(round(start / step)))  # round 防止 1.5 / 0.01 = 149.99999 这类小数误差
        last = min(n_frames, int(round(end / step)))
        if last > first:
            active[names.index(name), first:last] = True
    return names, active


def speaker_error_rate(ref_turns, hyp_turns, step: float = 0.01) -> float:
    """说话人标错的时长比例（0 表示全对，越小越好）。

    ref_turns：人工标注的"谁在什么时间说话"，每条 (开始秒, 结束秒, 说话人)，如 (0.0, 5.2, "导游")；
    hyp_turns：工具的结果，格式相同，说话人一般是"说话人1""说话人2"；也可以直接传段落列表。
    step：时间格子的宽度，默认 0.01 秒（10 毫秒）。

    怎么算：
        1. 把时间切成每 step 秒一格，记下每个人在哪些格里说话；
        2. 两边的人两两配对，数"同一格里两人都在说话"的格数，得到一张重叠表；
           用匈牙利算法（scipy.optimize.linear_sum_assignment）找一一对应，让对上的总格数最多。
           例如结果的"说话人2"对应标准答案的"导游"——所以只是名字互换不算错；
        3. 只看两边都有人说话的格子。每一格：
               要比对的人数 = min(标准答案里这一格说话的人数, 结果里这一格说话的人数)
               对上的人数   = 按第 2 步的对应，两边都在说话的对数
               标错的人数   = 要比对的人数 − 对上的人数
           没有人同时说话时，每格要比对的人数就是 1，对上了标错 0，没对上标错 1。
        4. 说话人标错的时长比例 = 所有格子标错的人数之和 ÷ 所有格子要比对的人数之和

    只有一边有人说话的时间（漏掉的、多出来的说话时间）不算在这个指标里。
    两边没有同时说话的时间（例如结果一段也没有）时没有可比的，返回 0.0，
    这时请同时看比对了多少秒，不要只看这个 0。
    """
    ref = [_turn(item) for item in ref_turns]
    hyp = [_turn(item) for item in hyp_turns]
    if not ref or not hyp:
        return 0.0

    # 格子总数：覆盖到两边最晚的结束时间
    last_end = max(end for _, end, _ in ref + hyp)
    n_frames = int(round(last_end / step))
    if n_frames <= 0:
        return 0.0
    _, ref_active = _activity(ref, step, n_frames)
    _, hyp_active = _activity(hyp, step, n_frames)

    # 每一格要比对的人数 = min(标准答案说话人数, 结果说话人数)；只有一边有人说话时为 0
    scored = np.minimum(ref_active.sum(axis=0), hyp_active.sum(axis=0)).sum()
    if scored == 0:
        return 0.0

    # 重叠表：overlap[i][j] = 标准答案第 i 个人和结果第 j 个人同时在说话的格数
    overlap = ref_active.astype(np.int64) @ hyp_active.T.astype(np.int64)

    # 匈牙利算法：找让对上的总格数最多的一一对应（scipy 在这里才导入）
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(overlap, maximize=True)
    correct = overlap[rows, cols].sum()  # 按这个对应，所有格子里对上的人数之和

    return float((scored - correct) / scored)


# ---------------- 话术分类：混淆矩阵、各类准确率和召回率、误报率 ----------------

def _check_same_length(gold, pred) -> None:
    if len(gold) != len(pred):
        raise ValueError(f"标准答案有 {len(gold)} 句，预测有 {len(pred)} 句，数量必须一样（一句对一句）")


def confusion_matrix(gold, pred, labels) -> list[list[int]]:
    """混淆矩阵：matrix[i][j] = 标准答案是 labels[i]、被预测成 labels[j] 的句子数。

    行是标准答案，列是预测；对角线（i == j）上的是分对的句数。
    gold、pred 是一样长的类别名列表（一句对一句），labels 是类别表（决定行列的顺序）。
    长度不一样、或者出现类别表里没有的类别时抛 ValueError。
    """
    _check_same_length(gold, pred)
    labels = list(labels)
    index = {name: i for i, name in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        for name in (g, p):
            if name not in index:
                raise ValueError(f"类别 {name!r} 不在类别表里，只能是：{'、'.join(labels)}")
        matrix[index[g]][index[p]] += 1
    return matrix


def per_class_pr(gold, pred, labels) -> dict[str, dict]:
    """各类的准确率和召回率，按类别表的顺序返回 {类别: {...}}。

    每类的字典：
        tp        分对的句数（标准答案是这一类、也预测成这一类）
        n_pred    被预测成这一类的句数（混淆矩阵这一列的和）
        n_gold    标准答案是这一类的句数（混淆矩阵这一行的和）
        precision 准确率 = tp ÷ n_pred：工具说是这一类的句子里，真是这一类的比例
        recall    召回率 = tp ÷ n_gold：真是这一类的句子里，被工具找出来的比例
    分母为 0 时（一句也没预测成这一类，或标准答案里没有这一类）对应的值按 0.0 算。
    """
    labels = list(labels)
    matrix = confusion_matrix(gold, pred, labels)
    result = {}
    for i, name in enumerate(labels):
        tp = matrix[i][i]
        n_gold = sum(matrix[i])
        n_pred = sum(row[i] for row in matrix)
        result[name] = {
            "precision": tp / n_pred if n_pred else 0.0,
            "recall": tp / n_gold if n_gold else 0.0,
            "tp": tp,
            "n_pred": n_pred,
            "n_gold": n_gold,
        }
    return result


def false_positive_rate(gold, pred) -> float:
    """误报率 = 标准答案为"正常讲解"的句子中，被预测成任一疑似类别的句数 ÷ 标准答案为"正常讲解"的句数。

    疑似类别就是 data/labels.json 里 flag=true 的 5 个类别名（pipeline.data.FLAG_LABELS：
    购物安排、费用、行程变更、服务态度、威胁消费）。
    正常讲解被分成"其他"不算误报（其他不会被标出来）；
    标准答案是费用等疑似类别的句子分错了，也不算误报（那是准确率、召回率管的事）。
    标准答案里没有正常讲解时没有可比的，返回 0.0。
    """
    from pipeline.data import FLAG_LABELS, LABEL_NAMES  # 放在函数里：本文件顶层只导入标准库和 numpy

    _check_same_length(gold, pred)
    unknown = sorted({x for x in list(gold) + list(pred) if x not in LABEL_NAMES})
    if unknown:
        # 常见错误：传进来的是显示用的标签（如"疑似·费用"），而不是类别名（如"费用"）
        raise ValueError(f"误报率要用类别名（{'、'.join(LABEL_NAMES)}），不认识：{'、'.join(unknown)}")
    n_normal = 0
    n_flagged = 0
    for g, p in zip(gold, pred):
        if g == NORMAL_LABEL:
            n_normal += 1
            if p in FLAG_LABELS:
                n_flagged += 1
    return n_flagged / n_normal if n_normal else 0.0


# ---------------- 片段起止误差 ----------------

def clip_boundary_error(ref_clips, hyp_clips) -> dict:
    """片段起止误差：人工标注的疑似片段和工具导出的片段，开始、结束各差多少秒。

    ref_clips：人工标注的片段，每个 (开始秒, 结束秒, 类别)；
    hyp_clips：工具的片段，每个 (开始秒, 结束秒, …) 或段落字典。只看起止时间，不看类别。

    怎么配对（按最大重叠）：
        1. 两边的片段两两算重叠秒数 = min(两个结束) − max(两个开始)，大于 0 才算有重叠；
        2. 从重叠最多的一对开始配，已经配过的片段不再参加，直到没有能配的。
           只是挨着（重叠 0 秒）不算配上。
    返回的字典：
        start_mae      起点平均绝对误差（秒）= 每对 |结果开始 − 标注开始| 的平均
        end_mae        终点平均绝对误差（秒）= 每对 |结果结束 − 标注结束| 的平均
                       一对也没配上时，两个都是 None
        n_matched      配上的对数
        n_ref、n_hyp   标注片段数、结果片段数
        unmatched_ref  标注里没配上的片段数（工具漏掉的）
        unmatched_hyp  结果里没配上的片段数（工具多出来的）
        unmatched      未配对数 = unmatched_ref + unmatched_hyp
    """
    ref = [_span(item) for item in ref_clips]
    hyp = [_span(item) for item in hyp_clips]

    # 1. 所有有重叠的 (重叠秒数, 标注序号, 结果序号)
    candidates = []
    for i, (ref_start, ref_end) in enumerate(ref):
        for j, (hyp_start, hyp_end) in enumerate(hyp):
            overlap = min(ref_end, hyp_end) - max(ref_start, hyp_start)
            if overlap > 0:
                candidates.append((overlap, i, j))
    # 重叠多的排前面；重叠一样时按序号，保证每次结果相同
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    # 2. 从重叠最多的开始配对
    used_ref, used_hyp = set(), set()
    start_errors, end_errors = [], []
    for _, i, j in candidates:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        start_errors.append(abs(hyp[j][0] - ref[i][0]))
        end_errors.append(abs(hyp[j][1] - ref[i][1]))

    n_matched = len(start_errors)
    unmatched_ref = len(ref) - n_matched
    unmatched_hyp = len(hyp) - n_matched
    return {
        "start_mae": sum(start_errors) / n_matched if n_matched else None,
        "end_mae": sum(end_errors) / n_matched if n_matched else None,
        "n_matched": n_matched,
        "n_ref": len(ref),
        "n_hyp": len(hyp),
        "unmatched_ref": unmatched_ref,
        "unmatched_hyp": unmatched_hyp,
        "unmatched": unmatched_ref + unmatched_hyp,
    }
