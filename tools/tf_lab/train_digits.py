"""选做实验 · 第 2 步：用全班的数字录音训练一个小卷积网络，认"零到九"。

用法（在项目文件夹里运行；要用装了 TensorFlow 的 Python，即机房自带的 Python）：
    python tools/tf_lab/train_digits.py                          数据默认在数据池的 digits 文件夹
    python tools/tf_lab/train_digits.py --data D:/digits         指定数据文件夹
    python tools/tf_lab/train_digits.py --epochs 50              多训练几轮

数据：tools/tf_lab/split_digits.py 切好的文件，<data>/<数字>/<学号后四位>_<第几遍>.wav。
只需要 numpy 和 TensorFlow（读 WAV 用 Python 自带的 wave 模块，不需要 soundfile、sherpa-onnx、gradio）。

做了什么（对应教材项目 4：用神经网络认数字命令）：
    1. 读数据：每段录音补零（或截断）到正好 1 秒，算 MFCC（pipeline/features.py 的 mfcc），
       得到 98 帧 × 13 维的"小图片"；文件夹名就是答案（0—9），文件名下划线前面是说话人。
    2. 按人分训练集和测试集：随机留出约 20% 的人（至少 1 人）只做测试，他们的录音训练时一段都不用。
       为什么按人分、不按录音分：同一个人念的"三"每次都很像，如果他的录音一半训练一半测试，
       网络等于"背过答案"，准确率会虚高。按人分，测的才是"没听过的人说的数字能不能认出来"。
    3. 标准化：用训练集算出每一维 MFCC 的平均值和标准差，训练集、测试集都减平均值、除以标准差
       （各维数值大小差很多，统一一下网络更好学）。
    4. 小卷积网络（把 MFCC 当成一张 98×13 的单色图片）：
           Conv2D(16 个 3×3 卷积核, relu) → MaxPooling2D(2×2) → Conv2D(32 个 3×3 卷积核, relu)
           → GlobalAveragePooling2D（每个通道取平均，变成 32 个数）→ Dense(10, softmax)（10 个数字各自的概率）
       用 adam 训练 --epochs 轮（默认 30），每批 16 段。
    5. 在测试集上打印准确率和混淆矩阵（行：实际念的数字，列：网络认成的数字；对角线上是认对的）。

基线做法：
    上面的 5 步；固定随机种子，同一台电脑上重复运行结果基本一样。
可改进方向：
    - 网络加一层卷积、加 Dropout，或者把 GlobalAveragePooling2D 换成 Flatten，看准确率怎么变；
    - 数据增强：给训练录音加一点噪声、前后挪一点位置、音量调大调小；
    - 把 MFCC 换成 40 维的梅尔能量（不做最后的 DCT），比较哪种输入更好；
    - 看混淆矩阵里哪两个数字最容易认混（比如韵母相同的"一 yī"和"七 qī"、"六 liù"和"九 jiǔ"），想想为什么。
测评指标：
    测试集（没参加训练的人）的准确率；混淆矩阵。人数越多结果越可信；只有三四个人时，结果起伏会很大。

隐私：录音只放在本机，不上传网络、不进代码仓库；文件名只用学号后四位。
"""
import argparse
import random
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 先把项目文件夹加进 sys.path 才能导入 pipeline；这两个模块只用 numpy，不会带进 sherpa_onnx、gradio
from pipeline.audio import SR  # noqa: E402
from pipeline.features import mfcc  # noqa: E402

DIGIT_NAMES = "零一二三四五六七八九"
NUM_CLASSES = 10
CLIP_SAMPLES = SR  # 每段录音补零或截断到 1 秒（16000 个采样点）→ MFCC 98 帧 × 13 维
TEST_RATIO = 0.2  # 留出约 20% 的人做测试
MIN_SPEAKERS = 3  # 至少 3 个人：留 1 人测试，还剩至少 2 人训练
DEFAULT_EPOCHS = 30
BATCH_SIZE = 16
SEED = 0  # 随机种子：固定下来，重复运行结果基本一样


def default_data_dir() -> Path:
    """数据池里的 digits 文件夹（config.yaml 的 paths.data_pool）。

    机房自带的 Python 可能没装 PyYAML，读不了 config.yaml，这时用项目文件夹里的 data_pool/digits。
    """
    try:
        from pipeline.config import load_config

        return Path(load_config()["paths"]["data_pool"]) / "digits"
    except ImportError:
        return ROOT / "data_pool" / "digits"


def read_wav16(path: Path) -> np.ndarray:
    """用 Python 自带的 wave 模块读 16000 Hz、单声道、16 位的 WAV，返回 -1 到 1 之间的 float32 数组。"""
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SR or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("不是 16000 Hz 单声道 16 位的 WAV，请用 split_digits.py 重新切")
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def fix_length(samples: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """补零或截断到正好 length 个采样点：短了在后面补 0，长了只取前面。"""
    if len(samples) >= length:
        return samples[:length]
    return np.pad(samples, (0, length - len(samples)))


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """读 <data_dir>/<数字>/<说话人>_<第几遍>.wav，返回 (MFCC [段数, 98, 13], 数字 [段数], 说话人列表)。

    读不了的文件打印一句提示后跳过。
    """
    features, labels, speakers = [], [], []
    for digit in range(NUM_CLASSES):
        for path in sorted((data_dir / str(digit)).glob("*.wav")):
            try:
                samples = read_wav16(path)
            except (ValueError, wave.Error, EOFError) as e:
                print(f"跳过 {path}：{e}")
                continue
            features.append(mfcc(fix_length(samples), SR))
            labels.append(digit)
            speakers.append(path.stem.rsplit("_", 1)[0])  # "0123_2" → "0123"
    if not features:
        return np.zeros((0, 98, 13), dtype=np.float32), np.zeros(0, dtype=np.int64), []
    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64), speakers


def pick_test_speakers(speakers, ratio: float = TEST_RATIO, seed: int = SEED) -> list[str]:
    """随机挑出约 ratio 的人做测试（至少 1 人），返回排好序的说话人列表。种子固定，每次挑的一样。"""
    people = sorted(set(speakers))
    count = max(1, round(len(people) * ratio))
    random.Random(seed).shuffle(people)
    return sorted(people[:count])


def build_model(input_shape):
    """小卷积网络：Conv2D 16 → MaxPool → Conv2D 32 → GlobalAveragePooling2D → Dense 10。"""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),  # (98 帧, 13 维, 1 个通道)
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def confusion_matrix(truth, predicted, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """混淆矩阵：第 i 行第 j 列 = 实际是数字 i、被认成数字 j 的段数。"""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[int(t), int(p)] += 1
    return matrix


def print_confusion(matrix: np.ndarray) -> None:
    """打印混淆矩阵：行是实际念的数字，列是网络认成的数字，对角线上是认对的。"""
    print("混淆矩阵（行：实际念的数字；列：认成的数字）：")
    print("      " + "".join(f"{name:>3}" for name in DIGIT_NAMES))  # 汉字占两格，3 + 1 = 4 格宽
    for i, row in enumerate(matrix):
        print(f"  {DIGIT_NAMES[i]}  " + "".join(f"{n:>4}" for n in row))


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # 命令行窗口显示不了的字用 ? 代替，不让程序因此出错

    parser = argparse.ArgumentParser(description="用数字录音训练小卷积网络认'零到九'，按人留出测试。")
    parser.add_argument("--data", help="数据文件夹（默认：数据池里的 digits 文件夹）")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"训练轮数（默认 {DEFAULT_EPOCHS}）")
    args = parser.parse_args(argv)

    # 1. 读数据（先检查数据，再导入 TensorFlow：导入要好几秒）
    data_dir = Path(args.data) if args.data else default_data_dir()
    features, labels, speakers = load_dataset(data_dir)
    people = sorted(set(speakers))
    if len(people) < MIN_SPEAKERS:
        print(f"数据文件夹：{data_dir}")
        print(f"找到 {len(labels)} 段录音、{len(people)} 个人。至少需要 {MIN_SPEAKERS} 个人的数字录音：")
        print("  测试用的人不能出现在训练集里，所以要留 1 个人测试、至少 2 个人训练。")
        print("  请先让更多同学用 tools/tf_lab/split_digits.py 切好自己的录音，存到同一个文件夹。")
        return 1

    try:
        import tensorflow as tf
    except ImportError:
        print("这个 Python 没装 TensorFlow，训练不了。请用机房自带的 Python 运行（它装了 TensorFlow），例如：")
        print("    python tools/tf_lab/train_digits.py --data 数字录音文件夹")
        print("（便携包的 Python 没装 TensorFlow；自己的电脑可以 pip install -r requirements-tf.txt。）")
        return 1
    tf.keras.utils.set_random_seed(SEED)  # 固定权重初始值和打乱顺序，重复运行结果基本一样

    # 2. 按人分训练集和测试集
    test_people = pick_test_speakers(people)
    is_test = np.array([s in test_people for s in speakers])
    x_train, y_train = features[~is_test], labels[~is_test]
    x_test, y_test = features[is_test], labels[is_test]
    print(f"数据：{len(labels)} 段录音、{len(people)} 个人（{data_dir}）")
    print(f"训练：{len(people) - len(test_people)} 人、{len(y_train)} 段；"
          f"测试：{len(test_people)} 人（{'、'.join(test_people)}）、{len(y_test)} 段")

    # 3. 标准化：只用训练集的平均值和标准差（测试集当作"没见过"，不能偷看）
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1)) + 1e-6
    x_train = ((x_train - mean) / std)[..., np.newaxis]  # 加一个"通道"维：[段数, 98, 13, 1]
    x_test = ((x_test - mean) / std)[..., np.newaxis]

    # 4. 建网络、训练（每轮打印一行：loss 越来越小、accuracy 越来越大，说明网络在学）
    model = build_model(x_train.shape[1:])
    model.summary()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=2)

    # 5. 在测试集上看效果
    predicted = np.argmax(model.predict(x_test, verbose=0), axis=1)
    accuracy = float(np.mean(predicted == y_test))
    print()
    print(f"测试集准确率：{accuracy * 100:.1f}%（{int(np.sum(predicted == y_test))}/{len(y_test)} 段认对，"
          f"测试的人没有参加训练）")
    print_confusion(confusion_matrix(y_test, predicted))
    if len(people) < 10:
        print(f"提示：现在只有 {len(people)} 个人的录音，测试只有 {len(test_people)} 个人，准确率起伏会很大；人越多越可信。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
