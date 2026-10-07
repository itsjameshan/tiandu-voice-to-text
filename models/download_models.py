"""下载本工具要用的模型到 models/ 文件夹。

用法（在项目文件夹里运行；便携包里把 python 换成 python\\python.exe）：
    python models/download_models.py                            下载必需模型（已下载的自动跳过）
    python models/download_models.py --list                     只看哪些模型已下载、哪些缺失，不下载
    python models/download_models.py --optional punct itn_fst   另外下载可选模型（名字见 --list）
    python models/download_models.py --models-dir E:/asr/models 下载到别的文件夹

基线做法：
    模型清单在 pipeline/models.py 的 MODEL_SPECS 里。逐个检查：check 列出的文件都在就跳过；
    否则用 Python 自带的 urllib 从 GitHub 下载（失败自动重试 3 次），
    .tar.bz2 压缩包解压后删除；最后列出每个模型 OK 还是缺失。
    必需模型有缺失时退出码为 1（便携包构建和自动测试靠这个发现问题）。
可改进方向：
    断点续传（大文件下到一半断网不用从头再来）；下载后核对 SHA-256；GitHub 慢时换国内镜像地址。
测评指标：
    不涉及识别效果；看必需模型是否齐全（pipeline.models.missing_models 为空）、
    重复运行是否全部跳过、下载用时。

国内访问 GitHub 慢时：在能上网的电脑上先按 --list 显示的地址把文件下载好，
把文件（或解压好的文件夹，或没解压的 .tar.bz2 压缩包）拷进 models/，再运行本脚本检查；
拷进来的压缩包会被自动解压。
本脚本只下载 sherpa-onnx 公开发布的模型和测试音频，不上传任何东西。
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# 把项目文件夹加入搜索路径，这样直接运行本脚本也能 import pipeline
ROOT = Path(os.path.abspath(__file__)).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.models import MODEL_SPECS, model_ready  # noqa: E402  （必须先改好搜索路径再导入）

RETRIES = 3                # 每个文件最多试几次
CHUNK = 1024 * 1024        # 每次读 1 MB
HINT = "国内访问 GitHub 慢，可以先在别处下载好再拷进 models/"


def download_file(url: str, dest: str | Path) -> None:
    """把 url 下载成文件 dest。失败自动重试，试 3 次都失败就抛出 RuntimeError。

    先下载成“文件名.part”，下载完整后才改成正式文件名：
    这样中途断网不会留下半截文件，下次运行也不会把它当成“已下载”。
    """
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            _download_once(url, part)
            part.replace(dest)
            return
        except Exception as e:  # 网络出错的种类很多（断线、超时、服务器出错），一律重试
            last_error = e
            print(f"    第 {attempt} 次下载失败：{e}", flush=True)
            if part.exists():
                part.unlink()
            if attempt < RETRIES:
                time.sleep(5 * attempt)  # 等一会儿再试，等待时间逐次加长
    raise RuntimeError(f"试了 {RETRIES} 次都没有下载成功，最后一次的错误：{last_error}")


def _download_once(url: str, part: Path) -> None:
    """下载一次，每下完 10% 打印一次进度；下载的字节数不对就算失败。"""
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)  # 文件总大小（服务器可能不告诉）
        done = 0
        reported = 0  # 已经报告到第几个 10%
        with open(part, "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and done * 10 // total > reported:
                    reported = done * 10 // total
                    print(f"    已下载 {done / 1e6:.2f} / {total / 1e6:.2f} MB（{reported * 10}%）", flush=True)
    if total and done != total:
        raise OSError(f"下载不完整：应该是 {total} 字节，实际只有 {done} 字节")
    if not total:
        print(f"    已下载 {done / 1e6:.2f} MB", flush=True)


def extract_tar_bz2(archive: Path, folder: Path) -> None:
    """把 .tar.bz2 压缩包解压到 folder。"""
    with tarfile.open(archive, "r:bz2") as tar:
        if hasattr(tarfile, "data_filter"):
            # filter="data"：不让压缩包里的文件写到 models 文件夹以外（Python 3.11.4 起支持）
            tar.extractall(folder, filter="data")
        else:
            tar.extractall(folder)


def fetch_model(spec: dict, folder: Path) -> None:
    """下载一个模型；如果是压缩包，解压后删除压缩包。"""
    archive = folder / spec["archive"]
    if archive.is_file():
        # 只有压缩包会走到这里：老师可能已经在别处下载好压缩包拷了进来
        print(f"    发现已经拷进来的 {archive.name}，直接解压", flush=True)
    else:
        print(f"    下载 {spec['url']}（约 {spec['size_mb']} MB）", flush=True)
        download_file(spec["url"], archive)
    if archive.name.endswith(".tar.bz2"):
        print("    解压中……", flush=True)
        try:
            extract_tar_bz2(archive, folder)
        except Exception as e:  # 压缩包损坏、磁盘满等
            raise RuntimeError(f"解压失败：{e}。如果压缩包损坏，请删除 {archive} 后重新运行") from e
        archive.unlink()


def print_status(folder: Path) -> None:
    """列出每个模型是 OK 还是缺失（可选模型没下载的显示“未下载”，不影响使用）。"""
    print(f"\n模型情况（文件夹：{folder}）：", flush=True)
    for spec in MODEL_SPECS:
        if model_ready(spec, folder):
            state = "OK    "
        elif spec["required"]:
            state = "缺失  "
        else:
            state = "未下载"
        kind = "必需" if spec["required"] else "可选"
        print(f"  [{state}] {spec['name']:<18} {kind}  {spec['title']}（约 {spec['size_mb']} MB）", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="下载模型到 models/ 文件夹（已下载的自动跳过）")
    parser.add_argument("--optional", nargs="+", action="extend", default=[], metavar="NAME",
                        help="另外下载的可选模型，可以写多个名字；名字用 --list 查看")
    parser.add_argument("--list", action="store_true", help="只列出全部模型和下载情况，不下载")
    parser.add_argument("--models-dir", help="模型文件夹；不填就用 config.yaml 里的 paths.models（即项目里的 models/）")
    args = parser.parse_args(argv)

    if args.models_dir:
        folder = Path(os.path.abspath(args.models_dir))
    else:
        from pipeline.config import load_config

        folder = Path(load_config()["paths"]["models"])

    if args.list:
        print_status(folder)
        for spec in MODEL_SPECS:
            print(f"  {spec['name']}：{spec['url']}", flush=True)
        return 0

    names = [spec["name"] for spec in MODEL_SPECS]
    unknown = [n for n in args.optional if n not in names]
    if unknown:
        print(f"没有叫 {'、'.join(unknown)} 的模型。可以下载的有：{'、'.join(names)}", flush=True)
        return 2

    folder.mkdir(parents=True, exist_ok=True)
    wanted = [spec for spec in MODEL_SPECS if spec["required"] or spec["name"] in args.optional]
    for i, spec in enumerate(wanted, start=1):
        print(f"[{i}/{len(wanted)}] {spec['name']}：{spec['title']}", flush=True)
        if model_ready(spec, folder):
            print("    已存在，跳过", flush=True)
            continue
        start = time.time()
        try:
            fetch_model(spec, folder)
        except Exception as e:  # 一个模型失败不影响下载下一个，最后统一报告
            print(f"    失败：{e}", flush=True)
            print(f"    {HINT}（下载地址：{spec['url']}）", flush=True)
            continue
        if model_ready(spec, folder):
            print(f"    完成，用时 {time.time() - start:.0f} 秒", flush=True)
        else:
            lost = [rel for rel in spec["check"] if not (folder / rel).is_file()]
            print(f"    下载完成，但没有找到：{'、'.join(lost)}", flush=True)

    print_status(folder)
    missing = [spec["name"] for spec in MODEL_SPECS if spec["required"] and not model_ready(spec, folder)]
    if missing:
        print(f"\n必需模型没有下齐：{'、'.join(missing)}。{HINT}", flush=True)
        return 1
    print("\n必需模型都在。", flush=True)
    return 0


if __name__ == "__main__":
    # Windows 上输出被转到日志（例如自动构建）时，默认编码打不出中文会报错，这里统一改成 UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
