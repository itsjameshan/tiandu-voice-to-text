# Windows 安装与运行

两种方式，**第 1 周用方式一**。

## 方式一：便携包（不用联网、不用安装）

1. 拷贝 `tiandu-portable-win64-*.zip`（约 400 MB，老师 U 盘或共享文件夹里有）。
2. 右键 → 全部解压缩，目标文件夹填**纯英文路径**：`D:\asr`（或 U 盘 `E:\asr`）。压缩包里自带一层 `tiandu` 文件夹，解压后就是 `D:\asr\tiandu`。
   - ❌ 不要解压到"桌面""下载""文档"（这些路径里有你的用户名，可能是中文）；
   - ❌ 路径里不要有中文和空格。
3. 打开 `D:\asr\tiandu`，应该能看到 `start.bat`、`selfcheck.bat`、`README.txt`、`python` 文件夹等。
   > 如果打开后还是只有一个 `tiandu` 文件夹（目标文件夹填成了 `D:\asr\tiandu`），说明多套了一层，进去就是；以后各种说明里的 `D:\asr\tiandu` 都换成你实际的文件夹。
4. 双击 **`selfcheck.bat`** 自检，约 1 分钟。每一项都应该是"通过"，最后一行是"全部通过"。
5. 双击 **`start.bat`**，等黑色窗口出现"旅游纠纷录音材料整理（教学原型）：本机浏览器打开 http://127.0.0.1:7860"，浏览器会自动打开页面。
6. 用完关闭黑色窗口。

### 机房电脑会还原

- 便携包放在**不还原的盘**（老师第 0 周测过是哪个）或**自己的 U 盘**里。从 U 盘运行也可以，只是第一次启动稍慢。
- **每次下课前**把自己的录音和 `outputs` 文件夹里的结果拷到 U 盘。

## 方式二：联网安装（老师电脑、自己的电脑）

1. 安装 **Python 3.11（64 位）**：https://www.python.org/downloads/windows/ ，安装时勾选"Add python.exe to PATH"。
2. 下载本仓库：网页上 Code → Download ZIP，解压到 `D:\asr\repo`（纯英文路径）；会用 Git 的话 `git clone`。
3. 双击 `scripts\windows\install.bat`：建虚拟环境、用清华镜像装依赖、下载模型（约 200 MB，从 GitHub 下载，可能较慢）。
4. 双击 `scripts\windows\start.bat` 启动。
5. 需要 TensorFlow（第 6、7 组训练模型）：`.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-tf.txt`

## 自检没通过怎么办

| 自检提示 | 原因和办法 |
|---|---|
| 安装路径含中文或特殊字符（警告） | 把整个文件夹移到 `D:\asr\tiandu` 这样的路径 |
| ffmpeg 不通过 | 便携包被杀毒软件误删了文件，重新解压；联网安装的话重新运行 `install.bat` |
| 模型文件不通过，缺少某某 | 便携包解压不完整，重新解压；联网安装的话运行 `python models\download_models.py` |
| 完整流程不通过 | 把黑色窗口的全部文字截图发给老师 |

## 其他常见问题

- **黑色窗口一闪就关了**：在文件夹地址栏输入 `cmd` 回车，打开命令行，输入 `python\python.exe app.py` 回车，就能看到错误信息。
- **提示端口 7860 被占用**：已经开着一个了。关掉多余的黑色窗口；或者运行 `python\python.exe app.py --port 7861`，浏览器打开 `http://127.0.0.1:7861`。
- **杀毒软件拦截**：便携包里的 `python.exe` 和 `ffmpeg` 是正常程序，请老师在机房的杀毒软件里放行 `D:\asr` 文件夹。
- **别的电脑打不开 `http://IP:7860`**：Windows 防火墙拦了，第一次启动时弹出的防火墙提示要点"允许访问"；机房不允许的话，每人在自己电脑上开。
