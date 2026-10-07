"""旅游纠纷录音材料整理（教学原型）的处理流程。

对外接口：
- run_pipeline(path, options=None, progress=None, cfg=None) -> (segments, meta)：按顺序执行八步。
- 各步骤的公开函数（见 pipeline/step1_ingest.py … step8_report.py）。
- 做法登记（见 pipeline/methods.py）：每组在 pipeline/groups/ 里给某个槽位加自己的做法。

注意：本文件导入时不能加载 sherpa-onnx、gradio、tensorflow 等重依赖，
这样只装了 TensorFlow 的电脑也能运行分类训练脚本。重依赖一律在函数内部导入。
"""

__version__ = "0.1.0"
