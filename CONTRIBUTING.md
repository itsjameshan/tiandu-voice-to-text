# 参与贡献

这是一门实训课的教学仓库。

- **学生**：只有组长提交代码，只改本组文件。步骤见 [docs/guides/submit_code.md](docs/guides/submit_code.md)；用 AI 编程工具前读 [docs/guides/ai_tools.md](docs/guides/ai_tools.md)。
- **老师**：审核和合并见 [docs/teacher/merging.md](docs/teacher/merging.md)。
- **开发模板本身**：先读 [CLAUDE.md](CLAUDE.md)（红线和约定）、[设计文档](docs/superpowers/specs/2026-10-07-teaching-template-design.md) 和 [GLOSSARY.md](GLOSSARY.md)。先写失败的测试再实现；提交前 `pytest -q` 和 `ruff check .` 都要通过。

所有提交都会在 Windows 和 Linux 上自动跑测试（`.github/workflows/ci.yml`）。
