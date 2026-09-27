# 多智能体企业知识推理系统

一个会规划、会委派、会验证、会诚实拒答的**真多 Agent** 企业知识推理系统：在 BM25 + 向量混合检索与 Neo4j 知识图谱之上做多跳推理，并用自建评测集与对照实验量化多 Agent 编排相对单 Agent 基线的差异。

> 本仓库是**公开发布版**：内部开发记录（交接日志、偏差日志、实现计划）未包含在内。代码注释与文档里出现的「内部偏差日志第 X 节」均指这些未公开的记录；凡涉及数字的结论，出处一律以 `eval/runs/*.jsonl` 为准。

## 文档

- 设计规格（主文档）：
  [docs/superpowers/specs/2026-09-23-agent-knowledge-hub-refactor-design.md](docs/superpowers/specs/2026-09-23-agent-knowledge-hub-refactor-design.md)
  涵盖参考基线缺陷分析、系统架构、编排层 / MCP 工具层 / 模型层 / 数据层设计、评测体系、里程碑与风险。
- 参考基线缺陷复现记录：[docs/baseline-reproduction.md](docs/baseline-reproduction.md)
- 语料选型：[docs/corpus-selection.md](docs/corpus-selection.md)
- 实验结果：[docs/experiments/2026-09-23-results.md](docs/experiments/2026-09-23-results.md)

## 参考基线

本项目的参考与对照基线为上游开源项目
[bcefghj/agent-knowledge-hub](https://github.com/bcefghj/agent-knowledge-hub)（MIT License）。

按设计决策 D-02，参考项目**仅作为只读基线与素材库**，其源码不复制进本仓库；本地通过 `scripts/mount_baseline.ps1` 把参考项目的解压目录以目录联结（junction）方式挂到 `baseline/`，该挂载点已在 `.gitignore` 中排除。基线仅用于对照实验与缺陷复现，不作为本项目代码本体。

## 核心命题

把一个「固定流水线式的伪多 Agent 企业知识库」，重做成一个真正具备动态路由、任务分解、独立验证与预算控制的真多 Agent 系统，并用自建评测集量化其相对**单 Agent 基线**的差异。

主要对照实验为**两组**（原设计三组中的「Baseline B：参考项目固定 DAG」已于 2026-09-24 放弃，报告里会显式写明由此带来的结论边界）：

| 组别 | 说明 |
|---|---|
| `baseline_a` | 单 Agent ReAct（全工具） |
| `ours` | 真多 Agent（Supervisor + Worker + Verifier） |

上表只说明分组。**实测数字见 [实验结果](docs/experiments/2026-09-23-results.md)**：那里的每一行都由 `eval/runs/*.jsonl` 经 [eval/runners/render_report.py](eval/runners/render_report.py) 渲染生成，**禁止手填**；每项指标同时标出「改善 / 退步 / 持平」与数据缺口，并列出已知局限。

## 实现状态

- 参考基线缺陷复现（只读源码逐条核对，R-1…R-5）：完成，见 `docs/baseline-reproduction.md`
- 系统本体（模型层 / 数据层 / MCP 工具层 / 编排层）：完成，`tests/` 覆盖到单元与编排层
- 评测框架（30 题核心任务集、judge 与评分规则、故障注入、结果渲染器）：完成
- 主实验：已跑并落盘，但**当前口径存在已知缺陷**（图谱写入 0 事件、120 秒墙钟打断、judge 未人工校准），因此实验结果表只作为记录，不作为结论；详细缺口见结果文档的「已知局限」

## 关键约束

以下约束贯穿全部代码、配置与文档：

- 模型名统一为 `deepseek-flash`，`base_url = https://api.deepseek.com`；代码、配置与文档一律不得出现 `deepseek-chat` / `deepseek-reasoner`
- API 花费硬上限 ¥5（含开发联调与正式实验），分配见设计文档 §7.7
- 单元测试一律不发起真实 API 调用；真实调用只出现在打 `@pytest.mark.integration` 的测试与评测 runner 中
- 工具层不调用 LLM；图谱写入必须参数化（Cypher 禁止字符串拼接）；禁止静默吞异常
- 向量化走硅基流动（SiliconFlow）的 `BAAI/bge-large-zh-v1.5`（`.env` 的 `SILICONFLOW_API_KEY`）；不占 DeepSeek 的 ¥5 预算，也不经 DeepSeek
- 语料为 A 股上市公司**年报 + 公告** PDF（光伏产业链 5 家 × 近 2 年，名单见设计文档 §8.1）；`data/` 不入库，只提交 `data/manifest.json`

## 技术栈

Python 3.12 · DeepSeek API（单一模型 `deepseek-flash`，thinking / non-thinking 双模式路由）· 自研显式状态机（**不引入 LangGraph**）· MCP（官方 Python SDK）· Chroma · Neo4j 5.x · SQLite · 嵌入 `BAAI/bge-large-zh-v1.5`（硅基流动 API；`.[embed]` 保留本地 `bge-small-zh-v1.5` 离线兜底）· BM25（`rank-bm25` + `jieba`）+ RRF 融合 · pydantic v2 · pytest + pytest-asyncio · Docker Compose

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # 填入自己的 Key；.env 已被 .gitignore 排除
pytest                           # 单元测试不发起真实 API 调用
```

`@pytest.mark.integration` 的用例需要本地 Neo4j（`docker compose up -d`）或真实 API Key，默认不跑。

## 许可证

本仓库当前未附 LICENSE 文件；如需复用，请先与作者联系。
