# 参考项目缺陷复现记录

> 参考项目：`https://github.com/bcefghj/agent-knowledge-hub`（MIT），本机下载解压到任意本地目录
> （下称「基线目录」，可用 `scripts/mount_baseline.ps1 -Source <基线目录>` 挂到 `baseline/`）。
> 该目录无 `.git`（zip 下载），仅作只读基线。
> 本文件中的「结论」均为 2026-09-24 在本机逐条打开文件核对后的记录，行号对应当前这版代码。

## M0 冒烟：上传 → 提问 → 回答（**已放弃，2026-09-24**）

**状态：已放弃执行，不再作为交付项。** 决定与理由见下方「决定记录」。本节其余内容保留为 runbook，日后若需要真跑参考项目，可直接照做。

### 决定记录（2026-09-24）

| 项 | 内容 |
|---|---|
| 决定 | 放弃 M0「参考项目在本机跑通一次 上传 → 提问 → 回答」的冒烟；不再等待凭据、不再回填执行记录表 |
| 触发原因 | ①该冒烟需要一组**同时提供 chat 与 embeddings** 的 OpenAI 兼容凭据，而 DeepSeek 官方 API 没有 `/embeddings`（实测 404），单独一个 DeepSeek Key 跑不通；②要额外拉起 Neo4j / Chroma / Kafka / MongoDB 四件套，与本项目 Task 3 的 `akr-neo4j` 抢 7474 / 7687 |
| 替代证据 | 基线的对照论点改由**只读源码逐条核对**承担：R-1…R-5 均记录了实测文件、行号与代码片段，足以支撑「固定 DAG / 伪增量 / 图谱残留」这些结论 |
| 已知缺口 | 因此**没有**「参考项目在本机真的能跑起来」的直接证据。Baseline B（spec §9.3 定义为「参考项目原样运行」）若要作为 M6 对照实验的一臂，必须在 M6 之前重新决策：①按本节 runbook 真跑一次；②改用本仓库内自研的固定 DAG 复刻版；③放弃 Baseline B，只保留 Baseline A 与本系统对比 |
| 影响面 | 只影响 M0 的 DoD 完整性（spec §13 四项达成 3 项）与 Baseline B 的证据强度，**不影响 M1 及之后任何代码**——本仓库的模型层 / 数据层 / 工具层与参考项目零依赖 |

- 目的：满足 spec §13 的 M0 验收——参考项目在本机跑通一次最小链路
- 环境：在参考项目目录 `docker compose up -d`（Neo4j / Chroma / Kafka / MongoDB），按 `python/.env.example` 建 `python/.env` 并填 OpenAI 兼容的 Key 与 `OPENAI_BASE_URL`
- 文档：任意一份文本文件即可（参考项目 `python/agents/doc_parser_agent.py:57` 的 `SUPPORTED_EXTENSIONS` 支持 `.pdf` / `.png` / `.jpg` / `.jpeg` / `.csv` / `.xlsx` / `.xls` / `.txt` / `.md`；**不要求年报 PDF**，年报从 M1 起才是正式语料）
- **关键前提：基线同时需要 chat 与 embeddings 两类端点**。`python/services/vector_store.py:61-73` 按 `OPENAI_BASE_URL` 分流——含 `dashscope` 时走自带的 `DashScopeEmbeddings`（打 `https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings`，模型取 `EMBEDDING_MODEL`，默认 `text-embedding-v1`），否则走 `OpenAIEmbeddings`（打 `{OPENAI_BASE_URL}/embeddings`，默认模型 `text-embedding-3-small`）。
  - **DeepSeek 官方 API 只有 chat 端点，没有 `/embeddings`**。因此只给一个 DeepSeek Key 跑不通这条链路：入库阶段生成向量时会失败（而 `python/api/main.py:54-61` 又把 `vector_store.init()` 的异常 `pass` 掉，表现为「服务起来了但检索永远为空」）。可选方案：
    1. 用阿里云百炼 Key（`OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`，`OPENAI_MODEL=qwen-*`，`EMBEDDING_MODEL=text-embedding-v1`）——chat 与 embeddings 都在同一个兼容端点，链路最省事；
    2. 用 OpenAI Key（`gpt-4o-mini` + `text-embedding-3-small`）；
    3. 任意同时提供 chat 与 embeddings 的 OpenAI 兼容网关。
  - 无论选哪种，M0 冒烟只需问答一次，花费可忽略；本项目自身的嵌入走硅基流动 `BAAI/bge-large-zh-v1.5`（`.env` 的 `SILICONFLOW_API_KEY`），与本节的 `OPENAI_*` 端点无关，也不占 DeepSeek 的 ¥5 预算。
- 步骤：
  1. `docker compose up -d`（参考项目根目录），等待 Neo4j / Chroma / Kafka / MongoDB 就绪
  2. `cd python` 建 `.env`（从 `.env.example` 复制），填 `OPENAI_API_KEY` 与 `OPENAI_BASE_URL`
  3. 启动后端：`python -m api.main`（`api_host=0.0.0.0` / `api_port=8080`，见 `python/config/settings.py:38-39`）
  4. 启动前端：`cd fontend/vue-aapp && npm run serve`（dev server 端口 8081，反代到 `http://localhost:8080`，见 `fontend/vue-aapp/vue.config.js:5,8`）
  5. 上传一份 `.txt` / `.md`，提一个答案明确写在文档里的问题，记录返回答案与引用 chunk
  6. 收尾：`docker compose down`，避免与本项目 Task 3 的 Neo4j 抢 7474 / 7687
- 记录：执行命令、耗时、返回答案片段、异常日志（如有）——执行后回填到本段下方

**执行前必须先解决的冲突（2026-09-24 实测）：**

| 端口 | 参考项目用途 | 本机现状 | 处理 |
|---|---|---|---|
| 7474 / 7687 | Neo4j | 已被本项目 `akr-neo4j` 占用 | 先在本项目目录 `docker compose stop` |
| 8000 | Chroma | 2026-09-24 曾被动本机另一套 Docker 栈（15 个容器）占用；**现已 `docker stop`，端口空闲** | 若该栈被重新拉起，再 `docker stop` 对应容器，或临时改参考项目 compose 映射 |
| 9092 / 27017 | Kafka / MongoDB | 空闲 | — |
| 8080 / 8081 | 后端 API / 前端 dev server | 空闲 | — |

### 执行记录

**未执行：本项已于 2026-09-24 放弃（见上方「决定记录」）。** 下表留档，若日后重新真跑参考项目，按同样字段回填。

| 项 | 值 |
|---|---|
| 执行时间 | —（已放弃） |
| 语料文件 | — |
| 提问 | — |
| 返回答案（片段） | — |
| 引用 chunk | — |
| 异常 / 观察 | — |

## R-1 「多 Agent」实为固定 DAG

- 文件：`python/orchestrator/graph.py`
- 操作：搜索 `add_edge`
- 证据（`graph.py:246-249`，入库流水线）：

  ```python
  graph.add_edge("parse", "extract")            # 文档解析节点到知识提取节点的边
  graph.add_edge("extract", "store_vectors")    # 知识提取节点到向量存储节点的边
  graph.add_edge("store_vectors", "store_graph")
  graph.add_edge("store_graph", END)
  ```

- 实测：全仓库 `add_conditional_edges` 只出现 **1** 次，且位于 `_build_update_graph` 的「失败重试」回边（`graph.py:315`），与 Agent 选择无关；问答图 `_build_qa_graph` 同样只有一条 `answer → END` 直边（`graph.py:273`）
- 结论：所谓的「四个 Agent」（解析 / 提取 / 向量 / 图谱）之间没有任何交互与协商，本质是函数串联；路由完全静态

## R-2 「增量更新」实为文档级全量重建

- 文件：`python/agents/knowledge_update_agent.py` 的 `_handle_modify`（`knowledge_update_agent.py:238-245`）

  ```python
  async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
      doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]
      if self.vector_store:
          deleted = await self.vector_store.delete_by_doc_id(doc_id)
          result.vectors_deleted = deleted
      await self._handle_create(change, result)
  ```

- 实测：修改路径先按 `doc_id` 删掉该文档全部向量，然后走完整的 `_handle_create` 重新解析入库；对比同文件 `_handle_delete`（`:247-255`）会调用 `knowledge_graph.delete_by_source`，修改路径**不清理图谱旧实体**
- 文件：`python/services/cdc_processor.py` 的 `compute_diff`（`cdc_processor.py:101-124`）

  ```python
  before_set = set(before_lines)
  after_set = set(after_lines)
  added = after_set - before_set
  removed = before_set - after_set
  ```

- 实测：用集合差做行级差量，丢失重复行与顺序信息；`change_ratio` 也只是 `len(added | removed) / (len(before) + len(after))`
- 结论：既不是 chunk 级增量，也不保证图谱与向量一致（图谱会残留已删除内容）

## R-3 混合检索分数不可比

- 文件：`python/agents/qa_agent.py`
- 实测：
  - 向量分支（`:280-288`）直接取 `vector_store.search` 返回的 `score`
  - 图谱分支（`:313-326`）对每条记录硬编码 `score=0.8`（`:320`），分支内的异常被 `except Exception: continue` 吞掉（`:324-325`）
  - 两支结果随后在 `_hybrid_rerank`（`:330-349`）里乘权重 `{"vector": 1.0, "graph": 1.2, "hybrid": 1.1}`（`:336`）后统一排序
  - 置信度 `_calc_confidence`（`:424-429`）只是上下文分数的算术平均
- 结论：向量相似度（0–1 连续值）与图谱常量 0.8、以及重排权重三者量纲混用，排序结果不可解释；「置信度」不反映答案是否有引用支撑

## R-4 无测试无评测

- 路径：`python/tests/`
- 实测：目录下**只有**一个 0 字节的 `__init__.py`，无任何测试用例
- 同时仓库 `README.md` 却给出 `pytest tests/` 的使用指引
- 结论：参考项目没有任何自动化验证，也无法给出质量指标——这正是本项目自建评测集（M2）的动机

## R-5 健壮性与安全性硬伤

- **Cypher 拼接注入**：`python/services/knowledge_graph.py` 的 `add_relation`（`:83-99`）把 LLM 抽出的关系类型直接拼进语句：

  ```python
  rel_type = relation.relation.upper().replace(" ", "_")
  cypher = f"""
  MATCH (h:Entity {{name: $head}})
  MATCH (t:Entity {{name: $tail}})
  MERGE (h)-[r:{rel_type}]->(t)
  ...
  ```

  只有参数用了 `$` 占位，关系类型没有白名单校验
- **带病启动**：`python/api/main.py` 的 `lifespan`（`:49-61`）把两个关键初始化包在 `except Exception: pass` 里

  ```python
  try:
      await vector_store.init()
  except Exception:
      pass
  try:
      await knowledge_graph.init()
  except Exception:
      pass
  ```

  向量库或图谱起不来时服务照常对外提供服务，失败被静默吞掉
- **静默吞异常**：`python/agents/qa_agent.py:324`（图谱检索整段 `continue`）、`python/agents/doc_parser_agent.py` 的 `_parse_pdf` 及同类解析分支共 5 处 `except Exception:`（`:136, :155, :179, :211, :250`）
- 结论：这三类问题在目标架构里被显式约束反向排除——Cypher 必须参数化 + 关系类型白名单、禁止静默吞异常（本项目 README「关键约束」与 spec §16）
