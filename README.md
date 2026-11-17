# DIVE Memory

长期AI交互记忆项目。当前主线为`dive_memory.py`及`oracle_probe.py validate-design`，提供来源可追溯的声明提取、查旧、关系判断、证据检索、回答和记录。

**状态：**80项离线测试通过；1题真实LongMemEval Oracle完整调用链完成，人工核对回答与参考一致，尚无官方Judge成绩。当前仍是设计验证程序，通用增量交互、删除、容量管理和500题批量入口待实现。

## 文件与入口

| 文件 | 用途 |
|---|---|
| `dive_memory.py` | 声明、编号组、操作校验、来源级纠错及派生状态 |
| `oracle_probe.py validate-design` | 当前主线：检索检查、自拟语义验证、单题Oracle与受限恢复 |
| `embedding_probe.py` | 独立BGE-M3环境检查、首次下载、自拟排序验证 |
| `dive_minimal.py` | 早期标准库词法原型；同时提供现有DeepSeek适配器 |
| `oracle_probe.py plan/run` | 早期准入与检索探测，不包含当前W1/W2更新逻辑 |
| `DIVE_方案工作稿.md` | 首版方案、进展、预算与后续计划 |

## 1. 安装环境

已验证Windows、Python 3.12.16。以下在PowerShell执行，使用独立Conda环境；已有项目环境也可使用其Python。`D:\DIVE`为本机示例，其他位置使用实际项目路径。

```powershell
Set-Location D:\DIVE
conda create -n dive-memory python=3.12 pip
conda activate dive-memory
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers==6.2.0 transformers==5.22.2 numpy==2.5.4 huggingface-hub==2.4.0
```

SQLite由Python标准库提供，不另部署数据库服务。

## 2. 模型与公开数据

初次运行下列命令下载固定版本BGE-M3并检查编码，需要网络和磁盘空间。后续验证只加载缓存，不自动重新下载。

```powershell
python -X utf8 -B .\embedding_probe.py
```

缓存为`runs/models/`，修订为`5617a9f61b028005a4858fdac845db406aefb181`。使用CPU，最多8线程、批量2条、上限512个BGE Token。

按[data/README.md](data/README.md)获取Oracle，或在项目根目录执行：

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json" -OutFile .\data\longmemeval_oracle.json
Get-FileHash -LiteralPath .\data\longmemeval_oracle.json -Algorithm SHA256
```

预期SHA256：`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`。原文件不改写，不上传仓库。

## 3. 离线检查

单元测试使用模拟模型，不读取密钥、不下载模型、不产生API费用：

```powershell
python -X utf8 -B -m unittest discover -s tests -v
```

本地语义检索检查需要已有模型缓存，无付费请求：

```powershell
python -X utf8 -B .\oracle_probe.py validate-design --validation-mode retrieval
```

测试覆盖来源、引用、时间、部分纠错、冲突、持久化重开、证据依赖、完整预算、用量和恢复。离线通过不能代替真实模型效果评测。

## 4. 单题真实Oracle

```powershell
python -X utf8 -B .\oracle_probe.py validate-design --validation-mode oracle --start 0 --max-calls 16
```

`--start`从0开始。密钥仅在本机隐藏输入，不写入代码、参数或日志。当前模型为`deepseek-flash`，关闭思考模式。新付费请求逐次显示阶段，无自动重试。

`--max-calls`是新请求硬上限，不是每题固定次数。W1提取后，按声明和完整输入大小重新规划W2并核对剩余额度。W1/W2/回答输出上限为4096/2048/512 Token，输入另计。

路径：历史整理→W1选择和提取→原文向量查旧→W2提案及程序校验→SQLite保存重开→有限证据→回答与预测。摄入不传未来问题、答案、题型、`has_answer`或带标签的原会话ID。

## 5. 结果与恢复

每次验证在`runs/design-validation/`下生成独立目录：

- `summary.json`：版本、配置、结果、复用、用量与失败位置。
- `calls.jsonl`：新请求开始及结果；同一编号统计最后一条，不能重复计费。
- `responses.jsonl`：有界请求及返回，校验前保存，用于离线复核。
- `memory.sqlite3`：完整处理后的验证快照，非生产数据库结构。
- `predictions.jsonl`：成功答题后生成`question_id`和字符串`hypothesis`。

在相同Oracle命令后添加`--resume-from`并指定本人先前生成的目录，可核验复用返回。先附加`--replay-only`进行不读取密钥的离线恢复；缺少下一阶段返回时，状态为`replay_incomplete`，程序退出码为2，表示还有真实步骤待执行。真实续跑时移除该参数，并设置新增请求上限。

恢复检查数据、题目、模型、输入、声明编号和依赖。不匹配或记录不完整时停止，不自动重新付费。原日志不改写，复用与新调用分开记录；修改提示/规则后的旧返回复核不代表新提示已实测。仅支持受限单题恢复，不提供任意跨版本迁移。

用量缺失标未知，推导和矛盾单列；接口故障不转为正常拒答。日志仅用于公开实验数据，可能包含原文与回答。

## 6. 规则与边界

声明是来源、时间和有效状态的唯一依据，持久组只保存`record_id/support_ids`。支持保留声明身份；变化方向由可靠时间计算；纠错或撤回定位具体声明，其他独立依据继续有效。合法来源的矛盾关系保留待确认，未知ID、伪造引用仍拒绝。

问题检索最多8个原文候选。证据补齐关键观察、上下文及变化/纠错依赖，同时受8个片段、16 KiB证据和24 KiB完整请求约束。超限省略整个依赖组并标记不完整；依赖很大时可能没有可用证据，准确率需另测。消息日期不自动等于事件日期。

当前单次Oracle从空空间处理历史并保存快照，不代表已完成持续增量SDK或交互应用。删除、容量设计见方案，尚未实现。

## 7. 已有结果

2026-11-14首题`gpt4_2655b836`完成：33条声明、30次新增、3次待确认，数据库重开一致。回答为首次保养后的GPS故障，引用两条原文，人工核对与参考一致，未运行官方Judge。

续跑复用7条返回，新增8次请求、55,612 Token；本题两次运行累计15次请求、87,867 Token，模型请求耗时44.13秒，不含本地加载等时间。本题未成功执行支持合并、变化、纠错或撤回，不能认定全部机制通过真实评测。

离线测试80项通过。本地记录为`runs/design-validation/acceptance-20261114-165233/`及`runs/design-validation/20261114-100213-20bafda1/`，不随仓库上传。当前没有全量500题成绩。

## 提交范围与参考

只提交代码、测试、方案及复现说明。忽略`runs/`、`.venv/`、数据JSON、`.env`和Python缓存。手册及导师邮件不纳入代码提交，密钥不写入提交文件。

- [技术方案](DIVE_方案工作稿.md)
- [Oracle来源与检查](data/README.md)
- [LongMemEval官方代码](https://github.com/xiaowu0162/LongMemEval)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [DeepSeek接口](https://api-docs.deepseek.com/api/create-chat-completion/)
