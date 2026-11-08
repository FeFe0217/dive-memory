# LongMemEval Oracle 数据

获取与检查日期：2026-10-24。仅下载手册指定的Oracle档，未下载S/M或V2，未运行模型或评分。

## 来源与版本

- [官方仓库的下载说明](https://github.com/xiaowu0162/LongMemEval#data)
- [官方清理版数据集](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)，数据页标注MIT许可。
- 固定版本：`98d7416c24c778c2fee6e6f3006e7a073259d48f`。
- [该版本Oracle原文件](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json)。本地原文件未改写。
- 文件大小：15,388,478字节。
- SHA256：`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`，与官方文件信息匹配。

## 实际结构检查

- JSON解析成功；500题、500个唯一ID，30题ID以`_abs`结尾。
- 6类题型：temporal-reasoning 133、multi-session 133、knowledge-update 78、single-session-preference 30、single-session-assistant 56、single-session-user 70。
- 共948个历史会话、10,960条消息；每题1–6个会话。各题的会话ID、日期和会话内容三组数组长度一致，消息角色与文本字段检查通过。
- 所有提问日期均可解析，答案会话ID均能对应历史会话。存在非标准的消息角色顺序，导入时不能假定所有会话都从用户发言开始并严格交替。
- 34题的历史会话不是时间升序。导入时将ID、日期、内容一起按解析后的日期稳定排序，不改变原文件；同一时间保留原相对顺序。
- `answer`有468个字符串、32个整数。参考答案类型不统一不表示文件损坏；最终预测`hypothesis`仍须为字符串。

## 哪些内容交给系统

- 摄入阶段：每个实例单独处理，仅提供来源会话ID、日期及消息的`role`和`content`；不提供未来问题。
- 回答阶段：再提供`question`和`question_date`。后者表示提问时间，有助于解释“现在”“去年”等表达。
- 只在外部评估/追踪使用：`question_id`、`question_type`、`answer`、`answer_session_ids`、消息中的`has_answer`。不能将答案位置或正确答案传入记忆与回答路径。

本次仅检查元数据与结构，未阅读具体题目答案做调参，未生成预测或成绩。后续开发样例若用于调试，应明确记录，不能作为未见过的测试样例。
