"""Local BGE-M3 feasibility check with synthetic examples; not Oracle evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import time
import uuid


ROOT = Path(__file__).resolve().parent
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

# Expected IDs are used only for checking results, never as model inputs.
PASSAGES = [
    ("zh-home", "zh", "10月1日，我已经从成都搬到了杭州，之后长期住在杭州。"),
    ("zh-trip", "zh", "10月18日至20日，我会去上海开客户会议，结束后返回。"),
    ("zh-learning", "zh", "学习新知识时，我更容易理解具体例子，再看抽象解释。"),
    ("zh-food", "zh", "我吃花生会过敏，需要避开含花生的食物。"),
    ("en-home", "en", "On October 1 I moved from Chengdu to Hangzhou. Hangzhou is now my permanent residence."),
    ("en-trip", "en", "I will be in Shanghai for a client meeting on October 18-20, and will return afterward."),
    ("en-learning", "en", "When I study something unfamiliar, concrete worked examples help me understand before abstract explanations."),
    ("en-food", "en", "I am allergic to peanuts and need to avoid food containing them."),
]
QUESTIONS = [
    ("zh", "我如今把家安在哪座城市？", "zh-home"),
    ("zh", "给我讲一个陌生概念，最好怎么开头？", "zh-learning"),
    ("en", "Where do I live these days?", "en-home"),
    ("en", "What teaching style suits me best?", "en-learning"),
]


def main():
    report = {
        "kind": "synthetic-embedding-feasibility-check",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "requested_model_revision": MODEL_REVISION,
        "device": "cpu",
        "max_sequence_tokens": 512,
        "batch_size": 2,
        "paid_model_api_calls": 0,
        "limits": "Four synthetic queries only. No memory ingestion, state update, Oracle score or peak RAM measurement.",
        "passages": [{"id": pid, "language": lang, "text": text} for pid, lang, text in PASSAGES],
    }
    report["packages"] = {}
    for name in ("torch", "sentence-transformers", "transformers", "numpy", "huggingface-hub"):
        try:
            report["packages"][name] = version(name)
        except PackageNotFoundError:
            report["packages"][name] = None

    stage = "import_libraries"
    exit_code = 0
    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        report["cpu_threads"] = torch.get_num_threads()
        stage = "load_or_download_model"
        print("正在加载BGE-M3；首次运行会下载模型，缓存保存在runs/models。", flush=True)
        started = time.perf_counter()
        model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device="cpu",
            cache_folder=str(ROOT / "runs" / "models"),
            trust_remote_code=False,
            token=False,
        )
        model.max_seq_length = report["max_sequence_tokens"]
        report["load_seconds_including_any_download"] = round(time.perf_counter() - started, 3)
        report["resolved_model_revision"] = getattr(model[0].auto_model.config, "_commit_hash", None)

        stage = "encode_texts"
        texts = [text for _, _, text in PASSAGES] + [query for _, query, _ in QUESTIONS]
        token_lengths = [len(model.tokenizer.encode(text)) for text in texts]
        report["input_token_lengths"] = token_lengths
        if max(token_lengths) > model.max_seq_length:
            raise ValueError("Probe input exceeds the configured sequence length; nothing was silently truncated.")
        started = time.perf_counter()
        vectors = model.encode(
            texts, batch_size=report["batch_size"], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        )
        report["encode_seconds"] = round(time.perf_counter() - started, 3)
        report["vector_shape"] = list(vectors.shape)
        if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
            raise ValueError("Unexpected vector shape or non-finite values.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=0.001):
            raise ValueError("Vectors are not unit length as requested.")

        stage = "rank_passages"
        report["results"] = []
        for number, (language, query, expected_id) in enumerate(QUESTIONS):
            candidates = [i for i, (_, lang, _) in enumerate(PASSAGES) if lang == language]
            # Dot product of unit-length vectors equals cosine similarity.
            scores = vectors[candidates] @ vectors[len(PASSAGES) + number]
            order = np.argsort(-scores, kind="stable")
            ranked = [
                {"id": PASSAGES[candidates[int(i)]][0], "score": round(float(scores[i]), 6)}
                for i in order
            ]
            item = {"query": query, "expected_id": expected_id, "ranked": ranked,
                    "expected_is_first": ranked[0]["id"] == expected_id}
            report["results"].append(item)
            print("\n问题：" + query)
            for result in ranked[:3]:
                print(f"  {result['id']}: {result['score']:.4f}")
            print("预期片段是否排第一：" + str(item["expected_is_first"]))

        hits = sum(item["expected_is_first"] for item in report["results"])
        report["top1_matches"] = hits
        report["status"] = "completed"
        print(f"\n向量检查通过；自拟查询首位匹配：{hits}/{len(QUESTIONS)}。")
        print(f"文本编码耗时：{report['encode_seconds']}秒。")
        print("这是本地可行性检查，不是Oracle成绩，也不证明更新机制已实现。")
    except (Exception, KeyboardInterrupt) as exc:
        exit_code = 1
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        # Download errors may contain long signed URLs; exclude them from saved output.
        detail = re.sub(r"https?://\S+", "[URL omitted]", str(exc))[:600]
        report["error"] = {"stage": stage, "type": type(exc).__name__, "detail": detail}
        print(f"检查未完成：{stage} / {type(exc).__name__}\n{detail}")

    folder = ROOT / "runs" / "embedding-probe"
    path = folder / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".json")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("运行记录：" + str(path))
    except OSError as exc:
        print("运行记录未能保存：" + type(exc).__name__)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
