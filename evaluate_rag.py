"""Evaluate the existing local OSHA RAG pipeline without rebuilding ChromaDB.

Run from the project folder:
    python evaluate_rag.py
    python evaluate_rag.py --limit 3

Outputs:
    evaluation_results.csv
    evaluation_results.json

Metrics are deliberately transparent. Lexical evidence overlap is only a heuristic
proxy for faithfulness; it is not a semantic or human faithfulness judgment.
The script calls the existing retrieve_documents/generate_answer functions and does
not modify the persistent vector database.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chat as rag

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "evaluation_results.csv"
JSON_PATH = PROJECT_DIR / "evaluation_results.json"
ABSTENTION = (
    "The available documents do not provide enough specific information to answer this question safely."
)

# Ground-truth labels are based on the known relevant passages in the two indexed
# OSHA PDFs. Page numbers are PDF page metadata, not printed booklet page numbers.
# For broad/unsupported questions, expected_pages is empty and retrieval metrics are N/A.
EVALUATION_CASES: list[dict[str, Any]] = [
    {
        "id": "jha_five_questions",
        "question": "What are the five questions to ask when conducting a JHA?",
        "expected_pages": {("OSHA_3071_Job_Hazard_Analysis.pdf", 12)},
        "should_abstain": False,
        "must_contain": ["What can go wrong?", "What are the consequences?", "How likely is it that the hazard will occur?"],
    },
    {
        "id": "jha_definition_importance",
        "question": "What is a Job Hazard Analysis and why is it important?",
        "expected_pages": {
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 1),
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 2),
        },
        "should_abstain": False,
        "must_contain": ["hazard"],
    },
    {
        "id": "casting_hazards",
        "question": "What are the hazards associated with casting?",
        "expected_pages": {
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 15),
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 16),
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 17),
        },
        "should_abstain": False,
        "must_contain": ["casting"],
    },
    {
        "id": "construction_core_elements",
        "question": "What are the seven core elements of a construction safety and health program?",
        "expected_pages": {("OSHA_3886_Construction_Safety.pdf", 11)},
        "should_abstain": False,
        "must_contain": ["Management Leadership", "Worker Participation", "Program Evaluation and Improvement"],
    },
    {
        "id": "chemical_exposure_controls",
        "question": "How should chemical exposure hazards be controlled?",
        "expected_pages": {
            ("OSHA_3886_Construction_Safety.pdf", 21),
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 46),
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 47),
        },
        "should_abstain": False,
        "must_contain": ["SDS", "product labels"],
    },
    {
        "id": "document_comparison",
        "question": "Compare the purpose of the two OSHA documents.",
        "expected_pages": {
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 1),
            ("OSHA_3886_Construction_Safety.pdf", 11),
        },
        "should_abstain": False,
        "must_contain": ["task-focused", "construction safety"],
    },
    {
        "id": "hierarchy_of_controls",
        "question": "Explain the hierarchy of controls described in the construction safety guide.",
        "expected_pages": {("OSHA_3886_Construction_Safety.pdf", 25)},
        "should_abstain": False,
        "must_contain": ["control"],
    },
    {
        "id": "specific_respirator_abstention",
        "question": "What specific respirator should I use for chemical exposure?",
        "expected_pages": set(),
        "should_abstain": True,
        "must_contain": [],
    },
    {
        "id": "unsupported_topic_abstention",
        "question": "What is the exact lockout procedure for a hydraulic press model XJ-400?",
        "expected_pages": set(),
        "should_abstain": True,
        "must_contain": [],
    },
    {
        "id": "jha_correction_prevention",
        "question": "How can hazards be corrected or prevented during a JHA?",
        "expected_pages": {("OSHA_3071_Job_Hazard_Analysis.pdf", 18)},
        "should_abstain": False,
        "must_contain": ["hazard"],
    },
    {
        "id": "casting_sharp_edge_controls",
        "question": "How can cuts from sharp casting edges be prevented?",
        "expected_pages": {("OSHA_3071_Job_Hazard_Analysis.pdf", 16)},
        "should_abstain": False,
        "must_contain": ["casting"],
    },
    {
        "id": "casting_foot_injury_controls",
        "question": "What injury can happen if a worker drops a casting?",
        "expected_pages": {("OSHA_3071_Job_Hazard_Analysis.pdf", 15)},
        "should_abstain": False,
        "must_contain": ["foot"],
    },
    {
        "id": "construction_hazard_information",
        "question": "Explain hazard identification in the construction guide.",
        "expected_pages": {
            ("OSHA_3886_Construction_Safety.pdf", 19),
            ("OSHA_3886_Construction_Safety.pdf", 20),
            ("OSHA_3886_Construction_Safety.pdf", 21),
        },
        "should_abstain": False,
        "must_contain": ["hazard"],
    },
    {
        "id": "chemical_identification_sds",
        "question": "How do I identify chemical hazards on a construction site?",
        "expected_pages": {("OSHA_3886_Construction_Safety.pdf", 21)},
        "should_abstain": False,
        "must_contain": ["SDS"],
    },
    {
        "id": "hierarchy_control_order",
        "question": "What is the hierarchy of controls?",
        "expected_pages": {("OSHA_3886_Construction_Safety.pdf", 25)},
        "should_abstain": False,
        "must_contain": ["Elimination", "PPE"],
    },
    {
        "id": "unsupported_specific_respirator_cartridge",
        "question": "Which respirator cartridge is required for xylene at 50 ppm?",
        "expected_pages": set(),
        "should_abstain": True,
        "must_contain": [],
    },
    {
        "id": "unsupported_machine_specific_isolation",
        "question": "Give the exact isolation sequence for a hydraulic press model XJ-400.",
        "expected_pages": set(),
        "should_abstain": True,
        "must_contain": [],
    },
]

CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,\]]+),\s*PDF page\s*(\d+)\]", re.IGNORECASE
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "the", "and", "for", "are", "what", "how", "does", "can", "with", "from",
    "that", "this", "these", "those", "about", "into", "when", "where", "which",
    "should", "would", "could", "your", "you", "their", "they", "have", "has",
    "was", "were", "will", "then", "than", "also", "such", "only", "using", "used",
    "use", "not", "but", "its", "our", "all", "any", "may", "more", "most", "each",
    "from", "page", "source", "osha", "pdf", "document", "documents",
}


def parse_pages(text: str) -> set[tuple[str, int]]:
    return {
        (match.group(1).strip(), int(match.group(2)))
        for match in CITATION_PATTERN.finditer(text or "")
    }


def informative_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(CITATION_PATTERN.sub("", text or ""))
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def rounded(value: float | None) -> float | str:
    return round(value, 4) if value is not None else "N/A"


def evidence_overlap_proxy(answer: str, context: str) -> float | None:
    """Token overlap only; useful as a diagnostic, not a true faithfulness score."""
    answer_tokens = informative_tokens(answer)
    context_tokens = informative_tokens(context)
    if not answer_tokens:
        return None
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def judge_faithfulness_with_local_llm(question: str, context: str, answer: str) -> dict[str, Any]:
    """Optional local LLM-as-judge score; useful diagnostic, not ground truth."""
    response = rag.ollama.chat(
        model=rag.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of RAG answer faithfulness. Compare the answer only "
                    "with the supplied context. Do not reward general correctness unless the context "
                    "supports it. Return only valid JSON with keys: score (integer 1-5), "
                    "unsupported_claims (array of short strings), missing_important_evidence "
                    "(array of short strings), rationale (one short sentence). Score 5 means every "
                    "material claim is supported and no important supported point is omitted; 1 "
                    "means mostly unsupported or contradicted. If the answer is the exact abstention "
                    "sentence, score 5 when the context is insufficient to answer safely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"
                    f"\n\nANSWER:\n{answer}"
                ),
            },
        ],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 250},
    )
    raw = (response.message.content or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"score": None, "unsupported_claims": [], "missing_important_evidence": [],
                "rationale": "Judge response was not valid JSON."}
    try:
        data = json.loads(match.group(0))
        score = int(data.get("score"))
        if score < 1 or score > 5:
            raise ValueError("score outside 1-5")
        return {
            "score": score,
            "unsupported_claims": data.get("unsupported_claims", []),
            "missing_important_evidence": data.get("missing_important_evidence", []),
            "rationale": str(data.get("rationale", "")),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"score": None, "unsupported_claims": [], "missing_important_evidence": [],
                "rationale": "Judge response could not be parsed."}


def evaluate_case(case: dict[str, Any], collection: Any, use_llm_judge: bool = False) -> dict[str, Any]:
    question = case["question"]
    started = time.perf_counter()
    context = rag.retrieve_documents(question, collection)
    retrieval_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    answer = rag.generate_answer(question, context)
    answer_ms = (time.perf_counter() - started) * 1000
    total_ms = retrieval_ms + answer_ms

    expected_pages = {tuple(item) for item in case["expected_pages"]}
    retrieved_pages = parse_pages(context)
    cited_pages = parse_pages(answer)
    has_ground_truth = bool(expected_pages)

    retrieval_precision = (
        safe_ratio(len(retrieved_pages & expected_pages), len(retrieved_pages))
        if has_ground_truth else None
    )
    retrieval_recall = (
        safe_ratio(len(retrieved_pages & expected_pages), len(expected_pages))
        if has_ground_truth else None
    )
    citation_precision_vs_expected = (
        safe_ratio(len(cited_pages & expected_pages), len(cited_pages))
        if has_ground_truth else None
    )
    citation_recall_vs_expected = (
        safe_ratio(len(cited_pages & expected_pages), len(expected_pages))
        if has_ground_truth else None
    )
    citation_support_precision = (
        safe_ratio(len(cited_pages & retrieved_pages), len(cited_pages))
        if cited_pages else 1.0
    )

    abstained = answer.strip() == ABSTENTION
    abstention_correct = abstained == bool(case["should_abstain"])
    missing_answer_terms = [
        term for term in case.get("must_contain", [])
        if term.casefold() not in answer.casefold()
    ]
    answer_expectations_pass = not missing_answer_terms

    judge_started = time.perf_counter()
    judge_result = judge_faithfulness_with_local_llm(question, context, answer) if use_llm_judge else None
    judge_latency_ms = (time.perf_counter() - judge_started) * 1000 if use_llm_judge else None

    return {
        "id": case["id"],
        "question": question,
        "category": case.get("category", "legacy"),
        "split": case.get("split", "legacy"),
        "answer": answer,
        "retrieved_pages": sorted(f"{source}#p{page}" for source, page in retrieved_pages),
        "cited_pages": sorted(f"{source}#p{page}" for source, page in cited_pages),
        "expected_pages": sorted(f"{source}#p{page}" for source, page in expected_pages),
        "retrieval_precision": rounded(retrieval_precision),
        "retrieval_recall": rounded(retrieval_recall),
        "citation_precision_vs_expected": rounded(citation_precision_vs_expected),
        "citation_recall_vs_expected": rounded(citation_recall_vs_expected),
        "citation_support_precision": rounded(citation_support_precision),
        "evidence_overlap_proxy_not_faithfulness": rounded(evidence_overlap_proxy(answer, context)),
        "should_abstain": bool(case["should_abstain"]),
        "did_abstain": abstained,
        "abstention_correct": abstention_correct,
        "answer_expectations_pass": answer_expectations_pass,
        "missing_expected_terms": missing_answer_terms,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "answer_latency_ms": round(answer_ms, 2),
        "total_latency_ms": round(total_ms, 2),
        "llm_judge_faithfulness_score_1_to_5": judge_result["score"] if judge_result else "NOT RUN",
        "llm_judge_unsupported_claims": judge_result["unsupported_claims"] if judge_result else [],
        "llm_judge_missing_important_evidence": judge_result["missing_important_evidence"] if judge_result else [],
        "llm_judge_rationale": judge_result["rationale"] if judge_result else "NOT RUN",
        "llm_judge_latency_ms": round(judge_latency_ms, 2) if judge_latency_ms is not None else "N/A",
        "human_faithfulness_review": "Recommended — local LLM judge is a diagnostic, not ground truth",
    }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local OSHA RAG pipeline.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (smoke test).")
    parser.add_argument("--benchmark", type=str, default="", help="Optional JSON benchmark path; defaults to the built-in 17-case suite.")
    parser.add_argument("--split", choices=["all", "development", "holdout"], default="all", help="Select benchmark split when --benchmark is supplied.")
    parser.add_argument("--no-save", action="store_true", help="Print summary without writing result files.")
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Use the local Ollama model as a faithfulness judge (slower; diagnostic only).",
    )
    args = parser.parse_args()

    benchmark_name = "built_in_17_case"
    if args.benchmark:
        benchmark_path = Path(args.benchmark).expanduser()
        if not benchmark_path.is_absolute():
            benchmark_path = PROJECT_DIR / benchmark_path
        try:
            benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
            cases = benchmark_data["cases"]
            benchmark_name = benchmark_data.get("dataset_name", benchmark_path.stem)
        except Exception as exc:
            print(f"ERROR: Could not load benchmark {benchmark_path}: {exc}", file=sys.stderr)
            return 2
        if args.split != "all":
            cases = [case for case in cases if case.get("split", "development") == args.split]
        if not cases:
            print("ERROR: The selected benchmark split contains no cases.", file=sys.stderr)
            return 2
    else:
        cases = EVALUATION_CASES
        if args.split != "all":
            print("ERROR: --split requires --benchmark.", file=sys.stderr)
            return 2
    cases = cases[:args.limit] if args.limit > 0 else cases
    print("OSHA RAG evaluation")
    print(f"Benchmark: {benchmark_name} | Split: {args.split}")
    print(f"Cases: {len(cases)} | DB: {rag.DB_DIR}")
    print("This uses the existing ChromaDB; it does not ingest or modify documents.\n")

    try:
        collection = rag.load_collection()
    except Exception as exc:
        print(f"ERROR: Could not load the existing collection: {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        try:
            row = evaluate_case(case, collection, use_llm_judge=args.llm_judge)
            rows.append(row)
            print(
                f"  retrieval P/R={row['retrieval_precision']}/{row['retrieval_recall']} | "
                f"citation support={row['citation_support_precision']} | "
                f"abstention={'PASS' if row['abstention_correct'] else 'FAIL'} | "
                f"latency={row['total_latency_ms']} ms"
            )
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({
                "id": case["id"], "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    completed = [row for row in rows if "error" not in row]
    unsupported_rows = [row for row in completed if row.get("should_abstain") is True]
    supported_rows = [row for row in completed if row.get("should_abstain") is False]
    summary = {
        "benchmark_name": benchmark_name,
        "split": args.split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": str(rag.DB_DIR),
        "collection_name": rag.COLLECTION_NAME,
        "embedding_model": rag.EMBED_MODEL,
        "answer_model": rag.LLM_MODEL,
        "cases_requested": len(cases),
        "cases_completed": len(completed),
        "cases_errored": len(rows) - len(completed),
        "mean_retrieval_precision": rounded(mean_or_none(numeric_values(completed, "retrieval_precision"))),
        "mean_retrieval_recall": rounded(mean_or_none(numeric_values(completed, "retrieval_recall"))),
        "mean_citation_precision_vs_expected": rounded(mean_or_none(numeric_values(completed, "citation_precision_vs_expected"))),
        "mean_citation_recall_vs_expected": rounded(mean_or_none(numeric_values(completed, "citation_recall_vs_expected"))),
        "mean_citation_support_precision": rounded(mean_or_none(numeric_values(completed, "citation_support_precision"))),
        "mean_evidence_overlap_proxy_not_faithfulness": rounded(mean_or_none(numeric_values(completed, "evidence_overlap_proxy_not_faithfulness"))),
        "abstention_accuracy": rounded(mean_or_none([1.0 if row["abstention_correct"] else 0.0 for row in completed])) if completed else "N/A",
        "supported_answer_rate": rounded(mean_or_none([1.0 if not row["did_abstain"] else 0.0 for row in supported_rows])) if supported_rows else "N/A",
        "unsupported_abstention_recall": rounded(mean_or_none([1.0 if row["did_abstain"] else 0.0 for row in unsupported_rows])) if unsupported_rows else "N/A",
        "unsupported_retrieval_candidate_rate": rounded(mean_or_none([1.0 if row.get("retrieved_pages") else 0.0 for row in unsupported_rows])) if unsupported_rows else "N/A",
        "answer_expectation_pass_rate": rounded(mean_or_none([1.0 if row.get("answer_expectations_pass") else 0.0 for row in supported_rows])) if supported_rows else "N/A",
        "mean_retrieval_latency_ms": rounded(mean_or_none(numeric_values(completed, "retrieval_latency_ms"))),
        "mean_answer_latency_ms": rounded(mean_or_none(numeric_values(completed, "answer_latency_ms"))),
        "mean_total_latency_ms": rounded(mean_or_none(numeric_values(completed, "total_latency_ms"))),
        "mean_llm_judge_faithfulness_score_1_to_5": rounded(mean_or_none(numeric_values(completed, "llm_judge_faithfulness_score_1_to_5"))),
        "mean_llm_judge_latency_ms": rounded(mean_or_none(numeric_values(completed, "llm_judge_latency_ms"))),
        "llm_judge_enabled": args.llm_judge,
        "important_note": "Evidence-overlap is a lexical proxy only. The optional local LLM judge is also a diagnostic, not ground truth; review low scores and unsupported claims manually.",
    }

    print("\n=== SUMMARY ===")
    for key, value in summary.items():
        if key not in {"generated_at_utc", "database_path", "important_note"}:
            print(f"{key}: {value}")
    print(f"\n{summary['important_note']}")
    if not args.llm_judge:
        print("Tip: add --llm-judge to request a local model-based faithfulness review for each answer.")

    if not args.no_save:
        output_stem = "evaluation_results" if not args.benchmark else f"evaluation_results_{Path(args.benchmark).stem}_{args.split}"
        json_path = PROJECT_DIR / f"{output_stem}.json"
        csv_path = PROJECT_DIR / f"{output_stem}.csv"
        json_path.write_text(json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                serializable = {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
                writer.writerow(serializable)
        print(f"\nSaved detailed results:\n- {csv_path}\n- {json_path}")

    return 0 if len(completed) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
