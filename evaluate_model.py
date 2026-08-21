import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


OUTCOME_TO_VERDICT = {
    "Success": "GO",
    "Partial Success": "CONDITIONAL GO",
    "Failure": "NO-GO",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_split(args):
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "knowledge_base_train"
    eval_path = output_dir / "eval_cases.json"
    train_path = train_dir / "train_decisions.json"

    records = load_json(source_path)
    usable_records = [
        record for record in records
        if record.get("Outcome") in OUTCOME_TO_VERDICT
    ]

    rng = random.Random(args.seed)
    rng.shuffle(usable_records)

    eval_size = min(args.eval_size, len(usable_records))
    eval_records = usable_records[:eval_size]
    train_records = usable_records[eval_size:]

    write_json(train_path, train_records)
    write_json(eval_path, eval_records)

    print(f"Prepared evaluation split")
    print(f"Training records: {len(train_records)} -> {train_path}")
    print(f"Evaluation records: {len(eval_records)} -> {eval_path}")
    print()
    print("Next build the no-leakage evaluation vector DB:")
    print("$env:KNOWLEDGE_BASE_DIR='evaluation/knowledge_base_train'")
    print("$env:CHROMA_DB_PATH='./chroma_db_eval'")
    print("python build_db.py")


def make_question(record):
    problem = record.get('Problem_Statement', '').strip()
    goal = record.get('Business_Goal', '').strip()
    market = record.get('Market_Condition', '').strip()
    economic = record.get('Economic_Environment', '').strip()
    competitors = record.get('Competitor_Situation', '').strip()
    customers = record.get('Customer_Segment', '').strip()
    decision = record.get('Decision_Taken', '').strip()

    financials = (
        f"Financials: Budget available: {record.get('Budget_Available', 'unknown')}. "
        f"Revenue before decision: {record.get('Revenue_Before_Decision', 'unknown')}. "
        f"Monthly burn rate: {record.get('Burn_Rate', 'unknown')}. "
        f"Cash runway: {record.get('Cash_Runway', 'unknown')} months."
    )

    parts = []
    if problem:
        parts.append(f"Problem: {problem}")
    if goal:
        parts.append(f"Business Goal: {goal}")
    if market or economic:
        parts.append(f"Market & Economic Context: {market}, {economic}")
    if competitors:
        parts.append(f"Competitor Situation: {competitors}")
    if customers:
        parts.append(f"Customer Segment: {customers}")
    parts.append(financials)
    if decision:
        parts.append(f"\nDecision being considered: {decision}")
    parts.append("Should we go ahead with this decision?")

    return "\n".join(parts)


def parse_verdict(text):
    if not text:
        return "UNPARSED"

    match = re.search(
        r"(?:\*{0,3}|#*\s*)VERDICT(?:\*{0,3})\s*:\s*\[?(?:\*{0,2})(CONDITIONAL\s+GO|NO\s*-?\s*GO|GO)(?:\*{0,2})\]?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(CONDITIONAL\s+GO|NO\s*-?\s*GO|GO)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return "UNPARSED"

    verdict = re.sub(r"\s+", " ", match.group(1).upper()).strip()
    verdict = verdict.replace("NO GO", "NO-GO")
    return verdict


def parse_confidence(text):
    if not text:
        return None

    match = re.search(
        r"(?:\*{0,3}|#*\s*)CONFIDENCE(?:\s*SCORE)?(?:\*{0,3})\s*:\s*\[?(?:\*{0,2})(\d{1,3})\s*%?(?:\*{0,2})\]?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"(?:confidence|certainty)[^\d\n]*(\d{1,3})\s*%", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"(\d{1,3})\s*%", text)
            if not match:
                return None

    val = int(match.group(1))
    return max(0, min(100, val))


def run_evaluation(args):
    from main import run_second_opinion

    records = load_json(Path(args.eval_cases))
    if args.limit:
        records = records[:args.limit]

    results = []
    confusion = defaultdict(Counter)

    for index, record in enumerate(records, start=1):
        expected = OUTCOME_TO_VERDICT.get(record.get("Outcome"), "UNKNOWN")
        question = make_question(record)

        print("=" * 80)
        print(f"Case {index}/{len(records)}: {record.get('Decision_ID')} | expected={expected}")

        try:
            answer = run_second_opinion(question)
            predicted = parse_verdict(answer)
            confidence = parse_confidence(answer)
            error = ""
        except Exception as exc:
            answer = ""
            predicted = "ERROR"
            confidence = None
            error = str(exc)

        correct = predicted == expected
        confusion[expected][predicted] += 1

        results.append({
            "decision_id": record.get("Decision_ID", ""),
            "company": record.get("Company", ""),
            "domain": record.get("Domain", ""),
            "year": record.get("Year", ""),
            "category": record.get("Decision_Category", ""),
            "outcome": record.get("Outcome", ""),
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "correct": correct,
            "confidence": confidence,
            "error": error,
            "question": question,
            "model_answer": answer,
        })

        print(f"predicted={predicted} confidence={confidence} correct={correct}")
        if error:
            print(f"error={error}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    json_path = output_dir / "evaluation_results.json"
    summary_path = output_dir / "evaluation_summary.json"

    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    write_json(json_path, results)

    total = len(results)
    correct_count = sum(1 for result in results if result["correct"])
    confidence_values = [
        result["confidence"] for result in results
        if result["confidence"] is not None
    ]
    correct_confidence = [
        result["confidence"] for result in results
        if result["correct"] and result["confidence"] is not None
    ]
    incorrect_confidence = [
        result["confidence"] for result in results
        if not result["correct"] and result["confidence"] is not None
    ]

    summary = {
        "total_cases": total,
        "correct_cases": correct_count,
        "accuracy": round(correct_count / total, 4) if total else 0,
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values else None,
        "average_confidence_when_correct": round(sum(correct_confidence) / len(correct_confidence), 2)
        if correct_confidence else None,
        "average_confidence_when_incorrect": round(sum(incorrect_confidence) / len(incorrect_confidence), 2)
        if incorrect_confidence else None,
        "confusion_matrix": {
            expected: dict(predictions)
            for expected, predictions in confusion.items()
        },
    }
    write_json(summary_path, summary)

    print("=" * 80)
    print(f"Accuracy: {summary['accuracy'] * 100:.2f}% ({correct_count}/{total})")
    print(f"Average confidence: {summary['average_confidence']}")
    print(f"Results CSV: {csv_path}")
    print(f"Results JSON: {json_path}")
    print(f"Summary JSON: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SecondOpinion decision model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create train/evaluation split.")
    prepare_parser.add_argument(
        "--source",
        default="knowledge_base/entrepreneurial_decisions.json",
        help="Path to the full entrepreneurial decisions JSON dataset.",
    )
    prepare_parser.add_argument("--output-dir", default="evaluation")
    prepare_parser.add_argument("--eval-size", type=int, default=100)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(func=prepare_split)

    run_parser = subparsers.add_parser("run", help="Run model evaluation on holdout cases.")
    run_parser.add_argument("--eval-cases", default="evaluation/eval_cases.json")
    run_parser.add_argument("--output-dir", default="evaluation/results")
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.set_defaults(func=run_evaluation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
