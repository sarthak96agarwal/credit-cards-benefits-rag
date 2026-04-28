# eval/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs RAGAS evaluation against your test dataset and saves results to CSV
# and Weights & Biases.
#
# Usage:
#   python eval/evaluate.py
#   python eval/evaluate.py --phase "Phase 3 - Score Gap 0.35"
#   python eval/evaluate.py --compare   # print comparison of all past runs
#   python eval/evaluate.py --upload-history  # upload all past CSVs to W&B
# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from ragas import evaluate, EvaluationDataset, SingleTurnSample
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LCOpenAIEmbeddings
from dotenv import load_dotenv
import wandb

from query import load_query_engine, query as rag_query
from eval.test_dataset import TEST_QUESTIONS

load_dotenv()

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "cc-benefits-rag")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", None)

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_to_wandb(df: pd.DataFrame, phase_label: str, means: dict):
    """Log a completed eval run to Weights & Biases."""
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=phase_label or "unlabeled",
        config={"phase": phase_label},
        reinit=True,
    )

    # Overall summary metrics
    wandb.log({f"overall/{m}": means[m] for m in METRIC_NAMES if m in means})

    # Per-category metrics
    df_cat = df.groupby("category")[METRIC_NAMES].mean()
    for category, row in df_cat.iterrows():
        for m in METRIC_NAMES:
            wandb.log({f"{category}/{m}": row[m]})

    # Per-question detail table (includes retrieved chunks for debugging)
    cols = ["user_input", "response", "reference", "card", "category",
            "retrieved_chunks", "retrieved_scores"] + METRIC_NAMES
    cols = [c for c in cols if c in df.columns]
    table = wandb.Table(dataframe=df[cols].reset_index(drop=True))
    wandb.log({"eval_results": table})

    # Per-category bar chart summary
    cat_table = wandb.Table(dataframe=df_cat.round(3).reset_index())
    wandb.log({"category_breakdown": cat_table})

    run.finish()
    print(f"   W&B run: {run.url}\n")


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(phase_label: str = ""):
    print(f"\n{'='*60}")
    print(f"  RAGAS Evaluation — {phase_label or 'Unlabeled run'}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    engine = load_query_engine()  # returns (index, all_nodes)

    # ── Step 1: Run every test question through your RAG pipeline ─────────────
    print(f"⏳ Running {len(TEST_QUESTIONS)} questions through RAG pipeline...\n")

    samples = []
    cards, categories, retrieved_chunks, retrieved_scores = [], [], [], []

    for i, item in enumerate(TEST_QUESTIONS):
        q = item["question"]
        print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {q}")

        result = rag_query(q, engine)

        samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=[src["text"] for src in result["sources"]],
            reference=item["ground_truth"],
        ))
        cards.append(item["card"])
        categories.append(item["category"])

        # Store chunks as "chunk1_text | chunk2_text | ..." for the W&B table
        chunks_str = "\n---\n".join(
            f"[{src['card_name']} | p{src['page']} | score={src['score']}]\n{src['text']}"
            for src in result["sources"]
        )
        retrieved_chunks.append(chunks_str)
        retrieved_scores.append(str([src["score"] for src in result["sources"]]))

        print(f"           → {result['answer'][:80]}...")
        print(f"           → {len(result['sources'])} chunks retrieved\n")

    # ── Step 2: Build RAGAS dataset ───────────────────────────────────────────
    ragas_dataset = EvaluationDataset(samples=samples)

    # ── Step 3: Run RAGAS ─────────────────────────────────────────────────────
    print("⏳ Running RAGAS scoring (this calls the OpenAI API)...\n")

    results = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        embeddings=LCOpenAIEmbeddings(model="text-embedding-3-small"),
        raise_exceptions=False,
    )

    # ── Step 4: Print summary ─────────────────────────────────────────────────
    df = results.to_pandas()
    means = {m: df[m].mean() for m in METRIC_NAMES if m in df.columns}

    print(f"\n{'='*60}")
    print(f"  Results — {phase_label}")
    print(f"{'='*60}")
    print(f"  Faithfulness       : {means.get('faithfulness', float('nan')):.3f}  (is answer grounded in chunks?)")
    print(f"  Answer Relevancy   : {means.get('answer_relevancy', float('nan')):.3f}  (does answer address the question?)")
    print(f"  Context Recall     : {means.get('context_recall', float('nan')):.3f}  (did retrieval find the right info?)")
    print(f"  Context Precision  : {means.get('context_precision', float('nan')):.3f}  (were retrieved chunks useful?)")
    print(f"{'='*60}\n")

    # ── Step 5: Save detailed results to CSV ──────────────────────────────────
    df["card"] = cards
    df["category"] = categories
    df["retrieved_chunks"] = retrieved_chunks
    df["retrieved_scores"] = retrieved_scores
    df["phase"] = phase_label
    df["timestamp"] = datetime.now().isoformat()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    phase_slug = phase_label.lower().replace(" ", "_").replace("-", "").replace("/", "") or "run"
    csv_path = RESULTS_DIR / f"{timestamp}_{phase_slug}.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Detailed results saved to: {csv_path}")
    print("   Open it to see per-question scores and spot where your RAG fails.\n")

    # ── Step 6: Per-category breakdown ────────────────────────────────────────
    print("Per-category breakdown:")
    df_cat = df.groupby("category")[METRIC_NAMES].mean()
    print(df_cat.round(3).to_string())
    print()

    # ── Step 7: Log to W&B ────────────────────────────────────────────────────
    print("⏳ Logging to Weights & Biases...")
    log_to_wandb(df, phase_label, means)

    return results, df


# ── Compare past runs ─────────────────────────────────────────────────────────

def compare_runs():
    """Print a comparison table of all saved eval runs."""
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        print("No eval runs found yet. Run `python eval/evaluate.py` first.")
        return

    rows = []
    for csv in csvs:
        df = pd.read_csv(csv)
        rows.append({
            "phase": df["phase"].iloc[0],
            "timestamp": df["timestamp"].iloc[0][:16],
            "faithfulness": df["faithfulness"].mean().round(3),
            "answer_relevancy": df["answer_relevancy"].mean().round(3),
            "context_recall": df["context_recall"].mean().round(3),
            "context_precision": df["context_precision"].mean().round(3),
        })

    comparison = pd.DataFrame(rows)
    print("\n📊 Evaluation History\n")
    print(comparison.to_string(index=False))
    print()


# ── Upload historical CSVs to W&B ────────────────────────────────────────────

def upload_history():
    """Upload all existing eval CSVs to W&B as separate runs."""
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        print("No eval CSVs found.")
        return

    print(f"Uploading {len(csvs)} runs to W&B...\n")
    for csv in csvs:
        df = pd.read_csv(csv)
        raw = df["phase"].iloc[0] if "phase" in df.columns else None
        phase_label = str(raw) if raw and str(raw) != "nan" else csv.stem
        means = {m: df[m].mean() for m in METRIC_NAMES if m in df.columns}
        print(f"  {phase_label}")
        log_to_wandb(df, phase_label, means)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="", help="Label for this run e.g. 'Phase 3 - Score Gap 0.35'")
    parser.add_argument("--compare", action="store_true", help="Print comparison of all past runs")
    parser.add_argument("--upload-history", action="store_true", help="Upload all past CSVs to W&B")
    args = parser.parse_args()

    if args.compare:
        compare_runs()
    elif args.upload_history:
        upload_history()
    else:
        run_evaluation(phase_label=args.phase)
