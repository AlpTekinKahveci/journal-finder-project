from pathlib import Path
import argparse
import html
import re
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "journal_recommender_tfidf_sgd.pkl"


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\\-\\+\\#\\.\\,\\;\\:\\(\\)\\[\\]\\/ ]+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def recommend_top5_journals(abstract_text: str, top_n: int = 5) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    cleaned_text = clean_text(abstract_text)

    probabilities = model.predict_proba([cleaned_text])[0]
    classes = model.named_steps["clf"].classes_

    top_indices = np.argsort(probabilities)[::-1][:top_n]

    return pd.DataFrame({
        "rank": range(1, top_n + 1),
        "journal_name": classes[top_indices],
        "score": probabilities[top_indices]
    })


def main():
    parser = argparse.ArgumentParser(
        description="Recommend top 5 computer science journals for a given article abstract."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Article abstract text."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file containing the article abstract."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of journals to recommend. Default is 5."
    )

    args = parser.parse_args()

    if args.file:
        abstract_text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        abstract_text = args.text
    else:
        print("Paste the article abstract below. Press Ctrl+D when finished:\n")
        abstract_text = sys.stdin.read()

    if not abstract_text.strip():
        print("Error: abstract text is empty.")
        sys.exit(1)

    results = recommend_top5_journals(abstract_text, top_n=args.top_n)

    print("\nTop Journal Recommendations")
    print("=" * 80)

    for _, row in results.iterrows():
        print(f"{int(row['rank'])}. {row['journal_name']}  |  score: {row['score']:.4f}")


if __name__ == "__main__":
    main()
