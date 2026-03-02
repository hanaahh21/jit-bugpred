import json
import os
from collections import defaultdict

import pandas as pd


BASE_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_PATH, "Repo data")
REPO = os.getenv("REPO_NAME", "kafka")

REPO_TEST_SET = os.path.join(DATA_DIR, f"{REPO}_test_set.csv")
REPO_SOURCE_DATASET = os.path.join(DATA_DIR, f"{REPO}_source_dataset.json")


def main() -> None:
    if not os.path.exists(REPO_TEST_SET):
        raise FileNotFoundError(f"Missing file: {REPO_TEST_SET}")
    if not os.path.exists(REPO_SOURCE_DATASET):
        raise FileNotFoundError(f"Missing file: {REPO_SOURCE_DATASET}")

    test_df = pd.read_csv(REPO_TEST_SET, usecols=["pr_number"])
    test_prs = set(test_df["pr_number"].dropna().astype(int).tolist())

    with open(REPO_SOURCE_DATASET, "r") as f:
        source_dataset = json.load(f)

    dataset_commit_count = len(source_dataset)
    prs_in_dataset = set()
    commits_by_pr = defaultdict(int)

    for commit_sha, commit_payload in source_dataset.items():
        pr_number = commit_payload.get("pr_number")
        if pr_number is None:
            continue
        try:
            pr_number = int(pr_number)
        except (TypeError, ValueError):
            continue

        prs_in_dataset.add(pr_number)
        commits_by_pr[pr_number] += 1

    processed_prs = test_prs.intersection(prs_in_dataset)
    missing_prs = test_prs - prs_in_dataset

    commits_for_processed_test_prs = sum(commits_by_pr[pr] for pr in processed_prs)
    total_prs_in_source_dataset = len(prs_in_dataset)
    matched_prs_with_test_set = len(processed_prs)

    print("==============================================")
    print(f"Repo: {REPO}")
    print(f"Total PRs in {REPO}_test_set.csv: {len(test_prs)}")
    print(f"Total unique PRs in {REPO}_source_dataset.json: {total_prs_in_source_dataset}")
    print(f"Matching PRs (source_dataset ∩ test_set): {matched_prs_with_test_set}")
    print(f"PRs present in {REPO}_source_dataset.json: {len(processed_prs)}")
    print(f"PRs missing from {REPO}_source_dataset.json: {len(missing_prs)}")
    print("----------------------------------------------")
    print(f"Total commits in {REPO}_source_dataset.json: {dataset_commit_count}")
    print(f"Commits linked to processed PRs from test set: {commits_for_processed_test_prs}")
    print("==============================================")


if __name__ == "__main__":
    main()
