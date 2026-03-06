#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pymongo import MongoClient


BASE_PATH = os.path.dirname(os.path.dirname(__file__))
PR_COMMIT_ROOT = os.path.join(BASE_PATH, "pr_commit_data", "repo")
SPLITS = ["train", "val", "test"]


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_repo_name(repo: str) -> str:
    repo = repo.strip().lower()
    if repo == "wildlfy":
        return "wildfly"
    return repo


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, dict):
        if "$date" in value:
            return parse_datetime(value["$date"])
        if "$numberLong" in value:
            return parse_datetime(value["$numberLong"])
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def extract_pr_number(doc: Dict[str, Any]) -> Optional[int]:
    keys = [
        "pr_number",
        "pr_id",
        "pull_number",
        "pull_request_number",
        "number",
        "PR_Number",
        "PR_number",
        "_id",
    ]
    for key in keys:
        if key in doc:
            pr = parse_int(doc.get(key))
            if pr is not None:
                return pr
    return None


def extract_created_at(doc: Dict[str, Any]) -> Optional[datetime]:
    keys = [
        "createdat",
        "createdAt",
        "created_at",
        "created",
        "createdDate",
        "created_date",
        "CreatedAt",
        "createdOn",
    ]
    for key in keys:
        if key in doc:
            dt = parse_datetime(doc.get(key))
            if dt is not None:
                return dt
    for parent in ["pull_request", "pullRequest", "pr", "PR", "metadata"]:
        child = doc.get(parent)
        if isinstance(child, dict):
            for key in keys:
                if key in child:
                    dt = parse_datetime(child.get(key))
                    if dt is not None:
                        return dt
    return None


def candidate_db_names(repo: str) -> List[str]:
    nrepo = normalize_repo_name(repo)
    if nrepo in {"hibernate", "wildfly"}:
        return [f"prism_{nrepo}_test", f"apache_{repo}", f"apache_{nrepo}"]
    return [f"apache_{repo}", f"apache_{nrepo}"]


def candidate_prs_collections(repo: str) -> List[str]:
    nrepo = normalize_repo_name(repo)
    if nrepo in {"hibernate", "wildfly"}:
        return [
            f"PRs_{nrepo}_{nrepo}-orm",
            f"PRs_{nrepo}_{nrepo}",
            f"PR_Features_{nrepo}_{nrepo}-orm",
            f"PR_Features_{nrepo}_{nrepo}",
            "PRs",
            "PR_Features",
        ]
    return [
        f"PRs_apache_{repo}",
        f"PRs_apache_{nrepo}",
        f"PR_Features_apache_{repo}",
        f"PR_Features_apache_{nrepo}",
        f"PRs_{repo}",
        f"PRs_{nrepo}",
        "PRs",
        "PR_Features",
    ]


def find_existing_name(candidates: Iterable[str], existing: Set[str]) -> Optional[str]:
    for name in candidates:
        if name in existing:
            return name
    return None


def read_split_pr_numbers(path: str) -> Set[int]:
    prs: Set[int] = set()
    if not os.path.exists(path):
        return prs
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pr = parse_int(row.get("pr_number"))
            if pr is not None:
                prs.add(pr)
    return prs


def load_pr_created_map(collection, target_prs: Set[int]) -> Dict[int, datetime]:
    out: Dict[int, datetime] = {}
    if not target_prs:
        return out
    projection = {
        "_id": 1,
        "pr_number": 1,
        "pr_id": 1,
        "number": 1,
        "pull_request_number": 1,
        "createdat": 1,
        "createdAt": 1,
        "created_at": 1,
        "created": 1,
        "createdDate": 1,
        "created_date": 1,
        "CreatedAt": 1,
        "createdOn": 1,
        "pull_request": 1,
        "pullRequest": 1,
        "pr": 1,
        "PR": 1,
        "metadata": 1,
    }
    for doc in collection.find({}, projection):
        pr = extract_pr_number(doc)
        if pr is None or pr not in target_prs:
            continue
        dt = extract_created_at(doc)
        if dt is None:
            continue
        if pr not in out or dt < out[pr]:
            out[pr] = dt
    return out


def format_timeframe(prs: Set[int], created_map: Dict[int, datetime]) -> str:
    dates = [created_map[p] for p in prs if p in created_map]
    if not dates:
        return "NA"
    return f"{min(dates).date().isoformat()} -> {max(dates).date().isoformat()}"


def count_buggy_clean(path: str) -> Tuple[int, int]:
    buggy = 0
    clean = 0
    if not os.path.exists(path):
        return buggy, clean
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = parse_int(row.get("label"))
            if label == 1:
                buggy += 1
            elif label == 0:
                clean += 1
    return buggy, clean


def repos_from_input(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    repos = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    return sorted(normalize_repo_name(r) for r in repos)


def resolve_repo_collection(client: MongoClient, repo: str):
    db = None
    db_name = None
    for cand in candidate_db_names(repo):
        if cand in client.list_database_names():
            db_name = cand
            db = client[cand]
            break
    if db is None:
        return None, None
    col_name = find_existing_name(candidate_prs_collections(repo), set(db.list_collection_names()))
    if col_name is None:
        return None, None
    return db_name, db[col_name]


def collect_stats(root: str, client: MongoClient) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for repo in repos_from_input(root):
        repo_dir = os.path.join(root, repo)
        train_path = os.path.join(repo_dir, "train.csv")
        val_path = os.path.join(repo_dir, "val.csv")
        test_path = os.path.join(repo_dir, "test.csv")

        train_prs = read_split_pr_numbers(train_path)
        val_prs = read_split_pr_numbers(val_path)
        test_prs = read_split_pr_numbers(test_path)
        all_prs = train_prs | val_prs | test_prs

        db_name, prs_collection = resolve_repo_collection(client, repo)
        created_map = load_pr_created_map(prs_collection, all_prs) if prs_collection is not None else {}

        train_buggy, train_clean = count_buggy_clean(os.path.join(repo_dir, "train.csv"))
        val_buggy, val_clean = count_buggy_clean(os.path.join(repo_dir, "val.csv"))
        test_buggy, test_clean = count_buggy_clean(os.path.join(repo_dir, "test.csv"))
        rows.append(
            {
                "repo": repo,
                "train_buggy": train_buggy,
                "train_clean": train_clean,
                "train_timeframe": format_timeframe(train_prs, created_map),
                "val_buggy": val_buggy,
                "val_clean": val_clean,
                "val_timeframe": format_timeframe(val_prs, created_map),
                "test_buggy": test_buggy,
                "test_clean": test_clean,
                "test_timeframe": format_timeframe(test_prs, created_map),
                "mongo_source": db_name if db_name is not None else "NA",
            }
        )
    return rows


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "repo",
        "train_buggy",
        "train_clean",
        "train_timeframe",
        "val_buggy",
        "val_clean",
        "val_timeframe",
        "test_buggy",
        "test_clean",
        "test_timeframe",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-repo train/val/test buggy-clean statistics from pr_commit_data."
    )
    parser.add_argument("--input-root", default=PR_COMMIT_ROOT)
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        help='MongoDB URI for timeframe lookup (default: env MONGODB_URI or "mongodb://localhost:27017").',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MongoClient(args.mongo_uri)
    rows = collect_stats(args.input_root, client)
    client.close()
    if not rows:
        raise ValueError(f"No repositories found under: {args.input_root}")
    print_table(rows)


if __name__ == "__main__":
    main()
