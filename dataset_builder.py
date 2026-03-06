#!/usr/bin/env python3
"""
Build/update pr_commit_data/repo/<repo>/{train,val,test}.csv using:
1) Existing rows from pr_commit_timesplit/<repo>/*_new.csv first.
2) Top-up from MongoDB only when split/label target count is not satisfied.

Targets per repo:
- train: buggy=560, clean=560
- val: buggy=120, clean=120
- test: buggy=120, clean=120

Time windows (strict):
- train: createdAt < 2021-01-01
- val:   2021-01-01 <= createdAt <= 2022-12-31
- test:  createdAt >= 2023-01-01
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from pymongo import MongoClient
except ImportError as exc:
    raise SystemExit("pymongo is required. Install with: pip install pymongo") from exc


DEFAULT_REPOS = [
    "kafka",
    "flink",
    "hadoop",
    "hbase",
    "beam",
    "camel",
    "wildlfy",
    "hibernate",
]

TARGETS = {
    "train": {1: 560, 0: 560},
    "val": {1: 120, 0: 120},
    "test": {1: 120, 0: 120},
}

# Time windows requested by user:
# train: before 2020-12-31 (implemented as < 2021-01-01)
# val:   2021-01-01 .. 2021-12-31 (inclusive)
# test:  after 2022-01-01 (implemented as >= 2022-01-01)
TRAIN_END_EXCL = datetime(2021, 1, 1, tzinfo=timezone.utc)
VAL_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
VAL_END = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


def normalize_repo_name(repo: str) -> str:
    repo = repo.strip().lower()
    if repo == "wildlfy":
        return "wildfly"
    return repo


def parse_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        return None
    if isinstance(value, dict):
        if "$numberInt" in value:
            return parse_int(value["$numberInt"])
        if "$numberLong" in value:
            return parse_int(value["$numberLong"])
    return None


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


def in_timeframe(split: str, created_at: datetime) -> bool:
    if split == "train":
        return created_at < TRAIN_END_EXCL
    if split == "val":
        return VAL_START <= created_at <= VAL_END
    if split == "test":
        return created_at >= TEST_START
    return False


def parse_commits_cell(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    return []


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
            val = parse_int(doc.get(key))
            if val is not None:
                return val
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


def extract_commits(doc: Dict[str, Any]) -> List[str]:
    commit_keys = [
        "commits",
        "commit_ids",
        "commit_shas",
        "commit_sha",
        "sha_list",
        "sha",
        "Commits",
        "Commit_SHA",
    ]
    raw = None
    for key in commit_keys:
        if key in doc:
            raw = doc.get(key)
            break
    return parse_commits_cell(raw)


def candidate_db_names(repo: str) -> List[str]:
    nrepo = normalize_repo_name(repo)
    if nrepo in {"hibernate", "wildfly"}:
        return [f"prism_{nrepo}_test", f"apache_{repo}", f"apache_{nrepo}"]
    return [f"apache_{repo}", f"apache_{nrepo}"]


def candidate_features_collections(repo: str) -> List[str]:
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
        "PR_Features",
        f"PR_Features_apache_{repo}",
        f"PR_Features_apache_{nrepo}",
        f"PR_Features_{repo}",
        f"PR_Features_{nrepo}",
        "PRs",
        f"PRs_apache_{repo}",
        f"PRs_apache_{nrepo}",
    ]


def candidate_buggy_collections(repo: str) -> List[str]:
    nrepo = normalize_repo_name(repo)
    if nrepo in {"hibernate", "wildfly"}:
        return [f"Buggy_PRs_{nrepo}_{nrepo}-orm", f"Buggy_PRs_{nrepo}_{nrepo}", "Buggy_PRs"]
    return [f"Buggy_PRs_apache_{repo}", f"Buggy_PRs_apache_{nrepo}", "Buggy_PRs"]


def candidate_nonbuggy_collections(repo: str) -> List[str]:
    nrepo = normalize_repo_name(repo)
    if nrepo in {"hibernate", "wildfly"}:
        return [f"NonBuggy_PRs_{nrepo}_{nrepo}-orm", f"NonBuggy_PRs_{nrepo}_{nrepo}", "NonBuggy_PRs"]
    return [f"NonBuggy_PRs_apache_{repo}", f"NonBuggy_PRs_apache_{nrepo}", "NonBuggy_PRs"]


def find_existing_name(candidates: Iterable[str], existing: Set[str]) -> Optional[str]:
    for name in candidates:
        if name in existing:
            return name
    return None


def load_label_prs(collection) -> Set[int]:
    out: Set[int] = set()
    projection = {"_id": 1, "pr_number": 1, "pr_id": 1, "number": 1, "pull_request_number": 1}
    for doc in collection.find({}, projection):
        pr = extract_pr_number(doc)
        if pr is not None:
            out.add(pr)
    return out


def build_feature_info_map(collection) -> Dict[int, Tuple[List[str], datetime]]:
    info: Dict[int, Tuple[List[str], datetime]] = {}
    projection = {
        "_id": 1,
        "pr_number": 1,
        "pr_id": 1,
        "number": 1,
        "pull_request_number": 1,
        "commits": 1,
        "commit_ids": 1,
        "commit_shas": 1,
        "commit_sha": 1,
        "sha_list": 1,
        "sha": 1,
        "Commits": 1,
        "Commit_SHA": 1,
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
        if pr is None:
            continue
        commits = extract_commits(doc)
        created = extract_created_at(doc)
        if not commits or created is None:
            continue
        # Keep the instance with more commits for stability if duplicates exist.
        if pr not in info or len(commits) > len(info[pr][0]):
            info[pr] = (commits, created)
    return info


def load_timesplit_rows(path: str, label: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            pr = parse_int(row.get("pr_number"))
            row_label = parse_int(row.get("label"))
            commits = parse_commits_cell(row.get("commits"))
            if pr is None or row_label != label or not commits:
                continue
            rows.append({"pr_number": pr, "commits": commits, "label": label})
    rows.sort(key=lambda x: x["pr_number"])
    return rows


def filter_rows_by_timeframe(
    rows: List[Dict[str, Any]],
    split: str,
    feature_info: Dict[int, Tuple[List[str], datetime]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        pr = row["pr_number"]
        info = feature_info.get(pr)
        if info is None:
            continue
        _, created = info
        if in_timeframe(split, created):
            filtered.append(row)
    return filtered


def choose_from_timesplit(timesplit_rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    # Requirement: if too many, keep the last N by pr_number.
    if len(timesplit_rows) <= target:
        return timesplit_rows
    return timesplit_rows[-target:]


def topup_from_mongo(
    split: str,
    label: int,
    need: int,
    label_prs: Set[int],
    feature_info: Dict[int, Tuple[List[str], datetime]],
    taken_prs: Set[int],
) -> List[Dict[str, Any]]:
    if need <= 0:
        return []
    candidates: List[Tuple[int, List[str]]] = []
    for pr in label_prs:
        if pr in taken_prs:
            continue
        info = feature_info.get(pr)
        if info is None:
            continue
        commits, created = info
        if not in_timeframe(split, created):
            continue
        candidates.append((pr, commits))
    candidates.sort(key=lambda x: x[0])
    selected = candidates[-need:] if len(candidates) >= need else candidates
    return [{"pr_number": pr, "commits": commits, "label": label} for pr, commits in selected]


def write_split_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows_sorted = sorted(rows, key=lambda x: x["pr_number"])
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["pr_number", "commits", "label"])
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    "pr_number": row["pr_number"],
                    "commits": json.dumps(row["commits"]),
                    "label": row["label"],
                }
            )


def process_repo(client: MongoClient, repo: str, out_root: str, timesplit_root: str) -> None:
    nrepo = normalize_repo_name(repo)
    db = None
    db_name = None
    for cand in candidate_db_names(repo):
        if cand in client.list_database_names():
            db = client[cand]
            db_name = cand
            break
    if db is None:
        raise ValueError(f"Could not find database for repo '{repo}'. Tried: {candidate_db_names(repo)}")

    existing = set(db.list_collection_names())
    buggy_col = find_existing_name(candidate_buggy_collections(repo), existing)
    clean_col = find_existing_name(candidate_nonbuggy_collections(repo), existing)
    feat_col = find_existing_name(candidate_features_collections(repo), existing)
    if not buggy_col or not clean_col or not feat_col:
        raise ValueError(
            f"Repo '{repo}' in db '{db_name}' missing collections: "
            f"buggy={buggy_col}, clean={clean_col}, features={feat_col}"
        )

    buggy_prs = load_label_prs(db[buggy_col])
    clean_prs = load_label_prs(db[clean_col])
    feature_info = build_feature_info_map(db[feat_col])

    repo_timesplit = os.path.join(timesplit_root, nrepo)
    repo_out = os.path.join(out_root, "repo", nrepo)
    os.makedirs(repo_out, exist_ok=True)

    taken_prs: Set[int] = set()
    summary: Dict[str, Dict[int, int]] = {s: {1: 0, 0: 0} for s in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        split_rows: List[Dict[str, Any]] = []
        for label, label_prs in [(1, buggy_prs), (0, clean_prs)]:
            target = TARGETS[split][label]
            timesplit_path = os.path.join(repo_timesplit, f"{split}_new.csv")
            ts_rows = load_timesplit_rows(timesplit_path, label)
            # Enforce strict timeframe even if timesplit files were generated using different windows.
            ts_rows = filter_rows_by_timeframe(ts_rows, split, feature_info)
            ts_rows = [r for r in ts_rows if r["pr_number"] not in taken_prs]
            selected = choose_from_timesplit(ts_rows, target)
            for r in selected:
                taken_prs.add(r["pr_number"])

            need = target - len(selected)
            topped = topup_from_mongo(split, label, need, label_prs, feature_info, taken_prs)
            for r in topped:
                taken_prs.add(r["pr_number"])

            final_rows = selected + topped
            split_rows.extend(final_rows)
            summary[split][label] = len(final_rows)

            if len(final_rows) < target:
                print(
                    f"[WARN] {nrepo}/{split} label={label}: target={target}, got={len(final_rows)} "
                    f"(deficit={target - len(final_rows)})"
                )

        write_split_csv(os.path.join(repo_out, f"{split}.csv"), split_rows)

    print(
        f"[{nrepo}] "
        f"train(buggy={summary['train'][1]}, clean={summary['train'][0]}) | "
        f"val(buggy={summary['val'][1]}, clean={summary['val'][0]}) | "
        f"test(buggy={summary['test'][1]}, clean={summary['test'][0]})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update pr_commit_data using pr_commit_timesplit + MongoDB top-up, "
            "with strict timeframe windows and fixed split/label counts."
        )
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        help='MongoDB URI (default: env MONGODB_URI or "mongodb://localhost:27017")',
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_REPOS,
        help=f"Repo list (default: {' '.join(DEFAULT_REPOS)})",
    )
    parser.add_argument(
        "--out-dir",
        default="pr_commit_data",
        help="Output root directory (default: pr_commit_data).",
    )
    parser.add_argument(
        "--timesplit-root",
        default="pr_commit_timesplit",
        help="Input timesplit root (default: pr_commit_timesplit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MongoClient(args.mongo_uri)
    try:
        for repo in args.repos:
            process_repo(
                client=client,
                repo=repo,
                out_root=args.out_dir,
                timesplit_root=args.timesplit_root,
            )
    finally:
        client.close()

    print(f"Dataset update complete. Output: {args.out_dir}/repo/<repo>/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
