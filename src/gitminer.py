import argparse
import ast
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BASE_PATH = os.path.dirname(os.path.dirname(__file__))
INPUT_ROOT = os.path.join(BASE_PATH, "pr_commit_data")
OUTPUT_ROOT = os.path.join(BASE_PATH, "source_files")
CHECKPOINT_ROOT = os.path.join(BASE_PATH, "checkpoints_source_files")

SPLITS = ["train", "val", "test"]

# Keep these exact filenames for split outputs.
OUTPUT_FILE_BY_SPLIT = {
    "train": "train_sourcefiles.json",
    "val": "val_sourcefiles.json",
    "test": "test_sourcefiles.json",
}

# TEMP fallback tokens (requested). Remove these from code later.
# Add more keys here when needed.
# FALLBACK_GITHUB_TOKENS = //fetch from .env


def normalize_repo_name(repo_name: str) -> str:
    repo_name = repo_name.strip().lower()
    if repo_name == "wildlfy":
        return "wildfly"
    return repo_name


def github_owner_repo(repo_name: str) -> Tuple[str, str]:
    """
    Resolve GitHub owner/repo from the dataset repo name.
    """
    repo_name = normalize_repo_name(repo_name)
    if repo_name == "hibernate":
        return "hibernate", "hibernate-orm"
    if repo_name == "wildfly":
        return "wildfly", "wildfly"
    return "apache", repo_name


def resolve_split_csv_path(repo_name: str, split: str) -> str:
    """
    Support both:
    1) pr_commit_data/repo/<repo>/<split>.csv
    2) pr_commit_data/<repo>/<split>.csv
    """
    repo_name = normalize_repo_name(repo_name)
    candidates = [
        os.path.join(INPUT_ROOT, "repo", repo_name, f"{split}.csv"),
        os.path.join(INPUT_ROOT, repo_name, f"{split}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Missing input split file for repo='{repo_name}', split='{split}'. "
        f"Tried: {candidates}"
    )


def parse_commits_field(raw_value) -> List[str]:
    """
    Parse CSV 'commits' column that may be:
    - JSON list string
    - Python-list-like string
    - already a list
    """
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return [str(x).strip() for x in raw_value if str(x).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [text]


def build_commit_list_from_split_csv(csv_path: str) -> List[Tuple[str, int]]:
    """
    Return ordered list of (commit_sha, pr_number), deduplicated by commit_sha.
    """
    df = pd.read_csv(csv_path, usecols=["pr_number", "commits"])
    commit_items: List[Tuple[str, int]] = []
    seen = set()

    for row in df.itertuples(index=False):
        pr_number = int(row.pr_number)
        commit_shas = parse_commits_field(row.commits)
        for sha in commit_shas:
            sha = sha.strip()
            if not sha or sha in seen:
                continue
            seen.add(sha)
            commit_items.append((sha, pr_number))

    return commit_items


def latest_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    def checkpoint_number(path: str) -> int:
        match = re.search(r"checkpoint_(\d+)\.json$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    files.sort(key=checkpoint_number)
    return files[-1]


def checkpoint_paths_desc(checkpoint_dir: str) -> List[str]:
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.json")
    files = glob.glob(pattern)
    if not files:
        return []

    def checkpoint_number(path: str) -> int:
        match = re.search(r"checkpoint_(\d+)\.json$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    files.sort(key=checkpoint_number, reverse=True)
    return files


def safe_load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Skipping invalid JSON file: {path} ({exc})")
        return None


def keep_last_n_checkpoints(checkpoint_dir: str, keep_n: int = 3) -> None:
    """
    Keep only the newest `keep_n` checkpoint files for the split.
    """
    paths = checkpoint_paths_desc(checkpoint_dir)
    for old_path in paths[keep_n:]:
        try:
            os.remove(old_path)
        except OSError as exc:
            print(f"[WARN] Could not remove old checkpoint {old_path}: {exc}")


def remove_all_checkpoints(checkpoint_dir: str) -> None:
    """
    Remove all checkpoint files for a split (used after successful completion).
    """
    for path in checkpoint_paths_desc(checkpoint_dir):
        try:
            os.remove(path)
        except OSError as exc:
            print(f"[WARN] Could not remove checkpoint {path}: {exc}")


class GitMiner:
    def __init__(self, owner: str, repo: str, token: str, max_workers: int = 5):
        self.base_url = "https://api.github.com"
        self.owner = owner
        self.repo = repo
        self.primary_token = token
        self.active_token = token
        self.fallback_tokens = [t for t in FALLBACK_GITHUB_TOKENS if t and t.strip()]
        self.max_workers = max_workers

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        self.session.mount("https://", adapter)

    def _auth_headers(self, accept: str) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.active_token}",
            "Accept": accept,
        }

    def _rate_limit_remaining(self, token: str) -> int:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        try:
            response = self.session.get(f"{self.base_url}/rate_limit", headers=headers, timeout=15)
            if response.status_code != 200:
                return 0
            return int(response.headers.get("X-RateLimit-Remaining", "0"))
        except requests.RequestException:
            return 0

    def _pick_available_fallback(self, exclude_token: Optional[str] = None) -> Optional[str]:
        for token in self.fallback_tokens:
            if exclude_token and token == exclude_token:
                continue
            if self._rate_limit_remaining(token) > 0:
                return token
        return None

    def _switch_to_fallback(self, exclude_token: Optional[str] = None) -> bool:
        token = self._pick_available_fallback(exclude_token=exclude_token)
        if token is None:
            return False
        if self.active_token != token:
            print("Switching to available fallback token until next checkpoint...")
            self.active_token = token
        return True

    def _primary_token_available(self) -> bool:
        return self._rate_limit_remaining(self.primary_token) > 0

    def checkpoint_token_rebalance(self) -> None:
        """
        At checkpoint boundaries:
        - if currently on fallback, attempt to switch back to primary when available.
        """
        if self.active_token != self.primary_token:
            if self._primary_token_available():
                self.active_token = self.primary_token
                print("Primary token available again. Switched back to argument token.")
            else:
                # If current fallback is exhausted, try another fallback.
                if self._rate_limit_remaining(self.active_token) <= 0:
                    switched = self._switch_to_fallback(exclude_token=self.active_token)
                    if switched:
                        print("Primary still limited. Switched to another fallback token.")
                    else:
                        print("Primary and all fallback tokens are rate-limited.")
                else:
                    print("Primary token still rate-limited. Continuing with active fallback token.")

    def _request_with_retries(
        self,
        url: str,
        headers: Dict[str, str],
        timeout: int = 30,
        max_retries: int = 5,
    ) -> Optional[requests.Response]:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, headers=headers, timeout=timeout)

                # Respect GitHub rate-limit resets.
                if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                    # Try token failover first (primary->fallbacks, fallback->other fallback).
                    switched = self._switch_to_fallback(exclude_token=self.active_token)
                    if switched:
                        headers["Authorization"] = f"token {self.active_token}"
                        continue

                    # No token available right now: wait until reset of current token.
                    reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
                    wait_seconds = max(reset_ts - int(time.time()), 0) + 2
                    print(f"All known tokens are rate-limited. Waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    if attempt < max_retries:
                        time.sleep(attempt * 2)
                        continue
                return response
            except requests.RequestException:
                if attempt < max_retries:
                    time.sleep(attempt * 2)
                    continue
                return None
        return None

    def get_commit(self, commit_sha: str) -> Optional[Dict]:
        headers = self._auth_headers("application/vnd.github.v3+json")
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits/{commit_sha}"
        response = self._request_with_retries(url, headers=headers, timeout=30)
        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else "no_response"
            print(f"Commit fetch failed: {commit_sha} (status={status})")
            return None
        return response.json()

    def _fetch_raw_file(self, contents_url: str, ref_sha: Optional[str]) -> str:
        if not contents_url:
            return ""

        # Commit API usually provides a contents_url without ref for current state.
        # For "before" version we append ref=<parent_sha>.
        url = contents_url
        if ref_sha:
            sep = "&" if "?" in contents_url else "?"
            url = f"{contents_url}{sep}ref={ref_sha}"

        raw_headers = self._auth_headers("application/vnd.github.v3.raw")
        response = self._request_with_retries(url, headers=raw_headers, timeout=20)
        if response is None or response.status_code != 200:
            return ""
        return response.text

    def get_before_after_content(self, commit_sha: str) -> Optional[List[List[str]]]:
        """
        For one commit, return:
        [
          [filename, before_content, after_content],
          ...
        ]
        """
        commit = self.get_commit(commit_sha)
        if commit is None:
            return None

        changed_files = commit.get("files", [])
        if not changed_files:
            return None

        # Keep only Java files and skip removed files (removed has no "after").
        files_to_fetch = [
            f for f in changed_files
            if f.get("status") != "removed"
            and str(f.get("filename", "")).lower().endswith(".java")
        ]
        if not files_to_fetch:
            return None

        parent_sha = None
        parents = commit.get("parents", [])
        if parents:
            parent_sha = parents[0].get("sha")

        def fetch_file(file_obj: Dict) -> List[str]:
            filename = file_obj.get("filename", "")
            contents_url = file_obj.get("contents_url", "")
            after_content = self._fetch_raw_file(contents_url=contents_url, ref_sha=None)
            before_content = self._fetch_raw_file(contents_url=contents_url, ref_sha=parent_sha) if parent_sha else ""
            return [filename, before_content, after_content]

        file_data: List[List[str]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fetch_file, f) for f in files_to_fetch]
            for future in as_completed(futures):
                try:
                    file_data.append(future.result())
                except Exception as exc:
                    print(f"File fetch error in commit {commit_sha[:8]}: {exc}")

        return file_data if file_data else None


def load_resume_dataset(output_path: str, checkpoint_dir: str) -> Tuple[Dict, int]:
    """
    Resume priority:
    1) latest checkpoint_<n>.json in split checkpoint directory
    2) existing final output json
    3) empty dataset
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try checkpoints from newest to oldest; skip corrupted/empty files.
    for checkpoint_path in checkpoint_paths_desc(checkpoint_dir):
        dataset = safe_load_json(checkpoint_path)
        if dataset is None:
            continue
        match = re.search(r"checkpoint_(\d+)\.json$", os.path.basename(checkpoint_path))
        checkpoint_idx = int(match.group(1)) if match else 0
        print(f"Resuming from checkpoint: {checkpoint_path} (commits={len(dataset)})")
        return dataset, checkpoint_idx

    # Fallback to final output json if valid.
    if os.path.exists(output_path):
        dataset = safe_load_json(output_path)
        if dataset is not None:
            print(f"Resuming from existing output: {output_path} (commits={len(dataset)})")
            return dataset, 0

    return {}, 0


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic write to reduce chance of partial/corrupt JSON on interruption.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    os.replace(tmp_path, path)


def process_split(
    miner: GitMiner,
    repo_name: str,
    split: str,
    checkpoint_every: int,
) -> None:
    csv_path = resolve_split_csv_path(repo_name, split)
    commit_items = build_commit_list_from_split_csv(csv_path)

    output_dir = os.path.join(OUTPUT_ROOT, normalize_repo_name(repo_name))
    output_path = os.path.join(output_dir, OUTPUT_FILE_BY_SPLIT[split])

    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, normalize_repo_name(repo_name), split)
    dataset, checkpoint_idx = load_resume_dataset(output_path=output_path, checkpoint_dir=checkpoint_dir)
    # On resume, keep only latest 3 checkpoints to save space.
    keep_last_n_checkpoints(checkpoint_dir, keep_n=3)

    total = len(commit_items)
    processed_now = 0
    skipped_now = 0
    checkpoint_count = checkpoint_idx

    print(f"\n[{normalize_repo_name(repo_name)}:{split}] Total commits in CSV: {total}")
    print(f"[{normalize_repo_name(repo_name)}:{split}] Already in resume dataset: {len(dataset)}")

    for idx, (commit_sha, pr_number) in enumerate(commit_items, start=1):
        if commit_sha in dataset:
            continue

        print(f"[{split}] {idx}/{total} commit={commit_sha[:8]} pr={pr_number}", end=" ... ")
        file_data = miner.get_before_after_content(commit_sha)
        processed_now += 1

        if file_data is None:
            skipped_now += 1
            print("SKIP")
        else:
            # Keep output structure identical to prior script.
            dataset[commit_sha] = {
                "pr_number": pr_number,
                "files": file_data,
            }
            print(f"OK ({len(file_data)} files)")

        if processed_now % checkpoint_every == 0:
            checkpoint_count += 1
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_count}.json")
            save_json(checkpoint_path, dataset)
            keep_last_n_checkpoints(checkpoint_dir, keep_n=3)
            print(f"[{split}] checkpoint saved: {checkpoint_path} (commits={len(dataset)})")
            miner.checkpoint_token_rebalance()

    save_json(output_path, dataset)
    # Split completed successfully, so checkpoints for this split are no longer needed.
    remove_all_checkpoints(checkpoint_dir)
    print(f"[{split}] removed all checkpoints in: {checkpoint_dir}")
    print(f"[{split}] output saved: {output_path} (commits={len(dataset)})")
    print(f"[{split}] processed_now={processed_now}, skipped_now={skipped_now}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch before/after source files for commits from PR split CSVs with checkpoint resume."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repo name: kafka|flink|hadoop|hbase|beam|camel|wildlfy|wildfly|hibernate",
    )
    parser.add_argument(
        "--github-token",
        required=True,
        help="GitHub API token passed as CLI flag.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel file-fetch workers per commit (default: 4).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint after every N processed commits (default: 50).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Splits to process (default: train val test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_name = normalize_repo_name(args.repo)
    owner, repo = github_owner_repo(repo_name)

    miner = GitMiner(
        owner=owner,
        repo=repo,
        token=args.github_token,
        max_workers=args.max_workers,
    )

    print(f"GitHub target: https://github.com/{owner}/{repo}")
    for split in args.splits:
        process_split(
            miner=miner,
            repo_name=repo_name,
            split=split,
            checkpoint_every=args.checkpoint_every,
        )

    print(f"\nCompleted all requested splits for repo {repo_name}.")


if __name__ == "__main__":
    main()
