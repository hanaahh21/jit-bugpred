import argparse
import json
import os
import re
import time
from typing import List, Optional, Tuple

import pandas as pd
import requests


PR_URL_PATTERN = re.compile(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)")


def load_github_token() -> Optional[str]:
    env_token = os.getenv("GITHUB_TOKEN")
    if env_token:
        return env_token.strip().strip('"').strip("'")

    candidate_paths = [
        os.path.join(".env"),
        os.path.join("src", ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() != "GITHUB_TOKEN":
                    continue
                cleaned = value.strip().strip('"').strip("'")
                if cleaned:
                    return cleaned

    return None


def parse_pr_url(url: str) -> Optional[Tuple[str, str, int]]:
    if not isinstance(url, str):
        return None
    match = PR_URL_PATTERN.match(url.strip())
    if not match:
        return None
    owner, repo, pr_number = match.groups()
    return owner, repo, int(pr_number)


def get_pr_commits(session: requests.Session, owner: str, repo: str, pr_number: int, token: Optional[str],
                   max_retries: int = 5) -> List[str]:
    commits: List[str] = []
    page = 1

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits"
        params = {"per_page": 100, "page": page}

        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, headers=headers, params=params, timeout=60)
                if response.status_code == 200:
                    break

                if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                    reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
                    now = int(time.time())
                    wait_seconds = max(reset_ts - now, 0) + 5
                    print(f"Rate limit hit. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue

                if attempt < max_retries:
                    wait_time = attempt * 3
                    print(f"PR {pr_number}: Retry {attempt}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, Exception) as e:
                if attempt < max_retries:
                    wait_time = attempt * 3
                    print(f"PR {pr_number}: Network error (attempt {attempt}/{max_retries}), retry in {wait_time}s: {type(e).__name__}")
                    time.sleep(wait_time)
                else:
                    print(f"PR {pr_number}: Failed after {max_retries} attempts: {type(e).__name__}")
                    return []

        if response is None:
            raise RuntimeError(f"No response while fetching PR #{pr_number}")

        if response.status_code == 404:
            return commits

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch commits for PR #{pr_number} in {owner}/{repo}. "
                f"Status: {response.status_code}, Body: {response.text[:300]}"
            )

        payload = response.json()
        if not payload:
            break

        commits.extend(item["sha"] for item in payload if "sha" in item)
        page += 1

    return commits


def load_completed_prs(output_csv: str) -> pd.DataFrame:
    if not os.path.exists(output_csv):
        return pd.DataFrame(columns=["pr_number", "commit_ids"])
    completed = pd.read_csv(output_csv)
    if "pr_number" not in completed.columns or "commit_ids" not in completed.columns:
        raise ValueError(f"Existing output file {output_csv} does not have required columns")
    return completed[["pr_number", "commit_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract commit IDs for each PR and store as two columns")
    parser.add_argument(
        "--input",
        default=os.path.join("Repo data", "ML_Label_Input_apache_kafka.csv"),
        help="Input CSV path containing pr_number and github_pr_url",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("Repo data", "apache_pr_commits.csv"),
        help="Output CSV path with columns: pr_number, commit_ids",
    )
    parser.add_argument(
        "--token",
        default=load_github_token(),
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Write progress to disk every N PRs",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, usecols=["pr_number", "github_pr_url"])
    df = df.dropna(subset=["pr_number", "github_pr_url"]).drop_duplicates(subset=["pr_number"])

    parsed_rows = []
    for _, row in df.iterrows():
        parsed = parse_pr_url(row["github_pr_url"])
        if parsed is None:
            continue
        owner, repo, parsed_pr = parsed
        parsed_rows.append(
            {
                "pr_number": int(row["pr_number"]),
                "owner": owner,
                "repo": repo,
                "parsed_pr_number": parsed_pr,
            }
        )

    prs_df = pd.DataFrame(parsed_rows)
    completed_df = load_completed_prs(args.output)
    completed_prs = set(completed_df["pr_number"].astype(int).tolist())

    pending = prs_df[~prs_df["pr_number"].isin(completed_prs)]

    print(f"Total PRs: {len(prs_df)}")
    print(f"Already completed: {len(completed_df)}")
    print(f"Pending: {len(pending)}")

    session = requests.Session()
    new_rows = []

    try:
        for idx, row in enumerate(pending.itertuples(index=False), start=1):
            pr_number = int(row.pr_number)
            owner = row.owner
            repo = row.repo

            commit_ids = get_pr_commits(session, owner, repo, pr_number, token=args.token)
            new_rows.append(
                {
                    "pr_number": pr_number,
                    "commit_ids": json.dumps(commit_ids),
                }
            )

            if idx % args.flush_every == 0:
                out_df = pd.concat([completed_df, pd.DataFrame(new_rows)], ignore_index=True)
                out_df = out_df.drop_duplicates(subset=["pr_number"], keep="last").sort_values("pr_number")
                out_df.to_csv(args.output, index=False)
                print(f"Saved progress: {len(out_df)} PRs")

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(pending)} pending PRs")

        final_df = pd.concat([completed_df, pd.DataFrame(new_rows)], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["pr_number"], keep="last").sort_values("pr_number")
        final_df.to_csv(args.output, index=False)
        print(f"Done. Output saved to: {args.output}")

    except Exception as exc:
        partial_df = pd.concat([completed_df, pd.DataFrame(new_rows)], ignore_index=True)
        if not partial_df.empty:
            partial_df = partial_df.drop_duplicates(subset=["pr_number"], keep="last").sort_values("pr_number")
            partial_df.to_csv(args.output, index=False)
            print(f"Partial progress saved to: {args.output}")
        raise exc


if __name__ == "__main__":
    main()
