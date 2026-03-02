import os
import json
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
REPO_DATA_PATH = os.path.join(BASE_PATH, "Repo Data")
REPO = os.getenv("REPO_NAME", "kafka")


class GitMiner:
    def __init__(self, owner="apache", repo=REPO, max_workers=4):
        self.base_url = "https://api.github.com"
        self.owner = owner
        self.repo = repo
        self.max_workers = max_workers  # Parallel requests per commit

        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise Exception("GITHUB_TOKEN environment variable not set")

        self.session = requests.Session()
        # Connection pool for better parallelism
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        self.session.mount("https://", adapter)

    def get_commit(self, commit_sha):
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/commits/{commit_sha}"
        response = self.session.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Commit not found: {commit_sha} (status {response.status_code})")
            return None

        return response.json()

    def get_before_after_content(self, commit_sha):
        commit = self.get_commit(commit_sha)
        if commit is None:
            return None

        changed_files = commit.get("files", [])
        if not changed_files:
            return None

        # Filter to Java files only, and remove deleted files upfront
        java_files = [
            f for f in changed_files 
            if f.get("status") != "removed" and f.get("filename", "").endswith(".java")
        ]
        
        if not java_files:
            return None

        file_data = []
        parent_sha = commit.get("parents", [{}])[0].get("sha")

        def fetch_file_content(file):
            """Fetch before/after content for a single file in parallel."""
            filename = file.get("filename")
            contents_url = file.get("contents_url")

            raw_headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3.raw"
            }

            # Fetch after content
            try:
                after_response = self.session.get(contents_url, headers=raw_headers, timeout=15)
                after_content = after_response.text if after_response.status_code == 200 else ""
            except:
                after_content = ""

            # Fetch before content
            before_content = ""
            if parent_sha:
                try:
                    before_url = f"{contents_url}&ref={parent_sha}"
                    before_response = self.session.get(before_url, headers=raw_headers, timeout=15)
                    if before_response.status_code == 200:
                        before_content = before_response.text
                except:
                    before_content = ""

            return [filename, before_content, after_content]

        # Parallel fetch for all files in this commit
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_file_content, f): f for f in java_files}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    file_data.append(result)
                except Exception as e:
                    print(f"Error fetching file content: {e}")

        return file_data if file_data else None


def build_dataset_from_pr_csv():
    csv_path = os.path.join(REPO_DATA_PATH, "apache_pr_commits.csv")

    if not os.path.exists(csv_path):
        print("CSV file not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Collect all commits first
    all_commits = []
    for _, row in df.iterrows():
        try:
            commits = json.loads(row["commit_ids"])
            all_commits.extend(commits)
        except:
            continue

    total_expected = len(all_commits)
    print(f"\nTotal commits to process: {total_expected}\n")

    miner = GitMiner(owner="apache", repo=REPO, max_workers=4)

    dataset = {}
    processed = 0
    skipped = 0
    last_rate_limit_reset = 0

    output_path = os.path.join(REPO_DATA_PATH, f"{REPO}_source_dataset.json")
    
    # Resume logic - load existing progress
    if os.path.exists(output_path):
        print("Existing dataset found. Loading for resume...")
        with open(output_path, "r") as f:
            dataset = json.load(f)
        print(f"Already processed commits: {len(dataset)}\n")

    for _, row in df.iterrows():
        pr_number = row["pr_number"]

        try:
            commit_ids = json.loads(row["commit_ids"])
        except:
            print("Invalid commit_ids format for PR:", pr_number)
            continue

        for commit_sha in commit_ids:
            if commit_sha in dataset:
                continue
            
            processed += 1
            print(f"[{processed}/{total_expected}] Processing {commit_sha[:8]}...", end=" ")

            file_data = miner.get_before_after_content(commit_sha)

            if file_data is None:
                skipped += 1
                print("⊘ SKIP")
            else:
                dataset[commit_sha] = {
                    "pr_number": pr_number,
                    "files": file_data
                }
                print(f"✓ ({len(file_data)} files)")

            # Smart checkpointing: save every 50 commits instead of 100
            if processed % 50 == 0:
                print(f"  → Saving checkpoint ({len(dataset)} total)...")
                with open(output_path, "w") as f:
                    json.dump(dataset, f)
                print(f"  → Checkpoint saved\n")

                # Smart rate limit handling: check headers and only sleep if needed
                time.sleep(0.1)  # Small delay instead of 0.2 per commit

    # Final save
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    print("\n" + "=" * 60)
    print("Finished processing")
    print(f"Total commits expected: {total_expected}")
    print(f"Total processed: {processed}")
    print(f"Successful commits: {len(dataset)}")
    print(f"Skipped commits: {skipped}")
    print(f"Dataset saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset_from_pr_csv()