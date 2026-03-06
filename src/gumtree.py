import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'Repo data')
REPO = os.getenv('REPO_NAME', 'kafka')
SOURCE_FILES_ROOT = os.path.join(BASE_PATH, 'source_files')
AST_SUBTREES_ROOT = os.path.join(BASE_PATH, 'ast_subtrees')

# Precompile regex patterns for performance
NODE_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+ \[label="(.+)", color=(red|blue)\];$')
EDGE_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+ -> n_[0-9]+_[0-9]+;$')
NODE_ID_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+')


# =========================================================
# GumTree Runner
# =========================================================

class GumTreeDiff:
    def __init__(self):
        # Use platform-appropriate GumTree launcher.
        bin_name = 'gumtree.bat' if os.name == 'nt' else 'gumtree'
        self.bin_path = os.path.join(BASE_PATH, 'gumtree-3.0.0', 'bin', bin_name)
        self.src_dir = os.path.join(data_path, 'src')
        os.makedirs(self.src_dir, exist_ok=True)

    def get_diff(self, fname, b_content, a_content):
        fname = fname.split('/')[-1]
        name, ext = fname.rsplit('.', 1)

        b_file = os.path.join(self.src_dir, f"{name}_b.{ext}")
        a_file = os.path.join(self.src_dir, f"{name}_a.{ext}")

        with open(b_file, 'w', encoding='utf-8') as f:
            f.write(b_content)

        with open(a_file, 'w', encoding='utf-8') as f:
            f.write(a_content)

        # Use list-based command (no shell=True) for better performance
        process = subprocess.Popen(
            [self.bin_path, 'dotdiff', b_file, a_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output, error = process.communicate()

        if process.returncode != 0:
            print("GumTree failed:")
            print(error.decode())
            return None

        return output.decode('utf-8', errors='replace')

    def get_dotfiles(self, file_tuple):
        dot = self.get_diff(file_tuple[0], file_tuple[1], file_tuple[2])
        if dot is None:
            raise SyntaxError()

        lines = dot.splitlines()

        dotfiles = {'before': [], 'after': []}
        current = 'before'

        for l in lines:
            if l.strip() == 'subgraph cluster_dst {':
                current = 'after'
                continue

            # Use precompiled regex patterns for better performance
            if NODE_PATTERN.match(l) or EDGE_PATTERN.match(l):
                dotfiles[current].append(l)

        return dotfiles['before'], dotfiles['after']


# =========================================================
# Subtree Extraction
# =========================================================

class SubTreeExtractor:
    def __init__(self, dot):
        self.dot = dot
        self.red_nodes = []
        self.node_dict = {}
        self.from_to = {}
        self.to_from = {}
        self.subtree_nodes = set()
        self.subtree_edges = set()

    def read_ast(self):
        for line in self.dot:
            # Use precompiled regex for better performance
            node_match = NODE_PATTERN.match(line)
            edge_match = EDGE_PATTERN.match(line)

            if node_match:
                # Extract node ID using precompiled pattern
                node_id_match = NODE_ID_PATTERN.match(line)
                node_id = node_id_match.group() if node_id_match else line.split(' ')[0]
                label = re.split(r'\[[0-9]+', node_match.group(1))[0]
                color = node_match.group(2)

                self.node_dict[node_id] = label
                if color == 'red':
                    self.red_nodes.append(node_id)

            elif edge_match:
                src, dst = line.replace(';', '').split(' -> ')
                self.from_to.setdefault(src, []).append(dst)
                self.to_from.setdefault(dst, []).append(src)

    def extract_subtree(self):
        self.read_ast()

        for n in self.red_nodes:
            self.subtree_nodes.add(n)

            for d in self.from_to.get(n, []):
                self.subtree_nodes.add(d)
                self.subtree_edges.add((n, d))

            for s in self.to_from.get(n, []):
                self.subtree_nodes.add(s)
                self.subtree_edges.add((s, n))
                for d in self.from_to.get(s, []):
                    self.subtree_nodes.add(d)
                    self.subtree_edges.add((s, d))

        vs = list(self.subtree_nodes)
        es = list(self.subtree_edges)

        features = [[self.node_dict.get(node_id, 'unknown')] for node_id in vs]
        colors = ['red' if node_id in self.red_nodes else 'blue' for node_id in vs]
        # Precompute node index map for O(1) lookup instead of O(n) list.index()
        node_to_idx = {node_id: i for i, node_id in enumerate(vs)}
        edges = [[], []]
        for src, dst in es:
            edges[0].append(node_to_idx[src])
            edges[1].append(node_to_idx[dst])

        return features, edges, colors


# =========================================================
# Parallel Processing Helper
# =========================================================

def process_single_commit(args):
    """Process a single commit - designed to be called in parallel"""
    commit_hash, commit_data, types = args
    
    gumtree = GumTreeDiff()
    commit_files = []
    file_skip_reasons = []
    candidate_files = 0
    success_files = 0
    gumtree_failures = 0
    
    for file_tuple in commit_data["files"]:
        filepath, before, after = file_tuple
        
        if types and not filepath.endswith(tuple(types)):
            file_skip_reasons.append((filepath, 'unsupported_file_type'))
            continue
        candidate_files += 1
        
        try:
            b_dot, a_dot = gumtree.get_dotfiles((filepath, before, after))
        except SyntaxError:
            file_skip_reasons.append((filepath, 'gumtree_failed'))
            gumtree_failures += 1
            continue
        except Exception:
            file_skip_reasons.append((filepath, 'gumtree_error'))
            gumtree_failures += 1
            continue
        
        try:
            b_subtree = SubTreeExtractor(b_dot).extract_subtree()
            a_subtree = SubTreeExtractor(a_dot).extract_subtree()
            
            if len(b_subtree[0]) == 0 and len(a_subtree[0]) == 0:
                file_skip_reasons.append((filepath, 'empty_subtree'))
                continue
            
            commit_files.append((filepath, b_subtree, a_subtree))
            success_files += 1
        except Exception:
            file_skip_reasons.append((filepath, 'subtree_extraction_error'))
            gumtree_failures += 1
            continue
    
    # Determine skip reason if no files succeeded
    skip_reason = None
    if success_files == 0:
        if candidate_files == 0:
            skip_reason = 'no_supported_files'
        elif gumtree_failures > 0:
            skip_reason = 'gumtree_failed_or_empty'
        else:
            skip_reason = 'empty_subtree'
    
    return commit_hash, commit_files, skip_reason, file_skip_reasons


# =========================================================
# Dataset Processor
# =========================================================

class ASTDatasetBuilder:

    def __init__(self, dataset_file, output_file, checkpoint_dir, types=None):
        self.dataset_file = dataset_file
        self.output_file = output_file
        self.output_backup_file = self.output_file + '.bak'
        self.progress_file = self.output_file.replace('.json', '_progress.json')
        self.progress_backup_file = self.progress_file + '.bak'
        self.checkpoint_dir = checkpoint_dir
        self.types = types or []
        self.ast_dict = {}
        self.processed_commits = set()
        self.skipped_commits = {}

        Path("logs/").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.INFO,
            handlers=[RotatingFileHandler('logs/ast_builder.log', maxBytes=5_000_000, backupCount=3)]
        )

    @staticmethod
    def time_since(start):
        s = time.time() - start
        m = int(s // 60)
        s = s % 60
        return f"{m} min {s:.2f} sec"

    def _safe_load_json(self, path, fallback, backup_path=None):
        if not os.path.exists(path):
            return fallback

        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"Warning: Corrupted JSON detected: {path} ({exc})")

            if backup_path and os.path.exists(backup_path):
                try:
                    with open(backup_path, 'r') as f:
                        recovered = json.load(f)
                    print(f"Recovered from backup: {backup_path}")
                    return recovered
                except json.JSONDecodeError:
                    pass

            corrupt_path = path + '.corrupt'
            try:
                os.replace(path, corrupt_path)
                print(f"Moved corrupted file to: {corrupt_path}")
            except OSError:
                pass
            return fallback

    def _atomic_write_json(self, path, payload, backup_path=None):
        temp_path = path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(payload, f)

        if backup_path and os.path.exists(path):
            try:
                shutil.copyfile(path, backup_path)
            except OSError:
                pass

        os.replace(temp_path, path)

    def load_progress(self):
        progress = self._safe_load_json(
            self.progress_file,
            fallback={'processed_commits': [], 'skipped_commits': {}},
            backup_path=self.progress_backup_file
        )

        self.processed_commits = set(progress.get('processed_commits', []))
        self.skipped_commits = progress.get('skipped_commits', {})

    def save_progress(self):
        progress = {
            'processed_commits': list(self.processed_commits),
            'skipped_commits': self.skipped_commits
        }
        self._atomic_write_json(self.progress_file, progress, self.progress_backup_file)

    def _checkpoint_number(self, path):
        match = re.search(r'checkpoint_(\d+)\.json$', os.path.basename(path))
        return int(match.group(1)) if match else -1

    def _checkpoint_paths_desc(self):
        if not os.path.isdir(self.checkpoint_dir):
            return []
        files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.json'))
        files.sort(key=self._checkpoint_number, reverse=True)
        return files

    def _keep_last_n_checkpoints(self, keep_n=3):
        for old_path in self._checkpoint_paths_desc()[keep_n:]:
            try:
                os.remove(old_path)
            except OSError as exc:
                print(f"Warning: could not remove old AST checkpoint {old_path}: {exc}")

    def _save_ast_checkpoint(self, checkpoint_no):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{checkpoint_no}.json')
        payload = {
            'ast_dict': self.ast_dict,
            'processed_commits': list(self.processed_commits),
            'skipped_commits': self.skipped_commits,
        }
        self._atomic_write_json(checkpoint_path, payload)
        self._keep_last_n_checkpoints(keep_n=3)
        return checkpoint_path

    def _load_latest_ast_checkpoint(self):
        for path in self._checkpoint_paths_desc():
            payload = self._safe_load_json(path, fallback=None)
            if payload is None:
                continue
            if isinstance(payload, dict) and 'ast_dict' in payload:
                self.ast_dict = payload.get('ast_dict', {})
                self.processed_commits = set(payload.get('processed_commits', []))
                self.skipped_commits = payload.get('skipped_commits', {})
            else:
                # Backward compatibility: old checkpoint format was raw ast_dict.
                self.ast_dict = payload if isinstance(payload, dict) else {}
                self.load_progress()
            checkpoint_no = self._checkpoint_number(path)
            print(f"Resuming from AST checkpoint: {path}")
            self._keep_last_n_checkpoints(keep_n=3)
            return checkpoint_no
        return 0

    def _load_resume_state_from_output_progress(self):
        resume_files_exist = any(
            os.path.exists(path)
            for path in [
                self.output_file,
                self.output_backup_file,
                self.progress_file,
                self.progress_backup_file,
            ]
        )
        if not resume_files_exist:
            return False

        output_payload = self._safe_load_json(
            self.output_file,
            fallback=None,
            backup_path=self.output_backup_file
        )
        ast_dict = output_payload if isinstance(output_payload, dict) else {}

        progress_payload = self._safe_load_json(
            self.progress_file,
            fallback=None,
            backup_path=self.progress_backup_file
        )
        if isinstance(progress_payload, dict):
            progress_processed = set(progress_payload.get('processed_commits', []))
            progress_skipped = progress_payload.get('skipped_commits', {})
            if not isinstance(progress_skipped, dict):
                progress_skipped = {}
        else:
            progress_processed = set()
            progress_skipped = {}

        ast_commits = set(ast_dict.keys())
        skipped_commits = dict(progress_skipped)
        processed_commits = set(ast_commits)
        processed_commits.update(skipped_commits.keys())

        # Keep only trustworthy "processed" entries:
        # commits with AST output and commits explicitly tracked as skipped.
        stale_progress_only = progress_processed - processed_commits
        if stale_progress_only:
            print(
                f"Ignoring {len(stale_progress_only)} stale processed-only commits "
                f"not present in AST output/skipped metadata."
            )

        self.ast_dict = ast_dict
        self.skipped_commits = skipped_commits
        self.processed_commits = processed_commits

        print(
            f"Resuming from existing output/progress files "
            f"(ast={len(self.ast_dict)}, skipped={len(self.skipped_commits)})."
        )
        return True

    def _clear_all_checkpoints(self):
        for path in self._checkpoint_paths_desc():
            try:
                os.remove(path)
            except OSError as exc:
                print(f"Warning: could not delete AST checkpoint {path}: {exc}")

    def run(self, max_workers=4):

        # Load full dataset
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)

        # Resume from latest AST checkpoint when available.
        start_checkpoint = self._load_latest_ast_checkpoint()
        if start_checkpoint == 0:
            resumed = self._load_resume_state_from_output_progress()
            if resumed:
                print("\nResuming AST generation with parallel processing\n")
            else:
                self.ast_dict = {}
                self.processed_commits = set()
                self.skipped_commits = {}
                print("\nNo checkpoints or AST/progress files found. Starting fresh AST generation.\n")
        else:
            print("\nResuming AST generation with parallel processing\n")

        total = len(dataset)
        commits_to_process = [c for c in dataset.keys() if c not in self.processed_commits]

        print(f"Total commits in dataset: {total}")
        print(f"Pending commits: {len(commits_to_process)}")
        print(f"Workers: {max_workers}\n")

        start_time = time.time()
        processed = len(self.processed_commits)
        skipped = len(self.skipped_commits)
        checkpoint_count = start_checkpoint

        # Prepare arguments for parallel processing
        process_args = [
            (commit_hash, dataset[commit_hash], self.types)
            for commit_hash in commits_to_process
        ]

        # Process commits in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_commit, args): args[0]
                for args in process_args
            }
            
            for future in as_completed(futures):
                commit_hash = futures[future]
                processed += 1
                
                try:
                    commit_hash, commit_files, skip_reason, file_skip_reasons = future.result()
                    skipped_file_count = len(file_skip_reasons)
                    reason_counts = {}
                    for _, reason in file_skip_reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    reason_summary = ", ".join(
                        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
                    )
                    
                    if skip_reason:
                        self.skipped_commits[commit_hash] = skip_reason
                        skipped += 1
                        print(
                            f"[{processed}/{total}] {commit_hash[:7]} skipped. "
                            f"files {skipped_file_count} skipped. reason: {skip_reason}"
                        )
                        if reason_summary:
                            print(f"[{processed}/{total}] {commit_hash[:7]} file-skip reasons: {reason_summary}")
                    else:
                        self.ast_dict[commit_hash] = commit_files
                        print(f"[{processed}/{total}] {commit_hash[:7]} ✓ ({len(commit_files)} files)")
                        if skipped_file_count > 0:
                            print(
                                f"[{processed}/{total}] {commit_hash[:7]} files {skipped_file_count} skipped. "
                                f"reason: {reason_summary}"
                            )
                    
                    self.processed_commits.add(commit_hash)
                    
                    # Save every 50 commits for safety
                    if processed % 50 == 0:
                        self._atomic_write_json(self.output_file, self.ast_dict, self.output_backup_file)
                        self.save_progress()
                        checkpoint_count += 1
                        checkpoint_path = self._save_ast_checkpoint(checkpoint_count)
                        elapsed = self.time_since(start_time)
                        rate = processed / (time.time() - start_time)
                        eta = (total - processed) / rate if rate > 0 else 0
                        eta_str = f"{int(eta//60)}m {int(eta%60)}s"
                        print(f"Checkpoint: {len(self.ast_dict)} commits | Skipped: {skipped} | ETA: {eta_str}")
                        print(f"Saved AST checkpoint: {checkpoint_path}")
                        
                except Exception as e:
                    print(f"[{processed}/{total}] {commit_hash[:7]} ✗ Error: {str(e)}")
                    self.skipped_commits[commit_hash] = f'error: {str(e)}'
                    self.processed_commits.add(commit_hash)

        # Final save
        self._atomic_write_json(self.output_file, self.ast_dict, self.output_backup_file)
        self.save_progress()
        # Split done; cleanup checkpoints for disk usage.
        self._clear_all_checkpoints()
        print(f"Cleared AST checkpoints in: {self.checkpoint_dir}")

        print("\n==============================================")
        print(f"Finished processing all commits")
        print(f"Total commits stored: {len(self.ast_dict)}")
        print(f"Total commits skipped: {len(self.skipped_commits)}")
        print(f"Total time: {self.time_since(start_time)}")
        print("==============================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    def normalize_repo_name(repo_name):
        repo_name = repo_name.strip().lower()
        if repo_name == 'wildlfy':
            return 'wildfly'
        return repo_name

    def resolve_input_file(repo_name, split):
        repo_dir = os.path.join(SOURCE_FILES_ROOT, repo_name)
        candidates = {
            'train': ['train_sourcefiles.json'],
            'val': ['val_sourcefiles.json'],
            'test': ['test_source_Files.json', 'test_source_files.json', 'test_sourcefiles.json'],
        }[split]

        for fname in candidates:
            fpath = os.path.join(repo_dir, fname)
            if os.path.exists(fpath):
                return fpath
        raise FileNotFoundError(
            f"Missing input source file for repo='{repo_name}', split='{split}'. "
            f"Tried: {[os.path.join(repo_dir, n) for n in candidates]}"
        )

    parser = argparse.ArgumentParser(
        description='Generate AST subtrees from source_files/<repo>/<split>_sourcefiles.json'
    )
    parser.add_argument(
        '--repo',
        default=REPO,
        help='Repository name (e.g., kafka, hadoop, hibernate, wildlfy/wildfly)',
    )
    args = parser.parse_args()

    repo_name = normalize_repo_name(args.repo)
    import multiprocessing
    max_workers = min(2, multiprocessing.cpu_count())  # Use up to 8 cores
    
    print(f"Starting AST extraction for repo '{repo_name}' with {max_workers} parallel workers...\n")

    output_repo_dir = os.path.join(AST_SUBTREES_ROOT, repo_name)
    os.makedirs(output_repo_dir, exist_ok=True)

    split_to_output = {
        'train': 'train_astsub.json',
        'val': 'val_astsub.json',
        'test': 'test_astsub.json',
    }

    for split in ['train', 'val', 'test']:
        dataset_file = resolve_input_file(repo_name, split)
        output_file = os.path.join(output_repo_dir, split_to_output[split])
        checkpoint_dir = os.path.join(BASE_PATH, 'checkpoints_source_files', 'ast_subtrees', repo_name, split)
        print(f"[{split}] input={dataset_file}")
        print(f"[{split}] output={output_file}")

        builder = ASTDatasetBuilder(
            dataset_file=dataset_file,
            output_file=output_file,
            checkpoint_dir=checkpoint_dir,
            # Process only Java files.
            types=['.java']
        )
        builder.run(max_workers=max_workers)
