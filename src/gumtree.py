import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'Repo data')
REPO = os.getenv('REPO_NAME', 'kafka')

# Precompile regex patterns for performance
NODE_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+ \[label="(.+)", color=(red|blue)\];$')
EDGE_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+ -> n_[0-9]+_[0-9]+;$')
NODE_ID_PATTERN = re.compile(r'^n_[0-9]+_[0-9]+')


# =========================================================
# GumTree Runner
# =========================================================

class GumTreeDiff:
    def __init__(self):
        self.bin_path = os.path.join(BASE_PATH, 'gumtree-3.0.0', 'bin', 'gumtree.bat')
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

        command = f'"{self.bin_path}" dotdiff "{b_file}" "{a_file}"'

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
# Dataset Processor
# =========================================================

class ASTDatasetBuilder:

    def __init__(self, dataset_file, output_file, types=['.java']):
        self.dataset_file = os.path.join(data_path, dataset_file)
        self.output_file = os.path.join(data_path, output_file)
        self.output_backup_file = self.output_file + '.bak'
        self.progress_file = self.output_file.replace('.json', '_progress.json')
        self.progress_backup_file = self.progress_file + '.bak'
        self.types = types
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

    def run(self):

        # Load full dataset
        with open(self.dataset_file, 'r') as f:
            dataset = json.load(f)

        # Load existing AST output if exists (resume mode)
        if os.path.exists(self.output_file):
            self.ast_dict = self._safe_load_json(self.output_file, fallback={}, backup_path=self.output_backup_file)
            print(f"\nResuming from existing file. Already processed: {len(self.ast_dict)} commits\n")
        else:
            self.ast_dict = {}
            print("\nStarting fresh AST generation\n")

        self.load_progress()
        self.processed_commits.update(self.ast_dict.keys())
        already_done = self.processed_commits

        total = len(dataset)
        remaining_commits = [c for c in dataset.keys() if c not in already_done]

        print(f"Total commits in dataset: {total}")
        print(f"Already processed (including skipped): {len(already_done)}")
        print(f"Remaining to process: {len(remaining_commits)}\n")

        start_time = time.time()
        gumtree = GumTreeDiff()

        processed = 0

        for commit_hash in remaining_commits:
            commit_data = dataset[commit_hash]
            processed += 1
            commit_start = time.time()
            candidate_files = 0
            success_files = 0
            gumtree_failures = 0

            for file_tuple in commit_data["files"]:
                filepath, before, after = file_tuple

                if not filepath.endswith(tuple(self.types)):
                    continue
                candidate_files += 1

                try:
                    b_dot, a_dot = gumtree.get_dotfiles((filepath, before, after))
                except SyntaxError:
                    print("GumTree failed for:", filepath)
                    gumtree_failures += 1
                    continue

                b_subtree = SubTreeExtractor(b_dot).extract_subtree()
                a_subtree = SubTreeExtractor(a_dot).extract_subtree()

                if len(b_subtree[0]) == 0 and len(a_subtree[0]) == 0:
                    continue

                self.ast_dict.setdefault(commit_hash, []).append(
                    (filepath, b_subtree, a_subtree)
                )
                success_files += 1

            self.processed_commits.add(commit_hash)

            if success_files > 0:
                if commit_hash in self.skipped_commits:
                    del self.skipped_commits[commit_hash]
                print(f"{commit_hash[:7]} processed in {self.time_since(commit_start)}")
            else:
                if candidate_files == 0:
                    reason = 'no_supported_files'
                elif gumtree_failures > 0:
                    reason = 'gumtree_failed_or_empty'
                else:
                    reason = 'empty_subtree'
                self.skipped_commits[commit_hash] = reason

            # Save every 50 commits for safety
            if processed % 50 == 0:
                self._atomic_write_json(self.output_file, self.ast_dict, self.output_backup_file)
                self.save_progress()
                print(f"Checkpoint saved ({len(self.ast_dict)} commits)")

        # Final save
        self._atomic_write_json(self.output_file, self.ast_dict, self.output_backup_file)
        self.save_progress()

        print("\n==============================================")
        print(f"Finished processing remaining commits")
        print(f"Total commits stored: {len(self.ast_dict)}")
        print(f"Total commits skipped: {len(self.skipped_commits)}")
        print(f"Total time: {self.time_since(start_time)}")
        print("==============================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    builder = ASTDatasetBuilder(
        dataset_file=f"{REPO}_source_dataset.json",
        output_file=f"{REPO}_ast_subtrees.json",
        types=['.java']
    )
    builder.run()