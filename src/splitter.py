import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_PATH, 'Repo data')
REPO = os.getenv('REPO_NAME', 'kafka')

REQUIRED_METRIC_COLUMNS = [
    'la', 'ld', 'nf', 'nd', 'ns', 'ent',
    'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp'
]


def stratified_split(commit_ids: List[str], labels: List[int], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)

    positives = [c for c, y in zip(commit_ids, labels) if y == 1]
    negatives = [c for c, y in zip(commit_ids, labels) if y == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_bucket(bucket: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(bucket)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_part = bucket[:n_train]
        val_part = bucket[n_train:n_train + n_val]
        test_part = bucket[n_train + n_val:]
        return train_part, val_part, test_part

    p_train, p_val, p_test = split_bucket(positives)
    n_train, n_val, n_test = split_bucket(negatives)

    train_ids = p_train + n_train
    val_ids = p_val + n_val
    test_ids = p_test + n_test

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    return train_ids, val_ids, test_ids


def pr_based_split(commit_ids: List[str], commit_to_pr: Dict[str, int], test_prs: set, val_prs: set) -> Tuple[List[str], List[str], List[str]]:
    """Split commits based on PR membership in predefined test/val sets"""
    train_ids = []
    val_ids = []
    test_ids = []
    
    for commit_id in commit_ids:
        pr_num = commit_to_pr.get(commit_id)
        if pr_num is None:
            continue
            
        if pr_num in test_prs:
            test_ids.append(commit_id)
        elif pr_num in val_prs:
            val_ids.append(commit_id)
        else:
            train_ids.append(commit_id)
    
    return train_ids, val_ids, test_ids


def write_commit_csv(path: str, commit_ids: List[str]) -> None:
    pd.DataFrame({'commit_id': commit_ids}).to_csv(path, index=False)


def write_ast_json(path: str, commit_ids: List[str], ast_dict: Dict[str, list]) -> None:
    subset = {commit_id: ast_dict[commit_id] for commit_id in commit_ids if commit_id in ast_dict}
    with open(path, 'w') as fp:
        json.dump(subset, fp)


def write_train_chunks(data_dir: str, train_ids: List[str], ast_dict: Dict[str, list], chunk_size: int, prefix: str) -> List[str]:
    output_files = []
    if not train_ids:
        path = os.path.join(data_dir, f'{prefix}_train_1.json')
        with open(path, 'w') as fp:
            json.dump({}, fp)
        return [os.path.basename(path)]

    for index in range(0, len(train_ids), chunk_size):
        chunk_ids = train_ids[index:index + chunk_size]
        chunk_path = os.path.join(data_dir, f'{prefix}_train_{(index // chunk_size) + 1}.json')
        write_ast_json(chunk_path, chunk_ids, ast_dict)
        output_files.append(os.path.basename(chunk_path))
    return output_files


def build_metrics_csv(path: str, commit_ids: List[str]) -> None:
    metrics_df = pd.DataFrame({'commit_id': commit_ids})
    for col in REQUIRED_METRIC_COLUMNS:
        metrics_df[col] = 0.0
    metrics_df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build train/val/test split files from AST subtrees')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--ast-file', default=None)
    parser.add_argument('--source-dataset-file', default=None)
    parser.add_argument('--test-set-file', default=None)
    parser.add_argument('--val-set-file', default=None)
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--output-prefix', default=None)
    parser.add_argument('--use-pr-split', action='store_true', help='Use PR-based split instead of random stratified')
    args = parser.parse_args()

    args.ast_file = args.ast_file or f'{args.repo}_ast_subtrees.json'
    args.source_dataset_file = args.source_dataset_file or f'{args.repo}_source_dataset.json'
    args.test_set_file = args.test_set_file or f'{args.repo}_test_set.csv'
    args.val_set_file = args.val_set_file or f'{args.repo}_val_set.csv'
    args.output_prefix = args.output_prefix or args.repo

    data_dir = args.data_dir
    ast_path = os.path.join(data_dir, args.ast_file)
    source_dataset_path = os.path.join(data_dir, args.source_dataset_file)
    test_set_path = os.path.join(data_dir, args.test_set_file)
    val_set_path = os.path.join(data_dir, args.val_set_file)

    with open(ast_path, 'r') as fp:
        ast_dict = json.load(fp)

    with open(source_dataset_path, 'r') as fp:
        source_dataset = json.load(fp)

    # Load test and val PRs
    test_df = pd.read_csv(test_set_path)
    val_df = pd.read_csv(val_set_path)
    
    if 'pr_number' not in test_df.columns or args.label_col not in test_df.columns:
        raise ValueError(f'{args.test_set_file} must contain pr_number and {args.label_col} columns')
    if 'pr_number' not in val_df.columns or args.label_col not in val_df.columns:
        raise ValueError(f'{args.val_set_file} must contain pr_number and {args.label_col} columns')

    # Extract PR sets
    test_prs = set(test_df['pr_number'].dropna().astype(int))
    val_prs = set(val_df['pr_number'].dropna().astype(int))
    
    # Build PR to label mapping from both sets
    pr_to_label = {}
    for _, row in test_df[['pr_number', args.label_col]].dropna().iterrows():
        pr_to_label[int(row.pr_number)] = int(row[args.label_col])
    for _, row in val_df[['pr_number', args.label_col]].dropna().iterrows():
        pr_to_label[int(row.pr_number)] = int(row[args.label_col])

    # Build commit to PR mapping and collect labeled commits
    commit_ids = []
    commit_labels = []
    commit_to_pr = {}

    for commit_id in ast_dict.keys():
        commit_meta = source_dataset.get(commit_id)
        if commit_meta is None:
            continue

        pr_number = commit_meta.get('pr_number')
        if pr_number is None:
            continue

        label = pr_to_label.get(int(pr_number))
        if label is None:
            # For train set, we still need labels, so skip if not available
            continue

        commit_ids.append(commit_id)
        commit_labels.append(label)
        commit_to_pr[commit_id] = int(pr_number)

    if not commit_ids:
        raise RuntimeError('No commits could be labeled. Check PR label/source dataset inputs.')

    # Create output directory structure: data/kafka/train, data/kafka/val, data/kafka/test
    output_base = os.path.join(BASE_PATH, 'data', args.repo)
    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')
    test_dir = os.path.join(output_base, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save labels file in base directory
    labels_path = os.path.join(output_base, f'{args.output_prefix}_labels.json')
    with open(labels_path, 'w') as fp:
        json.dump({commit_id: label for commit_id, label in zip(commit_ids, commit_labels)}, fp)

    # Split commits based on PR membership
    if args.use_pr_split:
        train_ids, val_ids, test_ids = pr_based_split(commit_ids, commit_to_pr, test_prs, val_prs)
    else:
        train_ids, val_ids, test_ids = stratified_split(
            commit_ids=commit_ids,
            labels=commit_labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

    # Write commit lists to respective directories
    write_commit_csv(os.path.join(train_dir, f'{args.output_prefix}_train.csv'), train_ids)
    write_commit_csv(os.path.join(val_dir, f'{args.output_prefix}_valid.csv'), val_ids)
    write_commit_csv(os.path.join(test_dir, f'{args.output_prefix}_test.csv'), test_ids)

    # Write AST data to respective directories
    train_files = write_train_chunks(train_dir, train_ids, ast_dict, args.chunk_size, args.output_prefix)
    write_ast_json(os.path.join(val_dir, f'{args.output_prefix}_valid.json'), val_ids, ast_dict)
    write_ast_json(os.path.join(test_dir, f'{args.output_prefix}_test.json'), test_ids, ast_dict)

    # Write metrics file to base directory
    metrics_path = os.path.join(output_base, f'{args.output_prefix}_metrics_kamei.csv')
    build_metrics_csv(metrics_path, commit_ids)

    print('==============================================')
    print(f'Labeled commits: {len(commit_ids)}')
    print(f'Train/Val/Test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}')
    print(f'Train chunks: {len(train_files)} -> {train_files[:5]}')
    print(f'Output directory: {output_base}')
    print(f'Labels file: {labels_path}')
    print(f'Metrics file: {metrics_path}')
    print('==============================================')


if __name__ == '__main__':
    main()
    print('==============================================')


if __name__ == '__main__':
    main()
