import json
import os
import pickle
from typing import Dict

import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')
repo_data_path = os.path.join(BASE_PATH, 'Repo data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768


class ASTDataset(Dataset):
    def __init__(self, data_dict, commit_lists, metrics_file, special_token=True, transform=None):
        self.transform = transform
        self.special_token = special_token
        self.data_dict = data_dict
        self.commit_lists = commit_lists
        self.labels = self.load_labels(self.data_dict['labels'])
        self.ast_dict = None
        self.c_list = None
        self.file_index = 0
        self.mode = 'train'
        self.vectorizer_model = None
        self.metrics = None
        self.load_metrics(metrics_file)
        self.learn_vectorizer()

    @staticmethod
    def resolve_path(path_like: str) -> str:
        if os.path.isabs(path_like) and os.path.exists(path_like):
            return path_like
        trimmed = path_like.lstrip('/\\')
        candidates = [
            os.path.join(BASE_PATH, trimmed),
            os.path.join(data_path, trimmed),
            os.path.join(repo_data_path, trimmed),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return os.path.join(BASE_PATH, trimmed)

    def load_labels(self, labels_source: str) -> Dict[str, int]:
        label_path = self.resolve_path(labels_source)
        if label_path.endswith('.csv'):
            labels_df = pd.read_csv(label_path)
            return dict(zip(labels_df['commit_id'].astype(str), labels_df['label'].astype(int)))
        with open(label_path, 'r') as file:
            loaded = json.load(file)
        return {str(k): int(v) for k, v in loaded.items()}

    def load_metrics(self, metrics_file):
        if not metrics_file:
            self.metrics = pd.DataFrame(columns=['commit_id'])
            return
        metrics_path = self.resolve_path(metrics_file)
        self.metrics = pd.read_csv(metrics_path)
        self.metrics = self.metrics.drop(
            ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp', 'orexp', 'osexp', 'osawr', 'project',
             'buggy', 'fix'],
            axis=1, errors='ignore')
        kept_columns = ['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                        'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']
        available = [col for col in kept_columns if col in self.metrics.columns]
        self.metrics = self.metrics[available]
        if 'commit_id' not in self.metrics.columns:
            self.metrics['commit_id'] = []
        self.metrics = self.metrics.fillna(value=0)

    def learn_vectorizer(self):
        files = list(self.data_dict['train']) + list(self.data_dict['val'])
        corpus = []
        for fname in files:
            with open(self.resolve_path(fname), 'r') as fp:
                subtrees = json.load(fp)
            for commit, files in subtrees.items():
                for f in files:
                    for node_feat in f[1][0]:
                        if len(node_feat) > 1:  # None
                            corpus.append(node_feat)
                        else:
                            if not self.special_token:
                                corpus.append(node_feat[0])
                            else:
                                feature = node_feat[0]
                                if ':' in node_feat[0]:
                                    feat_type = node_feat[0].split(':')[0]
                                    feature = feat_type + ' ' + '<' + feat_type[
                                                                      :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
                                corpus.append(feature)
                    for node_feat in f[2][0]:
                        if len(node_feat) > 1:  # None
                            corpus.append(node_feat)
                        else:
                            if not self.special_token:
                                corpus.append(node_feat[0])
                            else:
                                feature = node_feat[0]
                                if ':' in node_feat[0]:
                                    feat_type = node_feat[0].split(':')[0]
                                    feature = feat_type + ' ' + '<' + feat_type[
                                                                      :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
                                corpus.append(feature)

        vectorizer = CountVectorizer(lowercase=False, max_features=100000, binary=True)
        self.vectorizer_model = vectorizer.fit(corpus)

        # with open(os.path.join(BASE_PATH, 'trained_models/vectorizer.pkl'), 'wb') as fp:
        #     pickle.dump(vectorizer, fp)

    def set_mode(self, mode):
        self.mode = mode
        commit_list_path = self.resolve_path(self.commit_lists[self.mode])
        self.c_list = pd.read_csv(commit_list_path)['commit_id'].astype(str).tolist()
        self.file_index = 0
        with open(self.resolve_path(self.data_dict[self.mode][self.file_index]), 'r') as fp:
            self.ast_dict = json.load(fp)

    def switch_datafile(self):
        self.file_index += 1
        self.file_index %= len(self.data_dict[self.mode])
        with open(self.resolve_path(self.data_dict[self.mode][self.file_index]), 'r') as fp:
            self.ast_dict = json.load(fp)

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_adjacency_matrix(self, n_nodes, src, dst):
        edges = np.array([src, dst]).T
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # add supernode
        adj = sp.vstack([adj, np.ones((1, adj.shape[1]), dtype=np.float32)])
        adj = sp.hstack([adj, np.zeros((adj.shape[0], 1), dtype=np.float32)])
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def get_embedding(self, file_node_tokens, colors):
        for i, node_feat in enumerate(file_node_tokens):
            file_node_tokens[i] = node_feat.strip()
            if node_feat == 'N o n e':
                file_node_tokens[i] = 'None'
                colors.insert(i, 'blue')
                assert colors[i] == 'blue'
            if self.special_token:
                if ':' in node_feat:
                    feat_type = node_feat.split(':')[0]
                    file_node_tokens[i] = feat_type + ' ' + '<' + feat_type[
                                                                  :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
        # fix the data later to remove the code above.
        features = self.vectorizer_model.transform(file_node_tokens).astype(np.float32)
        # add color feature at the end of features
        color_feat = [1 if c == 'red' else 0 for c in colors]
        features = sp.hstack([features, np.array(color_feat, dtype=np.float32).reshape(-1, 1)])
        # add supernode
        features = sp.hstack([features, np.zeros((features.shape[0], 1), dtype=np.float32)])
        supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
        supernode_feat[-1, -1] = 1
        features = sp.vstack([features, supernode_feat])
        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        return features

    def __len__(self):
        return len(self.c_list)

    def __getitem__(self, item):
        c = self.c_list[item]
        attempts = 0
        max_attempts = max(1, len(self.data_dict[self.mode]))
        while attempts < max_attempts:
            try:
                commit = self.ast_dict[c]
                break
            except KeyError:
                attempts += 1
                self.switch_datafile()
        else:
            print('commit {} not found in AST files, skip!'.format(c))
            return None
        label = self.labels[c]
        metric_columns = [col for col in self.metrics.columns if col != 'commit_id']
        if len(metric_columns) == 0:
            metrics = torch.FloatTensor([])
        else:
            metric_row = self.metrics[self.metrics['commit_id'].astype(str) == c][metric_columns].to_numpy(dtype=np.float32)
            if metric_row.shape[0] == 0:
                metrics = torch.zeros(len(metric_columns), dtype=torch.float)
            else:
                metrics = torch.FloatTensor(self.normalize(metric_row)[0, :])

        b_node_tokens, b_edges, b_colors = [], [[], []], []
        a_node_tokens, a_edges, a_colors = [], [[], []], []
        b_nodes_so_far, a_nodes_so_far = 0, 0
        for file in commit:
            b_node_tokens += [' '.join(node) for node in file[1][0]]
            b_colors += [c for c in file[1][2]]
            b_edges = [
                b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],   # source nodes
                b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]    # destination nodes
            ]
            a_node_tokens += [' '.join(node) for node in file[2][0]]
            a_colors += [c for c in file[2][2]]
            a_edges = [
                a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],   # source nodes
                a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]    # destination nodes
            ]

            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])
            b_nodes_so_far += b_n_nodes
            a_nodes_so_far += a_n_nodes

        if b_nodes_so_far + a_nodes_so_far > 29000 or b_nodes_so_far > 19000 or a_nodes_so_far > 19000:
            print('{} is a large commit, skip!'.format(c))
            return None

        before_embeddings = self.get_embedding(b_node_tokens, b_colors)
        before_adj = self.get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = self.get_embedding(a_node_tokens, a_colors)
        after_adj = self.get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])
        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, label, metrics]

        if not b_nodes_so_far and not a_nodes_so_far:
            print('commit {} has no file tensors.'.format(c))

        return training_data


if __name__ == "__main__":
    # ast_dataset = ASTDataset(data_path + '/asts_300_synerr.json')
    # print(ast_dataset[0])
    # train_loader = DataLoader(ast_dataset, batch_size=1, shuffle=False)
    # train_iter = iter(train_loader)
    # data = train_iter.next()
    print()


def prepare_kafka_pr_splits(
        commits_csv=os.path.join(repo_data_path, 'kafka_commits.csv'),
        test_set_csv=os.path.join(repo_data_path, 'kafka_test_set.csv'),
        val_set_csv=os.path.join(repo_data_path, 'kafka_val_set.csv'),
        ast_file=os.path.join(repo_data_path, 'kafka_ast_subtrees.json'),
        output_dir=os.path.join(repo_data_path, 'kafka')):
    os.makedirs(output_dir, exist_ok=True)

    with open(ast_file, 'r') as fp:
        ast_dict = json.load(fp)
    ast_commit_ids = set(str(commit_id) for commit_id in ast_dict.keys())

    commits_df = pd.read_csv(commits_csv)
    commits_df['commit_id'] = commits_df['commit_id'].astype(str)
    commits_df = commits_df[commits_df['commit_id'].isin(ast_commit_ids)].copy()

    test_prs = set(pd.read_csv(test_set_csv)['pr_number'].astype(int).tolist())
    val_prs = set(pd.read_csv(val_set_csv)['pr_number'].astype(int).tolist())

    test_df = commits_df[commits_df['pr_id'].isin(test_prs)].copy()
    val_df = commits_df[commits_df['pr_id'].isin(val_prs)].copy()
    train_df = commits_df[~commits_df['pr_id'].isin(test_prs.union(val_prs))].copy()

    train_file = os.path.join(output_dir, 'kafka_train_commits.csv')
    val_file = os.path.join(output_dir, 'kafka_val_commits.csv')
    test_file = os.path.join(output_dir, 'kafka_test_commits.csv')

    train_df[['commit_id', 'pr_id', 'label']].to_csv(train_file, index=False)
    val_df[['commit_id', 'pr_id', 'label']].to_csv(val_file, index=False)
    test_df[['commit_id', 'pr_id', 'label']].to_csv(test_file, index=False)

    labels_file = os.path.join(output_dir, 'kafka_labels.csv')
    commits_df[['commit_id', 'label']].to_csv(labels_file, index=False)

    print('Kafka split from AST-backed commits only: {}'.format(len(commits_df)))
    print('Train/Val/Test commits: {}/{}/{}'.format(len(train_df), len(val_df), len(test_df)))

    return {
        'train': train_file,
        'val': val_file,
        'test': test_file,
        'labels': labels_file,
    }