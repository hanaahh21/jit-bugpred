import argparse
import os

import torch

from datasets import ASTDataset, prepare_kafka_pr_splits
from models import JITGNN
from train import test

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


def run_prediction(dataset_name='kafka'):
    if dataset_name != 'kafka':
        raise ValueError('Only kafka dataset is configured for this prediction workflow.')

    kafka_output_dir = os.path.join(BASE_PATH, 'Repo data', 'kafka')
    split_files = prepare_kafka_pr_splits(output_dir=kafka_output_dir)
    data_dict = {
        'train': ['/Repo data/kafka_ast_subtrees.json'],
        'val': ['/Repo data/kafka_ast_subtrees.json'],
        'test': ['/Repo data/kafka_ast_subtrees.json'],
        'labels': '/Repo data/kafka/kafka_labels.csv'
    }
    commit_lists = {
        'train': split_files['train'],
        'val': split_files['val'],
        'test': split_files['test']
    }

    dataset = ASTDataset(data_dict, commit_lists, metrics_file=None, special_token=False)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2
    metric_size = dataset.metrics.shape[1] - 1 if 'commit_id' in dataset.metrics.columns else 0
    message_size = 32

    _ = JITGNN(hidden_size, message_size, metric_size)
    model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
    test(model, dataset, clf=None, output_dir=kafka_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['kafka'], default='kafka')
    args = parser.parse_args()
    run_prediction(args.dataset)