import argparse
import os

import torch

from datasets import ASTDataset, prepare_kafka_pr_splits
from models import JITGNN
from train import test

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


def run_prediction(setup=1):
    kafka_output_dir = os.path.join(BASE_PATH, 'ast_embeddings', f'setup{setup}')
    split_files = prepare_kafka_pr_splits(output_dir=kafka_output_dir, setup=setup)
    data_dict = split_files['data_dict']
    commit_lists = split_files['commit_lists']

    dataset = ASTDataset(data_dict, commit_lists, metrics_file=None, special_token=False)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2
    metric_size = dataset.metrics.shape[1] - 1 if 'commit_id' in dataset.metrics.columns else 0
    message_size = 32

    _ = JITGNN(hidden_size, message_size, metric_size)
    model = torch.load(
        os.path.join(BASE_PATH, 'trained_models/model_best_f1.pt'),
        map_location='cpu',
        weights_only=False,
    )
    test(model, dataset, clf=None, output_dir=kafka_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    run_prediction(setup=args.setup)
