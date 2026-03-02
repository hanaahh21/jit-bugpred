import os
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from models import JITGNN
from datasets import ASTDataset, prepare_kafka_pr_splits
from train import pretrain, test, resume_training, plot_training, train
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", choices=['apache', 'kafka'], default='kafka')
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = 1
    n_classes = 2

    if args.dataset == 'kafka':
        kafka_output_dir = os.path.join(BASE_PATH, 'Repo data', 'kafka')
        os.makedirs(kafka_output_dir, exist_ok=True)
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
        metrics_file = None
        output_dir = kafka_output_dir
    else:
        data_dict = {
            'train': ['/apache_train_50_all_1.json', '/apache_train_50_all_2.json',
                      '/apache_train_50_all_3.json', '/apache_train_50_all_4.json'],
            'val': ['/apache_valid_50_all.json'],
            'test': ['/apache_test.json'],
            'labels': '/apache_labels.json'
        }
        commit_lists = {
            'train': '/apache_train_50_all.csv',
            'val': '/apache_valid_50_all.csv',
            'test': '/apache_test.csv'
        }
        metrics_file = 'apache_metrics_kamei.csv'
        output_dir = os.path.join(BASE_PATH, 'data')
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(BASE_PATH, 'trained_models'), exist_ok=True)

    dataset = ASTDataset(data_dict, commit_lists, metrics_file=metrics_file, special_token=False)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2   # plus supernode node feature and node colors
    metric_size = dataset.metrics.shape[1] - 1      # exclude commit_id column
    print('hidden_size is {}'.format(hidden_size))
    message_size = 32

    model = JITGNN(hidden_size, message_size, metric_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    pretrain(model, optimizer, criterion, epochs, dataset, output_dir=output_dir)
    # train_features = torch.load(os.path.join(BASE_PATH, 'trained_models/train_features.pt')).cpu().detach().numpy()
    # train_labels = torch.load(os.path.join(BASE_PATH, 'trained_models/train_labels.pt')).cpu().detach().numpy()
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # train(clf, train_features, train_labels)

    # resume training
    # print('resume training')
    # checkpoint = torch.load(os.path.join(BASE_PATH, 'trained_models/checkpoint.pt'))
    # print('checkpoint loaded.')
    # saved_stats = torch.load(os.path.join(BASE_PATH, 'trained_models/stats.pt'))
    # print('stats loaded.')
    # resume_training(checkpoint, saved_stats, model, optimizer, criterion, epochs, dataset)

    # plotting performance and loss plots
    # saved_stats = torch.load(os.path.join(BASE_PATH, 'trained_models/stats.pt'))
    # print('stats loaded.')
    # plot_training(saved_stats)

    if args.test:
        # need map_location=torch.device('cpu') if on CPU
        model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
        test(model, dataset, clf, output_dir=output_dir)