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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--setup", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = 1
    n_classes = 2

    output_dir = os.path.join(BASE_PATH, 'ast_embeddings', f'setup{args.setup}')
    os.makedirs(output_dir, exist_ok=True)
    split_files = prepare_kafka_pr_splits(output_dir=output_dir, setup=args.setup)
    data_dict = split_files['data_dict']
    commit_lists = split_files['commit_lists']
    metrics_file = None

    os.makedirs(os.path.join(BASE_PATH, 'trained_models'), exist_ok=True)

    dataset = ASTDataset(data_dict, commit_lists, metrics_file=metrics_file, special_token=False)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2   # plus supernode node feature and node colors
    metric_size = dataset.metrics.shape[1] - 1      # exclude commit_id column
    print('hidden_size is {}'.format(hidden_size))
    message_size = 32

    model = JITGNN(hidden_size, message_size, metric_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training (fresh or resumed)
    if args.resume:
        checkpoint_path = os.path.join(BASE_PATH, 'trained_models', 'checkpoint.pt')
        stats_path = os.path.join(BASE_PATH, 'trained_models', 'stats.pt')
        if os.path.exists(checkpoint_path) and os.path.exists(stats_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            saved_stats = torch.load(stats_path, map_location='cpu')
            done_epochs = int(checkpoint.get('epoch', 0))
            remaining_epochs = max(0, epochs - done_epochs)
            if remaining_epochs == 0:
                print(f"Checkpoint already at epoch {done_epochs}, target epochs={epochs}. Nothing to train.")
            else:
                print(f"Continuing from epoch {done_epochs} to target epoch {epochs} ({remaining_epochs} epochs remaining).")
                resume_training(checkpoint, saved_stats, model, optimizer, criterion, remaining_epochs, dataset)
        else:
            print('Resume requested but checkpoint/stats not found. Starting fresh training.')
            pretrain(model, optimizer, criterion, epochs, dataset, output_dir=output_dir)
    else:
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
        model = torch.load(
            os.path.join(BASE_PATH, 'trained_models/model_best_f1.pt'),
            map_location='cpu',
            weights_only=False,
        )
        test(model, dataset, clf, output_dir=output_dir)
