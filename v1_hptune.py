import sys
import os
sys.path.append('./')

from functools import partial

import numpy as np
import torch
import torch.optim as optim

from ray import tune
from ray.tune import CLIReporter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, Set2Set

from celltype.data import V1CellSets
from celltype.models import GlobalPoolingModel, MLP, GlobalAttention
from celltype.transforms import Dropout
from celltype.visualization import plot_confusion_matrix



def train(model, train_loader, criterion, optimizer, writer, current_step):
    model.train()

    for data in train_loader:
        x, batch, target = data.x, data.batch, data.y
        batch = batch.to(x.device)
        optimizer.zero_grad()
        output = model(x, batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), current_step)
        current_step += 1

    return current_step


def test(model, loader, writer, tag, epoch, class_names=None):
    model.eval()

    predictions = []
    targets = []
    with torch.inference_mode():
        for data in loader:
            x, batch, target = data.x, data.batch, data.y
            batch = batch.to(x.device)
            output = model(x, batch)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = np.ndarray.flatten(pred.cpu().numpy())
            targ = target.cpu().numpy()
            predictions.append(pred)
            targets.append(targ)

    targets = np.hstack(targets)
    predictions = np.hstack(predictions)

    # report accuracy
    acc = accuracy_score(targets, predictions)
    writer.add_scalar('{}/acc'.format(tag), acc, epoch)

    # report f1 score
    f1 = f1_score(targets, predictions, average='macro')
    writer.add_scalar('{}/f1_score'.format(tag), f1, epoch)

    # report confusion matrix
    if epoch % 50 == 0:
        cm = confusion_matrix(targets, predictions, normalize='true')
        fig = plot_confusion_matrix(cm, class_names=class_names)
        writer.add_figure('{}/confusion_matrix'.format(tag), fig, epoch)
    return f1


def run(config, root, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])

    # get data
    # todo adapt to task/ dataset
    train_dataset = V1CellSets(root, 'train', config['split_seed'], num_bins=config['num_bins'], transform=transform).to(device)
    train_eval_dataset = V1CellSets(root, 'train', config['split_seed'], num_bins=config['num_bins']).to(device)
    val_dataset = V1CellSets(root, 'val', config['split_seed'], num_bins=config['num_bins']).to(device)
    test_dataset = V1CellSets(root, 'test', config['split_seed'], num_bins=config['num_bins']).to(device)

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = train_dataset.get_sampler()

    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)

    # make model
    feature_size = config['mlp_layers'][-1]
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    classifier = MLP([feature_size, 32, num_classes], dropout=config['net_dropout'])

    if config['pool'] == 'mean':
        pool = global_mean_pool
    elif config['pool'] == 'max':
        pool = global_max_pool
    elif config['pool'] == 'sort':
        pool = partial(global_sort_pool, k=8)
    elif config['pool'] == 'attention':
        pool = GlobalAttention(feature_size, feature_size, heads=1)
    elif config['pool'] == 'set2set':
        pool = Set2Set(feature_size, processing_steps=2, num_layers=1)
    else:
        raise ValueError

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)

    # training
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config['milestones']], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)

    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in range(1, config['epochs'] + 1):
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step)
        scheduler.step()

        # eval
        train_f1 = test(model, train_eval_loader, writer, 'train', epoch, class_names=class_names)
        val_f1 = test(model, val_loader, writer, 'val', epoch, class_names=class_names)
        test_f1 = test(model, test_loader, writer, 'test', epoch, class_names=class_names)

        tune.report(train_f1=train_f1, val_f1=val_f1, test_f1=test_f1)

        if val_f1 > max_val_f1:
            max_train_f1 = train_f1
            max_val_f1 = val_f1
            max_test_f1 = test_f1

    writer.add_scalar('early_stopping/train_f1', max_train_f1)
    writer.add_scalar('early_stopping/val_f1', max_val_f1)
    writer.add_scalar('early_stopping/test_f1', max_test_f1)


def main():
    data_root = os.path.join(os.getcwd(), 'data/')  # path to data

    gpus_per_trial = 0.5
    cpus_per_trial = 1.0
    num_samples = 1000

    config = {
        "trial_dropout": tune.quniform(0.0, 0.9, 0.1),
        "split_seed": tune.choice([1]),
        "num_bins": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "mlp_layers": tune.choice([[-1, 128, 64, 64, 32], [-1, 64, 64, 32], [-1, 64, 64, 64, 64, 64, 32]]),
        "net_dropout": tune.quniform(0.0, 0.4, 0.1),
        "pool": tune.choice(['attention']),
        "batchnorm": tune.choice([True, False]),
        "lr": tune.loguniform(1e-5, 1),
        "weight_decay": tune.loguniform(1e-7, 1e-3),
        "epochs": tune.choice([100]),
        "milestones": tune.choice([80, 90, 100])
        }

    reporter = CLIReporter(
        parameter_columns=["num_bins", "mlp_layers", "lr"],
        metric_columns=["train_f1", "test_f1", "training_iteration"])

    result = tune.run(
        partial(run, root=data_root),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        local_dir='./runs',
        name="v1_17_celltypes"      # todo adapt to task/ dataset
    )

    best_trial = result.get_best_trial("test_f1", "max", "last")
    print("Best trial config: {}".format(best_trial.config))


if __name__ == '__main__':
    main()
