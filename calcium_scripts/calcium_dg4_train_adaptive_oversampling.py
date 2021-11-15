import sys
import os
sys.path.append('./')

import numpy as np
import torch
import torch.optim as optim
from functools import partial

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, Set2Set, GraphMultisetTransformer, GCNConv
from tqdm import tqdm


from lolcat import CalciumDGTorchDataset, CalciumNMTorchDataset
from lolcat import LOLCAT, MLP, GlobalAttention, MultiHeadPooling
from lolcat import Dropout, compute_mean_std, Normalize, Compose
from lolcat import MySampler
from lolcat.visualization import plot_confusion_matrix, visualize_sample


def train(model, train_loader, criterion, optimizer, writer, current_step, device, class_names=None, unnormalize=None):
    model.train()

    for data in train_loader:
        x, batch, target = data.x, data.batch, data.y
        x = x.to(device).float()
        target = target.to(device)
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, batch)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), current_step)
        current_step += 1

        if current_step % 100 == 0:
            # get 4 random samples:
            for i in range(6):
                x_, target_ = unnormalize(x[batch==i]).cpu(), target[i].cpu()
                logits_ = logits[i].detach().cpu()
                fig = visualize_sample(x_, target_, logits_, class_names=class_names)
                writer.add_figure('sample/{}'.format(i), fig, current_step)

    return current_step


def test(model, loader, writer, tag, epoch, device, class_names=None):
    model.eval()

    predictions = []
    targets = []
    losses = []
    with torch.inference_mode():
        for data in loader:
            x, batch, target = data.x, data.batch, data.y
            x = x.to(device).float()
            target = target.to(device)
            batch = batch.to(device)
            logits, _ = model(x, batch)
            loss = F.cross_entropy(logits, target, reduction='none')
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = np.ndarray.flatten(pred.cpu().numpy())
            targ = target.cpu().numpy()
            predictions.append(pred)
            targets.append(targ)
            losses.append(loss)

    targets = np.hstack(targets)
    predictions = np.hstack(predictions)
    losses = torch.cat(losses)

    # report accuracy
    acc = accuracy_score(targets, predictions)
    writer.add_scalar('{}/acc'.format(tag), acc, epoch)

    # report f1 score
    f1 = f1_score(targets, predictions, average='macro')
    writer.add_scalar('{}/f1_score'.format(tag), f1, epoch)

    # report avg acc
    avg_acc = balanced_accuracy_score(targets, predictions)
    writer.add_scalar('{}/balanced_acc'.format(tag), avg_acc, epoch)

    # report confusion matrix
    cm = confusion_matrix(targets, predictions, normalize='true')
    fig = plot_confusion_matrix(cm, class_names=class_names)
    writer.add_figure('{}/confusion_matrix'.format(tag), fig, epoch)

    cm = confusion_matrix(targets, predictions)
    fig = plot_confusion_matrix(cm, class_names=class_names)
    writer.add_figure('{}/unnormalized_confusion_matrix'.format(tag), fig, epoch)
    return f1, (torch.LongTensor(targets), torch.LongTensor(predictions), losses.detach().cpu())


def run(config, root, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # normalize
    train_dataset = CalciumDGTorchDataset(root, 'train', k='4', random_seed=config['split_seed'], num_bins=90)
    mean, std = compute_mean_std(train_dataset)

    # augmentation during training
    transform = Compose(Dropout(config['trial_dropout']), Normalize(mean, std, copy=False))
    normalize = Normalize(mean, std)

    # get data
    train_dataset = CalciumDGTorchDataset(root, 'train', k='4', random_seed=config['split_seed'], num_bins=90, transform=transform).to(device)
    train_eval_dataset = CalciumDGTorchDataset(root, 'train', k='4', random_seed=config['split_seed'], num_bins=90, transform=normalize).to(device)
    val_dataset = CalciumDGTorchDataset(root, 'val', k='4', random_seed=config['split_seed'], num_bins=90, transform=normalize).to(device)
    test_dataset = CalciumDGTorchDataset(root, 'test', k='4', random_seed=config['split_seed'], num_bins=90, transform=normalize).to(device)
    nm_test_dataset = CalciumNMTorchDataset(root, 'test', k='4', random_seed=config['split_seed'], num_bins=90, transform=normalize).to(device)

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = MySampler(train_dataset.target, num_samples=len(train_dataset))

    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
    nm_test_loader = DataLoader(nm_test_dataset, batch_size=eval_batch_size)

    # make model
    feature_size = config['mlp_layers'][-1]
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    classifier = MLP([2 * feature_size, feature_size, num_classes], dropout=config['net_dropout'])

    if config['pool'] == 'mean':
        pool = global_mean_pool
    elif config['pool'] == 'max':
        pool = global_max_pool
    elif config['pool'] == 'sort':
        pool = partial(global_sort_pool, k=8)
    elif config['pool'] == 'attention':
        pool = MultiHeadPooling(GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size // 2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1))
    elif config['pool'] == 'set2set':
        pool = Set2Set(feature_size, processing_steps=2, num_layers=1)
    elif config['pool'] == 'graph_attention':
        pool = GraphMultisetTransformer(
            in_channels=feature_size,
            hidden_channels=2*feature_size,
            out_channels=2*feature_size,
            num_nodes=600,
            pooling_ratio=0.25,
            pool_sequences=["GMPool_I", "SelfAtt", "GMPool_I"],
            num_heads=4,
            layer_norm=False,
        )
    else:
        raise ValueError

    model = LOLCAT(trial_encoder, classifier, pool=pool).to(device)

    # initialize last layer

    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)
    writer.add_scalars('oversampling_factors',
                       {class_name: factor.item() for class_name, factor in
                        zip(train_dataset.class_names, train_loader.sampler.factors)},
                       global_step=0)

    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step, device,
                             class_names=class_names, unnormalize=normalize.unnormalize_x)
        scheduler.step()

        # eval
        train_f1, (ttargets, tpredictions, tlosses) = test(model, train_eval_loader, writer, 'train', epoch, device, class_names=class_names)
        val_f1, (vtargets, vpredictions, vlosses) = test(model, val_loader, writer, 'val', epoch, device, class_names=class_names)
        test_f1, _ = test(model, test_loader, writer, 'test', epoch, device, class_names=class_names)
        test(model, nm_test_loader, writer, 'test_nm', epoch, device, class_names=class_names)

        # update sampler
        train_loader.sampler.step(tlosses, ttargets, vlosses, vtargets)
        writer.add_scalars('oversampling_factors',
                           {class_name: factor.item() for class_name, factor in zip(train_dataset.class_names, train_loader.sampler.factors)},
                           global_step=epoch)

        if val_f1 > max_val_f1:
            max_train_f1 = train_f1
            max_val_f1 = val_f1
            max_test_f1 = test_f1

            torch.save(model, os.path.join(logdir, 'model.pt'))
        torch.save(model, os.path.join(logdir, 'model_final.pt'))
    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))




def main():
    data_root = os.path.join(os.getcwd(), 'data/')  # path to data
    logdir = './runs/calcium_dg_adaptive_sampler_updated_5'

    config = {
        "trial_dropout": 0.8,
        "split_seed": 1,
        "num_bins": 90,
        "batch_size": 128,
        "pool": 'attention',
        "mlp_layers": [-1, 32, 16, 16, 16],
        "net_dropout": 0.5,
        "batchnorm": True,
        "lr": 1e-2,
        "weight_decay": 1e-4,
        "epochs": 1000,
        "milestones": [10, 100, 400],
        }

    run(config, root=data_root, logdir=logdir)


if __name__ == '__main__':
    main()
