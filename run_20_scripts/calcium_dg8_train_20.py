import sys
import os
sys.path.append('./')

import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from lolcat import CalciumDGTorchDataset, CalciumNMTorchDataset
from lolcat import LOLCAT, MLP, GlobalAttention, MultiHeadPooling, init_last_layer_imbalance
from lolcat import Dropout, compute_mean_std, Normalize, Compose
from lolcat import MySampler
from lolcat.visualization import plot_confusion_matrix


def train(model, train_loader, criterion, optimizer, writer, current_step, device, class_names=None, unnormalize=None, *, logdir):
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
    cm = confusion_matrix(targets, predictions)
    fig = plot_confusion_matrix(cm, class_names=class_names)
    writer.add_figure('{}/unnormalized_confusion_matrix'.format(tag), fig, epoch)

    cm = confusion_matrix(targets, predictions, normalize='true')
    fig = plot_confusion_matrix(cm, class_names=class_names)
    writer.add_figure('{}/confusion_matrix'.format(tag), fig, epoch)
    return avg_acc, cm, (torch.LongTensor(targets), torch.LongTensor(predictions), losses.detach().cpu())


def run(config, root, eval_batch_size=4096, logdir=None):
    device = torch.device("cuda")

    # normalize
    train_dataset = CalciumDGTorchDataset(root, 'train', k='8', random_seed=config['split_seed'], num_bins=90)
    mean, std = compute_mean_std(train_dataset)

    # augmentation during training
    transform = Compose(Dropout(config['trial_dropout']), Normalize(mean, std, copy=False))
    normalize = Normalize(mean, std)

    # get data
    train_dataset = CalciumDGTorchDataset(root, 'train', k='8', random_seed=config['split_seed'], num_bins=90, transform=transform)# .to(device)
    train_eval_dataset = CalciumDGTorchDataset(root, 'train', k='8', random_seed=config['split_seed'], num_bins=90, transform=normalize) #.to(device)
    val_dataset = CalciumDGTorchDataset(root, 'val', k='8', random_seed=config['split_seed'], num_bins=90, transform=normalize) #.to(device)
    test_dataset = CalciumDGTorchDataset(root, 'test', k='8', random_seed=config['split_seed'], num_bins=90, transform=normalize) # .to(device)
    nm_test_dataset = CalciumNMTorchDataset(root, 'test', k='8', random_seed=config['split_seed'], num_bins=90, transform=normalize) #.to(device)

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = MySampler(train_dataset.target, num_samples=len(train_dataset))

    # make dataloaders
    kwargs = dict(num_workers=8, prefetch_factor=4, pin_memory=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True, **kwargs)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, **kwargs)
    nm_test_loader = DataLoader(nm_test_dataset, batch_size=eval_batch_size, **kwargs)

    # make model
    feature_size = config['mlp_layers'][-1]
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    classifier = MLP([2 * feature_size, feature_size, num_classes], dropout=config['net_dropout'])

    if config['pool'] == 'mean':
        pool = global_mean_pool
    elif config['pool'] == 'attention':
        pool = MultiHeadPooling(GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1))
    else:
        raise ValueError

    model = LOLCAT(trial_encoder, classifier, pool=pool).to(device)

    # initialize last layer
    _, counts = torch.unique(train_dataset.target, return_counts=True)
    class_weights = counts / counts.sum()
    oversampled_class_weights = sampler.factors * class_weights
    init_last_layer_imbalance(model.classifier, oversampled_class_weights)

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

    max_train, max_val, max_test = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step, device,
                             class_names=class_names, unnormalize=None, logdir=logdir)
        scheduler.step()

        # eval
        train_balanced_acc, train_cm, (ttargets, tpredictions, tlosses) = test(model, train_eval_loader, writer, 'train', epoch, device, class_names=class_names, )
        val_balanced_acc, val_cm, (vtargets, vpredictions, vlosses) = test(model, val_loader, writer, 'val', epoch, device, class_names=class_names)
        test_balanced_acc, test_cm, _ = test(model, test_loader, writer, 'test', epoch, device, class_names=class_names)
        test(model, nm_test_loader, writer, 'test_nm', epoch, device, class_names=class_names)

        # update sampler
        train_loader.sampler.step(tlosses, ttargets, vlosses, vtargets)
        writer.add_scalars('oversampling_factors',
                           {class_name: factor.item() for class_name, factor in zip(train_dataset.class_names, train_loader.sampler.factors)},
                           global_step=epoch)

        if val_balanced_acc > max_val:
            max_train = train_balanced_acc
            max_val = val_balanced_acc
            max_test = test_balanced_acc

            torch.save(model, os.path.join(logdir, 'model_early_stopping.pt'))
            np.savez(os.path.join(logdir, 'confusion_matrices.npz'), train=train_cm, val=val_cm, test=test_cm)

        torch.save(model, os.path.join(logdir, 'model_final.pt'))
    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train, max_val, max_test))


def train_on_split(split_id):
    data_root = os.path.join(os.getcwd(), 'data/')  # path to data
    logdir = './runs/calcium_dg_k=4/run_v4_{}'.format(split_id)

    config = {
        "split_seed": split_id,
        "trial_dropout": 0.4,
        "num_bins": 90,
        "batch_size": 128,
        "pool": 'attention',
        "mlp_layers": [-1, 32, 16, 16, 16],
        "net_dropout": 0.5,
        "batchnorm": True,
        "lr": 1e-2,
        "weight_decay": 1e-4,
        "epochs": 100,
        "milestones": [5, 80],
        }

    run(config, root=data_root, logdir=logdir)


if __name__ == '__main__':
    for split_id in range(1, 21):
        print('Training for split #{}'.format(split_id))
        train_on_split(str(split_id))
