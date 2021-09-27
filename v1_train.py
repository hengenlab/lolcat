import sys
import os
sys.path.append('./')


import numpy as np
import torch
import torch.optim as optim
from functools import partial

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, Set2Set, global_add_pool
from tqdm import tqdm

from celltype.data import V1CellSets
from celltype.models import GlobalPoolingModel, MLP, GlobalAttention, MultiHeadPooling
from celltype.block_lin import BlockMLP
from celltype.transforms import Dropout
from celltype.visualization import plot_confusion_matrix, generate_fingerprint


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
    cm = confusion_matrix(targets, predictions, normalize='true')
    fig = plot_confusion_matrix(cm, class_names=class_names)
    writer.add_figure('{}/confusion_matrix'.format(tag), fig, epoch)
    return f1

def visualize(model, loader, writer, tag, epoch, class_names):
    model.eval()

    embeddings = []
    global_embeddings = []
    images = []
    metadata = []
    global_metadata = []
    cum_sum = 0
    global_cum_sum = 0

    with torch.inference_mode():
        for data in loader:
            x, batch, target = data.x, data.batch, data.y
            batch = batch.to(x.device)

            output, emb, global_emb = model(x, batch, return_trial_embeddings=True)
            embeddings.append(emb.detach().cpu().numpy())
            global_embeddings.append(global_emb.detach().cpu().numpy())
            # images.append(generate_fingerprint(x.cpu().numpy()))

            cell_id = cum_sum + batch.cpu().numpy()
            cell_type_global = np.array(class_names)[target.cpu().numpy()]
            cell_type = np.repeat(cell_type_global, 100)
            orientation = data.orientation.cpu().numpy()
            trial_md = data.trial.cpu().numpy()
            fr = torch.log10(1 + x.sum(axis=1)).cpu().numpy()
            sparsity = global_add_pool((x.sum(axis=1) == 0).float(), batch) / 100
            sparsity = sparsity.cpu().numpy()
            # combine metadata
            metadata.append(np.column_stack([orientation, trial_md, cell_id, cell_type, fr]))
            global_metadata.append(np.column_stack([cell_type_global, sparsity]))

            cum_sum += batch.size(0)
            global_cum_sum = global_cum_sum + data.num_graphs

    metadata_header = ['orientation', 'trial_md', 'cell_id', 'cell_type', 'firing_rate']
    embeddings = np.row_stack(embeddings)
    global_embeddings = np.row_stack(global_embeddings)
    # images = torch.from_numpy(np.row_stack(images))
    metadata = np.row_stack(metadata).tolist()
    global_metadata = np.column_stack([np.row_stack(global_metadata), np.arange(global_cum_sum)]).tolist()

    writer.add_embedding(embeddings, metadata=metadata, tag='trial_embedding',
                         metadata_header=metadata_header, global_step=epoch)
    writer.add_embedding(global_embeddings, metadata=global_metadata, tag='global_embedding',
                         metadata_header=['cell_type', 'sparsity', 'cell_id'], global_step=epoch)


def run(config, root, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])

    # get data
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
    # trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    trial_encoder = BlockMLP()
    classifier = MLP([feature_size*2, 32, num_classes], dropout=config['net_dropout'])

    if config['pool'] == 'mean':
        pool = global_mean_pool
    elif config['pool'] == 'max':
        pool = global_max_pool
    elif config['pool'] == 'sort':
        pool = partial(global_sort_pool, k=8)
    elif config['pool'] == 'attention':
        pool = MultiHeadPooling(GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1),
                                GlobalAttention(feature_size, feature_size//2, heads=1))
    elif config['pool'] == 'set2set':
        pool = Set2Set(feature_size, processing_steps=2, num_layers=1)
    else:
        raise ValueError

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)

    # training
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)

    # logging
    writer = SummaryWriter(logdir)

    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step)
        scheduler.step()

        # eval
        train_f1 = test(model, train_eval_loader, writer, 'train', epoch, class_names=class_names)
        val_f1 = test(model, val_loader, writer, 'val', epoch, class_names=class_names)
        test_f1 = test(model, test_loader, writer, 'test', epoch, class_names=class_names)

        if val_f1 > max_val_f1:
            max_train_f1 = train_f1
            max_val_f1 = val_f1
            max_test_f1 = test_f1

    visualize(model, test_loader, writer, 'train', epoch, class_names=class_names)

    torch.save(model, 'model.pt')

    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))


def main():
    data_root = os.path.join(os.getcwd(), 'data/')  # path to data
    logdir = './runs/v1_17_blockmlp_multihead_att_6'

    config = {
        "trial_dropout": 0.3,
        "split_seed": 1,
        "num_bins": 128,
        "batch_size": 256,
        "pool": 'attention',
        "mlp_layers": [-1, 64, 64, 32],
        "net_dropout": 0.3,
        "batchnorm": True,
        "lr": 1e-2,
        "weight_decay": 1e-6,
        "epochs": 120,
        "milestones": [50, 80],
        }

    run(config, root=data_root, logdir=logdir)


if __name__ == '__main__':
    main()
