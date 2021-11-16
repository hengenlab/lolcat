import sys
import os
sys.path.append('./')


import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from celltype.data import *
from celltype.models import GlobalPoolingModel, MLP
from celltype.transforms import Dropout
from celltype.visualization import plot_confusion_matrix



def train(model, train_loader, criterion, optimizer, writer, current_step, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        x, batch, target = data.x, data.batch, data.y
        optimizer.zero_grad()
        output = model(x, batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), current_step)
        current_step += 1

    return current_step


def test(model, loader, writer, tag, epoch, device, class_names=None):
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, batch, target = data.x, data.batch, data.y
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

def task_class(task): #select the right class for the task
    task_class_dict = {'v1_celltypes_17':V1Types17CellSets,
                       'v1_celltypes_13':V1Types13CellSets,
                       'v1_celltypes_11':V1Types11CellSets,
                       'v1_celltypes_4':V1Types4CellSets,
                       'v1_layers_5':V1Layers5CellSets,
                       'v1_celltypes_3':V1Types3CellSets,
                       'v1_transfer_celltypes_3':V1TransferTypes3CellSets,
                       'v1_celltypes_2':V1Types2CellSets,
                       'neuropixels_brain_region_4':NeuropixelsBrainRegion4CellSets,
                       'neuropixels_nm_brain_region_4':NeuropixelsNMBrainRegion4CellSets,
                      'neuropixels_brain_structure_29': NeuropixelsBrainStructure29CellSets,
                       'neuropixels_nm_brain_structure_29': NeuropixelsNMBrainStructure29CellSets,
                      'neuropixels_subclass_3':NeuropixelsSubclass3CellSets,
                       'neuropixels_transfer_subclass_3':NeuropixelsTransferSubclass3CellSets,
                       'neuropixels_nm_subclass_3':NeuropixelsNMSubclass3CellSets,
                      'neuropixels_subclass_4':NeuropixelsSubclass4CellSets,
                       'neuropixels_nm_subclass_4':NeuropixelsNMSubclass4CellSets,
                      'calcium_brain_region_6':CalciumBrainRegion6CellSets,
                      'calcium_subclass_4':CalciumSubclass4CellSets,
                      'calcium_subclass_13':CalciumSubclass13CellSets,
                      'calcium_class_2':CalciumClass2CellSets,
                      'calcium_newtypes_4':CalciumNewTypes4CellSets,
                       'calcium_nm_newtypes_4':CalciumNMNewTypes4CellSets,
                       'calcium_newtypes_5':CalciumNewTypes5CellSets,
                       'calcium_nm_newtypes_5':CalciumNMNewTypes5CellSets,
                       'calcium_newtypes_6':CalciumNewTypes6CellSets,
                       'calcium_nm_newtypes_6':CalciumNMNewTypes6CellSets,
                       'calcium_newtypes_7':CalciumNewTypes7CellSets,
                       'calcium_nm_newtypes_7':CalciumNMNewTypes7CellSets
                      }
    return task_class_dict[task]

def prepare_model(config, root, task, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])
    print('0')
    # get data
    TaskCellSets = task_class(task)
    train_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins'], transform=transform).to(device)
    train_eval_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins']).to(device)
    print('1')
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = train_dataset.get_sampler()
    print('2')
    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size)

    # make model
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    pool = global_mean_pool
    classifier = MLP([config['mlp_layers'][-1], 32, num_classes], dropout=config['net_dropout'])

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)
    print('4')
    # training
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)
    print('5')
    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        print('e{}'.format(str(epoch)))
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step, device)
        scheduler.step()
        
        torch.save(model, '{}_model.pt'.format(task))

        
    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))

def prepare_model(config, root, task, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])
    print('0')
    # get data
    TaskCellSets = task_class(task)
    train_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins'], transform=transform).to(device)
    train_eval_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins']).to(device)
    val_dataset = TaskCellSets(root, 'val', config['split_seed'], num_bins=config['num_bins']).to(device)
    test_dataset = TaskCellSets(root, 'test', config['split_seed'], num_bins=config['num_bins']).to(device)
    print('1')
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = train_dataset.get_sampler()
    print('2')
    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
    print('3')
    # make model
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    pool = global_mean_pool
    classifier = MLP([config['mlp_layers'][-1], 32, num_classes], dropout=config['net_dropout'])

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)
    print('4')
    # training
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)
    print('5')
    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        print('e{}'.format(str(epoch)))
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step, device)
        scheduler.step()

        # eval
        train_f1 = test(model, train_eval_loader, writer, 'train', epoch, device, class_names=class_names)
        val_f1 = test(model, val_loader, writer, 'val', epoch, device, class_names=class_names)
        test_f1 = test(model, test_loader, writer, 'test', epoch, device, class_names=class_names)

        if val_f1 > max_val_f1:
            max_train_f1 = train_f1
            max_val_f1 = val_f1
            max_test_f1 = test_f1
            
    #visualize(model, test_loader, writer, 'train', epoch, class_names=class_names)

    torch.save(model, '{}_model_medconfig.pt'.format(task))
    
    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))
    
def generalize_model(config, root, task, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])
    print('0')
    # get data
    TaskCellSets = task_class(task)
    test_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins']).to(device)
    print('1')
    class_names = test_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = test_dataset.get_sampler()
    print('2')
    # make dataloaders
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
    print('3')
    # make model
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    pool = global_mean_pool
    classifier = MLP([config['mlp_layers'][-1], 32, num_classes], dropout=config['net_dropout'])

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)
    model.load_state_dict(torch.load('v1_celltypes_3_model.pt').state_dict())
    print('4')
    # training
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)
    print('5')
    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        print('e{}'.format(str(epoch)))
        test_f1 = test(model, test_loader, writer, 'test', epoch, device, class_names=class_names)
        
        if test_f1 > max_test_f1:
            max_test_f1 = test_f1

    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))

    
def transfer_model(config, root, task, eval_batch_size=512, logdir=None):
    device = torch.device("cuda")

    # augmentation during training
    transform = Dropout(config['trial_dropout'])
    print('0')
    # get data
    TaskCellSets = task_class(task)
    train_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins'], transform=transform).to(device)
    train_eval_dataset = TaskCellSets(root, 'train', config['split_seed'], num_bins=config['num_bins']).to(device)
    val_dataset = TaskCellSets(root, 'val', config['split_seed'], num_bins=config['num_bins']).to(device)
    test_dataset = TaskCellSets(root, 'test', config['split_seed'], num_bins=config['num_bins']).to(device)
    print('1')
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # get training sampler
    sampler = train_dataset.get_sampler()
    print('2')
    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
    print('3')
    # make model
    trial_encoder = MLP(config['mlp_layers'], dropout=config['net_dropout'], batchnorm=config['batchnorm'])
    pool = global_mean_pool
    classifier = MLP([config['mlp_layers'][-1], 32, num_classes], dropout=config['net_dropout'])

    model = GlobalPoolingModel(trial_encoder, classifier, pool=pool).to(device)
    model.load_state_dict(torch.load('v1_transfer_celltypes_3_model_medconfig.pt').state_dict())
    for param in model.encoder.parameters():
        param.requires_grad = False
    print('4')
    # training
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=0.1)

    # logging
    writer = SummaryWriter(logdir)
    print('5')
    max_train_f1, max_val_f1, max_test_f1 = 0, 0, 0
    current_step = 0
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        print('e{}'.format(str(epoch)))
        # train
        current_step = train(model, train_loader, criterion, optimizer, writer, current_step, device)
        scheduler.step()

        # eval
        train_f1 = test(model, train_eval_loader, writer, 'train', epoch, device, class_names=class_names)
        val_f1 = test(model, val_loader, writer, 'val', epoch, device, class_names=class_names)
        test_f1 = test(model, test_loader, writer, 'test', epoch, device, class_names=class_names)

        if val_f1 > max_val_f1:
            max_train_f1 = train_f1
            max_val_f1 = val_f1
            max_test_f1 = test_f1
            
    #visualize(model, test_loader, writer, 'train', epoch, class_names=class_names)

    #torch.save(model, '{}_transfer_model_medconfig.pt'.format(task))
    
    print('Final metrics: train (%.2f), val (%.2f), test (%.2f)' %(max_train_f1, max_val_f1, max_test_f1))

def main():
    
    task = 'transfer'
    
    if task == 'generalize':
        data_root = os.path.join(os.getcwd(), 'data/')  # path to data
        logdir = './runs/neuropixels_transfer_subclass_3'
        task = 'neuropixels_transfer_subclass_3'
        config = {
            "trial_dropout": 0.3,
            "split_seed": 1,
            "num_bins": 128,
            "batch_size": 1000,
            "pool": 'attention',
            "mlp_layers": [-1, 64, 64, 32],
            "net_dropout": 0.3,
            "batchnorm": True,
            "lr": 1e-2,
            "weight_decay": 1e-6,
            "epochs": 1,
            "milestones": [50, 80],
            }
        generalize_model(config, root=data_root, logdir=logdir,task=task)        
    
    elif task == 'train':
        data_root = os.path.join(os.getcwd(), 'data/')  # path to data
        logdir = './runs/v1_transfer_celltypes_3_medconfig'
        task = 'v1_transfer_celltypes_3'
        config = {
            "trial_dropout": 0.7,
            "split_seed": 1,
            "num_bins": 64,
            "batch_size": 32,
            "pool": 'attention',
            "mlp_layers": [-1, 32, 16, 16],
            "net_dropout": 0.5,
            "batchnorm": True,
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "epochs": 50,
            "milestones": [500],
            }        
        prepare_model(config, root=data_root, logdir=logdir,task=task)

        
    elif task == 'transfer':
        data_root = os.path.join(os.getcwd(), 'data/')  # path to data
        logdir = './runs/neuropixels_transfer_subclass_3_medconfig_1e4_frznenc'
        task = 'neuropixels_transfer_subclass_3'
        config = {
            "trial_dropout": 0.7,
            "split_seed": 1,
            "num_bins": 64,
            "batch_size": 32,
            "pool": 'attention',
            "mlp_layers": [-1, 32, 16, 16],
            "net_dropout": 0.5,
            "batchnorm": True,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "epochs": 1000,
            "milestones": [500],
            }        
        transfer_model(config, root=data_root, logdir=logdir,task=task)    

if __name__ == '__main__':
    main()