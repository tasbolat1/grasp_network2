from os import stat
import torch
from models.networks import GraspEvaluator
from datasets import GraspDataset
from torch.utils.data import DataLoader
#from tqdm.auto import tqdm
from models.utils import save_model, add_weight_decay
import time
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter

import stats

# python train_evaluator4.py --data_dir_grasps data/grasps4/preprocessed --data_dir_pcs data/pcs4 --batch_size 256 --lr 0.0001 --device_ids 1 2


# sampler run
parser = argparse.ArgumentParser("Train Grasp Evaluator model.")
parser.add_argument("--tag", type=str, help="Frequency to test model.", default='evaluator')
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=1000)
parser.add_argument("--data_dir_grasps", type=str, help="Path to data.", required=True)
parser.add_argument("--data_dir_pcs", type=str, help="Path to data.", required=True)
parser.add_argument("--lr", type=float, help="Learning rate.", default=0.0001)
parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer.", default=0.0)
parser.add_argument("--batch_size", type=int, help="Batch Size.", default=128)
parser.add_argument("--num_workers", type=int, help="Number of workers for dataloader.", default=8)
parser.add_argument("--save_freq", type=int, help="Frequency to save model.", default=1)
parser.add_argument("--test_freq", type=int, help="Frequency to test model.", default=1)
parser.add_argument("--continue_train", action='store_true', help="Continue to train: checkpoint_dir must be indicated")
parser.add_argument("--checkpoint_info", type=str, help="Checkpoint info as: name_epoch", default=None)
parser.add_argument("--full_pc", action='store_true', help="Use full point cloud for training?")
parser.add_argument("--device_ids", nargs="+", type=int, help="Index of cuda devices. Pass -1 to set to cpu.", default=[0])
parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
args = parser.parse_args()

POS_TRAIN_COUNT = 1
NEG_TRAIN_COUNT = 1

POS_TEST_COUNT = 1
NEG_TEST_COUNT = 1

# prepare device ids
if len(args.device_ids) > 1:
    device = torch.device(f'cuda:{args.device_ids[0]}')
else:
    if args.device_ids[0] == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device_ids[0]}')

# prepare model
model = GraspEvaluator(args, output_channels=1)
weight_decay = args.weight_decay
if weight_decay:
    parameters = add_weight_decay(model, weight_decay)
    weight_decay = 0.
else:
    parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay, lr=args.lr)

if args.continue_train:
    assert args.checkpoint_info != None, 'please indicate checkpoint info.'
    model_name = args.checkpoint_info.split('_')[0]
    init_epoch = int(args.checkpoint_info.split('_')[1])
    model_save_path = f'saved_models/{args.tag}/{model_name}/'
    print(f'Continue to train {model_save_path} starting from {init_epoch} epoch ...')
    model.load_state_dict(torch.load(f'{model_save_path}/{init_epoch}.pt'))
else:
    model_name= int( time.time()*100 )
    model_save_path = f'saved_models/{args.tag}/{model_name}/'
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    init_epoch = 1

model.to(device)
if len(args.device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total learnable params: {total_params}.')

# prepare to save to tensorboard
writer = SummaryWriter(model_save_path)

# prepare data
train_dataset = GraspDataset(path_to_grasps = args.data_dir_grasps, path_to_pc=args.data_dir_pcs, split='train', augment=True, full_pc=args.full_pc)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataset = GraspDataset(path_to_grasps = args.data_dir_grasps, path_to_pc=args.data_dir_pcs, split='test', augment=False, full_pc=args.full_pc)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



pos_weight_train = torch.Tensor([NEG_TRAIN_COUNT/POS_TRAIN_COUNT]).to(device)
pos_weight_test = torch.Tensor([NEG_TEST_COUNT/POS_TEST_COUNT]).to(device)
criteria_train = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_train, reduction='mean')
criteria_test = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_test, reduction='mean')

len_trainloader = len(train_dataloader)
len_testloader = len(test_dataloader)

print(f'len_trainloader: {len_trainloader} for {len(train_dataset)}')
print(f'len_testloader: {len_testloader} for {len(test_dataset)}')

def train(epoch):
    train_loss = stats.AverageMeter('train/loss', writer)
    accuracy = stats.BinaryAccuracyWithCat('train', writer)
    # train
    model.train()
    for k, (quat, trans, pcs, labels, cats) in enumerate(train_dataloader):
        
        # move to device
        quat, trans, pcs, labels = quat.to(device), trans.to(device), pcs.to(device), labels.to(device)
        
        # normalize pcs
        pc_mean = pcs.mean(dim=1).unsqueeze(1)
        pcs = pcs - pc_mean
        trans = trans-pc_mean.squeeze(1)
        
        # forward pass
        out = model(quat, trans, pcs)
        out = out.squeeze(-1)
        #print(out.shape)
        
        # compute loss
        labels = labels.squeeze(-1)
        loss = criteria_train(out, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), quat.shape[0])
        if k % 400 == 0:
            print( train_loss.summary(epoch, some_txt=f'{k}/{len_trainloader}') )
        accuracy.update(out.squeeze().detach().cpu().numpy(), labels.detach().cpu().numpy(), cats)

    # print( train_loss.summary(epoch) )
    print( accuracy.summary(epoch) )

def test(epoch):
    
    model.eval()

    test_loss  = stats.AverageMeter('test/loss', writer)
    accuracy = stats.BinaryAccuracyWithCat('test', writer)

    with torch.no_grad():
        for k, (quat, trans, pcs, labels, cats) in enumerate(test_dataloader):

            # move to device
            quat, trans, pcs, labels = quat.to(device), trans.to(device), pcs.to(device), labels.to(device)

            # normalize pcs
            pc_mean = pcs.mean(dim=1).unsqueeze(1)
            pcs = pcs - pc_mean
            trans = trans-pc_mean.squeeze(1)

            # forward pass
            out = model(quat, trans, pcs)
            out = out.squeeze(-1)
            #print(out.shape)

            # compute loss
            labels = labels.squeeze(-1)
            loss = criteria_test(out, labels)

            # print( test_loss.summary(epoch, some_txt=f'{k}/{len_testloader}') )

            test_loss.update( loss.item(), quat.shape[0] )
            accuracy.update( out.squeeze().detach().cpu().numpy(), labels.detach().cpu().numpy(), cats )

    print( test_loss.summary(epoch) )
    print( accuracy.summary(epoch) )


# # test inital values
test(init_epoch)

for epoch in range(init_epoch+1,args.epochs+1):
    train(epoch)
    if epoch % args.test_freq == 0:
        test(epoch)
    
    # save model
    if epoch % args.save_freq == 0:
        save_model(model, path=model_save_path, epoch=epoch)
        
    print(f'Done with epoch {epoch} ...')