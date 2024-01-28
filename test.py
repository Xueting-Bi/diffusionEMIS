import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD
from models.scadata import ScaDataset
# from torch.autograd import Variable

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/home/xueting/code/scadiffusion4DCL/logs/IS_2024_01_28__13_06_42/ckpt_0.072683_100.pt')
parser.add_argument('--categories', type=str_list, default=['mnist'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='data/MnistScatData.h5')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint

ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ScaDataset(args.dataset_path,
    split='test',
)


test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
# print('Trainable parameters: ', trainable_params)


all_ref = []
all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    esct, points = batch
    model.eval()
    with torch.no_grad():
            
        B, _, _ = esct.shape
        esct = esct.view(B, 1, -1).to(args.device)

        recons = model.decode(esct, points.size(1), flexibility=ckpt['args'].flexibility).detach()


    all_ref.append(points.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
cd, _ = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
logger.info('CD:  %.12f' % cd)
