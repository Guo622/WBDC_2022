import torch
from transformers import BertConfig
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from dataset import *
from  model import *
from tqdm.autonotebook import tqdm
from utils import *
from torch.optim import swa_utils
from torch.optim.swa_utils import AveragedModel,SWALR
import logging
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    level=logging.INFO,
    filename='./logs/info2_pre.log',
    filemode='a',
    format='%(asctime)s-%(message)s'
)


def train(net: nn.Module, trainloader,testloader, loss_fn, optimizer,scheduler, epochs,writer):

    net.train()

    for e in range(epochs):
        tqdm_loader = tqdm(trainloader)
        total_train_loss = 0
        # num = 0

        for i, input_data in enumerate(tqdm_loader):

            loss,lm_loss,vm_loss,itm_loss = net(*input_data)
            loss.backward()

            optimizer.step()
            scheduler.step()
    
            net.zero_grad()

            total_train_loss += loss.item()

            writer.add_scalar('pretrain2/loss', loss.item(), e*len(trainloader)+i)

            tqdm_loader.set_description(
                f"epoch:{e+1} batch:{i+1} loss:{loss.item():.6f} lm_loss:{lm_loss.item():.6f} vm_loss:{vm_loss.item():.6f} itm_loss:{itm_loss.item():.6f}")

          
        torch.save(net.state_dict(),f'./check/pretrain/pretrain2.pth')

if __name__ == '__main__':
    config = Config()
    config.seed_torch()
    mode='pretrain2'
    config.epochs=2
    config.max_steps=130000
    config.warmup_steps=10000
    config.pretrain_batch_size=16  #48
    
    config.annotation_path='./data/annotations/unlabeled.json'
    config.feature_path='./data/zip_feats/unlabeled.zip'
   
    net=Model2(config,'pretrain').to(config.device)
    trainloader= create_dataloaders(config,mode)

    optimizer,scheduler=build_optimizer(config,net,mode)
    writer = SummaryWriter('./logs')
    train(net, trainloader, None,None, optimizer,scheduler ,config.epochs,writer)
    writer.close()




