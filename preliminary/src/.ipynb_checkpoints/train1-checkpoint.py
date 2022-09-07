import torch
import torch.nn as nn
from config import Config
from dataset import *
from  model import *
from tqdm.autonotebook import tqdm
from utils import *
import logging
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(
    level=logging.INFO,
    filename='./logs/info1.log',
    filemode='a',
    format='%(asctime)s-%(message)s'
)


def train(net: nn.Module, trainloader,testloader, loss_fn, optimizer,scheduler, epochs,writer):
    logging.info("new try\n")
    ema=EMA(net,0.999)
    ema.register()

    swa_start=2
    swa_batch=2000

    net.train()
#     pgd = PGD(net)
#     K = 3
    
    
    for e in range(epochs):
        tqdm_loader = tqdm(trainloader)
        total_train_loss = 0
        num = 0
        total_num=0
        for i, input_data in enumerate(tqdm_loader):
            label = input_data[-1]
            y = net(*input_data[:-1])
            loss = loss_fn(y, label)
            loss.backward()  
            
#             pgd.backup_grad()
#             for t in range(K):
#                 pgd.attack(is_first_attack=(t == 0))  
#                 if t != K - 1:
#                     net.zero_grad()
#                 else:
#                     pgd.restore_grad()
#                 outputs_adv =net(*input_data[:-1])
#                 loss_adv = loss_fn(outputs_adv,label)
#                 loss_adv.backward()  
#             pgd.restore() 
            
            optimizer.step()
            scheduler.step()
            ema.update()
            net.zero_grad()

            total_train_loss += loss.item()
            pred_label_id = torch.argmax(y, dim=1)
            acc_num = torch.sum((label == pred_label_id))
            num += acc_num
            total_num+=label.shape[0]
            acc = acc_num/label.shape[0]
            writer.add_scalar('train/loss_one', loss.item(), e*len(trainloader)+i)
            writer.add_scalar('train/acc_one',acc, e*len(trainloader)+i)
            tqdm_loader.set_description(
                f"epoch:{e+1} batch:{i+1} loss:{loss.item():.6f} acc:{acc:.6f}")

            if e >= swa_start and (i+1) % swa_batch==0:
                ema.apply_shadow()
                torch.save(net.state_dict(), f"check/train1/model_train_{e+1}_{i+1}.pth")
                ema.restore()
      
        ema.apply_shadow()
        message=f"average train loss: {total_train_loss/i: .6f} average train accuracy: {num/total_num: .6f}\n"
        print(message)
        logging.info(f" epoch {e+1} "+message)
        if e>=swa_start:
            torch.save(net.state_dict(), f"check/train1/model_train_{e+1}_{i+1}.pth")
        ema.restore()

if __name__ == '__main__':
    config = Config()
    config.seed_torch()
    config.train_batch_size=16
    config.epochs=4
    net = Model1(config).to(config.device)
    net.load_state_dict(torch.load('./check/pretrain/pretrain1.pth'))
    trainloader= create_dataloaders(config,'train1')
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    optimizer,scheduler=build_optimizer(config,net,'train1')
    writer = SummaryWriter('./logs')
    train(net, trainloader, None,loss_fn, optimizer,scheduler ,config.epochs,writer)
    writer.close()




