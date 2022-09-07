import logging
import os
import time
import torch

from model import MultiModal
from config import parse_args
from data_helper import create_dataloaders,create_dataloaders_pretrain,MultiModalDataset
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate,build_optimizer_pretrain
from util import EMA,FGM



def pretrain(args):
    # 1. load data
    dataset = MultiModalDataset(args, args.pretrain_annotation, args.pretrain_zip_frames)
    train_dataloader= create_dataloaders_pretrain(args,dataset)

    # 2. build model and optimizers
    model = MultiModal(args,'pretrain')
    model.load_state_dict(torch.load('save/pretrain_visual.bin', map_location='cpu')['model_state_dict'], strict=False)
    model.load_state_dict(torch.load('save/pretrain_text.bin', map_location='cpu')['model_state_dict'], strict=False)
    
    optimizer, scheduler =build_optimizer_pretrain(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss,lm_loss,vm_loss,itm_loss = model(inputs=batch)
            loss=loss.mean()
            lm_loss=lm_loss.mean()
            itm_loss=itm_loss.mean()
            vm_loss=vm_loss.mean()
            loss.backward()
            
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, lm_loss {lm_loss:.3f}, vm_loss {vm_loss:.3f}, itm_loss {itm_loss:.3f}")


        # 5. save checkpoint

        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                       f'{args.savedmodel_path}/pretrain.bin')



def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    
    args.max_epochs = 2
    args.warmup_steps=5000
    args.max_steps=100000   #31240 each epoch 

    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("PreTraining/evaluation parameters: %s", args)

    pretrain(args)

if __name__ == '__main__':
    main()
