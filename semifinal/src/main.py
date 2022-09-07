import logging
import os
import time
import torch
from model import MultiModal
from config import parse_args
from data_helper import create_dataloaders_kfolder
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from util import EMA,FGM,PGD,SWA


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    kfolder=5
    train_dataloaders, val_dataloaders = create_dataloaders_kfolder(args,kfolder)
    for folder,(train_dataloader, val_dataloader) in enumerate(zip(train_dataloaders, val_dataloaders)):
        
        # 2. build model and optimizers
        model = MultiModal(args)
        checkpoint = torch.load('save/pretrain.bin', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        optimizer, scheduler = build_optimizer(args, model)
        if args.device == 'cuda':
            model = torch.nn.parallel.DataParallel(model.to(args.device))

        ema=EMA(model,0.999)
        ema.register()
        fgm=FGM(model)

        # 3. training
        step = 0
        start_time = time.time()
        num_total_steps = len(train_dataloader) * args.max_epochs
        for epoch in range(args.max_epochs):
            for batch in train_dataloader:
                model.train()
                loss, accuracy, _, _ = model(inputs=batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()

                fgm.attack()
                loss_attack, _, _, _=model(inputs=batch)
                loss_attack = loss_attack.mean()
                loss_attack.backward()
                fgm.restore()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                ema.update()

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Folder {folder} Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            ema.apply_shadow()
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Folder {folder} Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

            # 5. save checkpoint
            mean_f1 = results['mean_f1']
            
            if epoch >= 3:
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_folder{folder}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
            ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)
    
    model = MultiModal(args).to(args.device)
    SWA(model,args.savedmodel_path)

if __name__ == '__main__':
    main()
