import torch
from torch.utils.data import SequentialSampler, DataLoader
from config import Config
from dataset import *
from category_id_map import lv2id_to_category_id
from model import *
from utils import *
from tqdm.autonotebook import tqdm

if __name__ == '__main__':
    config=Config()
    config.annotation_path='./data/annotations/test_b.json'
    config.feature_path='./data/zip_feats/test_b.zip'

    dataset1 = MyDataset1(config,'test')
    sampler = SequentialSampler(dataset1)
    dataloader1 = DataLoader(dataset1,
                                batch_size=config.val_batch_size,
                                sampler=sampler,
                                drop_last=False)
    dataset2 = MyDataset2(config,'test')
    sampler = SequentialSampler(dataset2)
    dataloader2 = DataLoader(dataset2,
                                batch_size=config.val_batch_size,
                                sampler=sampler,
                                drop_last=False)
    model1=Model1(config).to(config.device)
    model1=SWA(model1,'./check/train1')
    
    model2=Model2(config).to(config.device)
    model2=SWA(model2,'./check/train2')
    
    models=[model1,model2]
    dataloaders=[dataloader1,dataloader2]
    predictions = []
    for i,(model,dataloader) in enumerate(zip(models,dataloaders)) :
        model.eval()
        predictions.append([])
        tqdm_loader=tqdm(dataloader)
        tqdm_loader.set_description('inference...')
        with torch.no_grad():
            for batch in tqdm_loader:
                y = model(*batch)
                predictions[i].append(y)
    pred=[]
    l=len(predictions[0])
    for i in range(l):
        p=predictions[0][i]*0.5+predictions[1][i]*0.5
        pred_label=p.argmax(1)
        pred.extend(pred_label.cpu().numpy())
        
    with open('./data/result.csv', 'w') as f:
        for pred_label_id, ann in zip(pred, dataset1.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


