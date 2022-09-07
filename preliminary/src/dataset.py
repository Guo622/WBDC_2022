import json
import torch
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import Config
import random
from category_id_map import *

class MyDataset1(Dataset):  #4

    def __init__(self, config: Config,mode='train') -> None:
        super().__init__()
        self.device = config.device
        self.max_title_len, self.max_asr_len = config.max_title_len, config.max_asr_len
        self.max_video_T, self.video_features = config.max_video_T, config.video_features
        self.max_ocr_T, self.max_ocr_len,self.max_ocr_concatlen = config.max_ocr_T, config.max_ocr_len,config.max_ocr_concatlen
        self.max_tilte_asr_len=config.max_title_asr_len
        with open(config.annotation_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.feature_path = config.feature_path
        self.handler = ZipFile(self.feature_path, 'r')
        self.n = len(self.anns)
        self.mode=mode
        self.tokenizer = BertTokenizer.from_pretrained(
           config.bert_path,use_fast=True)

    def __del__(self):
        self.handler.close()

    def processText(self, text, max_len):
        x = self.tokenizer(text, return_tensors='pt', max_length=max_len,
                           truncation=True, padding='max_length')
        return x['input_ids'].squeeze(0), x['token_type_ids'].squeeze(0), x['attention_mask'].squeeze(0)

    def __getitem__(self, index):
        asr = self.anns[index]['asr']
        title = self.anns[index]['title']
        ocr=''
        for o in self.anns[index]['ocr']:
            ocr=ocr+o['text']

        asr=self.processText(asr,128)
        title=self.processText(title,128)
        ocr=self.processText(ocr,128)

        sep=torch.tensor(self.tokenizer.sep_token_id).view(1)
        att=torch.tensor(1).view(1)

        text_input_ids=torch.concat([title[0],sep,asr[0][1:],sep,ocr[0][1:],sep])
        text_attention_mask=torch.concat([title[2],att,asr[2][1:],att,ocr[2][1:],att])

        if self.mode=='pretrain':
            return (
                text_input_ids.to(self.device),
                text_attention_mask.to(self.device),
            )

        fname = self.anns[index]['id']
        fname += '.npy'
        video = torch.zeros((self.max_video_T, self.video_features))
        fea = torch.from_numpy(np.load(BytesIO(self.handler.read(fname))))

        num_frames = fea.shape[0]
        if num_frames <= self.max_video_T:
            video[:num_frames] = fea
        else:
            if self.mode != 'test':
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_video_T]
                select_inds = sorted(select_inds)
            else:
                step=num_frames//self.max_video_T
                select_inds=list(range(0,num_frames,step))
                select_inds=select_inds[:self.max_video_T]
            for i, j in enumerate(select_inds):
                video[i] = fea[j]

        if self.mode=='train':
            label = category_id_to_lv2id(self.anns[index]['category_id'])
            label = torch.tensor(label)
            return(            
                video.to(self.device),
                text_input_ids.to(self.device),
                text_attention_mask.to(self.device),
                label.to(self.device)
            )
        return(            
                video.to(self.device),
                text_input_ids.squeeze(0).to(self.device),
                text_attention_mask.squeeze(0).to(self.device),

            )


    def __len__(self):
        return len(self.anns)


class MyDataset2(Dataset): #5

    def __init__(self, config: Config,mode='train') -> None:
        super().__init__()
        self.device = config.device
        self.max_video_T, self.video_features = config.max_video_T, config.video_features
        self.max_concatall_len=config.max_concatall_len
        with open(config.annotation_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.feature_path = config.feature_path
        self.handler = ZipFile(self.feature_path, 'r')
        self.n = len(self.anns)
        self.mode=mode
        self.tokenizer = BertTokenizer.from_pretrained(
           config.bert_path,use_fast=True)

    def __del__(self):
        self.handler.close()

    def processText(self, text, max_len):
        x = self.tokenizer(text, return_tensors='pt', max_length=max_len,
                           truncation=True, padding='max_length')
        return x['input_ids'].squeeze(0), x['token_type_ids'].squeeze(0), x['attention_mask'].squeeze(0)

    def __getitem__(self, index):
        asr = self.anns[index]['asr']
        title = self.anns[index]['title']
        ocr=''
        for o in self.anns[index]['ocr']:
            ocr=ocr+o['text']

        asr=self.processText(asr,128)
        title=self.processText(title,128)
        ocr=self.processText(ocr,128)

        sep=torch.tensor(self.tokenizer.sep_token_id).view(1)
        att=torch.tensor(1).view(1)

        text_input_ids=torch.concat([title[0],sep,asr[0][1:],sep,ocr[0][1:],sep])
        text_attention_mask=torch.concat([title[2],att,asr[2][1:],att,ocr[2][1:],att])

        fname = self.anns[index]['id']
        fname += '.npy'
        video = torch.zeros((self.max_video_T, self.video_features))
        fea = torch.from_numpy(np.load(BytesIO(self.handler.read(fname))))
        video_mask = torch.ones((self.max_video_T,)).long()

        num_frames = fea.shape[0]
        if num_frames <= self.max_video_T:
            video[:num_frames] = fea
            video_mask[num_frames:] = 0

        else:
            if self.mode == 'train':
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_video_T]
                select_inds = sorted(select_inds)
            else:
                step=num_frames//self.max_video_T
                select_inds=list(range(0,num_frames,step))
                select_inds=select_inds[:self.max_video_T]
            for i, j in enumerate(select_inds):
                video[i] = fea[j]


        if self.mode=='train':
            label = category_id_to_lv2id(self.anns[index]['category_id'])
            label = torch.tensor(label)
            return(            
                video.to(self.device),
                video_mask.to(self.device),
                text_input_ids.squeeze(0).to(self.device),
                text_attention_mask.squeeze(0).to(self.device),
                label.to(self.device)
            )
        return(            
                video.to(self.device),
                video_mask.to(self.device),
                text_input_ids.squeeze(0).to(self.device),
                text_attention_mask.squeeze(0).to(self.device),

            )


    def __len__(self):
        return len(self.anns)
