import os
import json
import zipfile
import random
import zipfile
import torch
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from io import BytesIO
import numpy as np
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import Dataset,Subset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from category_id_map import category_id_to_lv2id


def create_dataloaders(args,dataset):
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader

def create_dataloaders_pretrain(args,dataset):
    size = len(dataset)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(dataset)
    train_dataloader = dataloader_class(dataset,
                                        batch_size=args.pretrain_batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    return train_dataloader

def create_dataloaders_kfolder(args,k_folder=5):

    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    n=len(anns)
    labels=np.zeros((n,))
    for i,ann in enumerate( anns):
        label = category_id_to_lv2id(ann['category_id'])
        labels[i] = label
    
    skf=StratifiedKFold(n_splits=k_folder,shuffle=True,random_state=args.seed)

    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames)
    train_loader=[]
    val_loader=[]

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
        
    for train_index,test_index in skf.split(np.zeros((n,1)),labels):
        sub_train=Subset(dataset,train_index)
        sub_val=Subset(dataset,test_index)
        train_sampler = RandomSampler(sub_train)
        val_sampler = SequentialSampler(sub_val)
        sub_train_loader = dataloader_class(sub_train,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True
                                  )
        sub_val_loader = dataloader_class(sub_val,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False
                                )
        
        train_loader.append(sub_train_loader)
        val_loader.append(sub_val_loader)
        
    return train_loader,val_loader

class SingleModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 ann_path: str = None,
                 zip_frame_dir: str =None,
                 mode: str = 'frame'):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.zip_frame_dir = zip_frame_dir
        self.mode = mode
            
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        if 'text' in mode:
            self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        # we use the standard image transform as in the offifical Swin-Transformer.
        if 'frame' in mode:
            self.transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_frame]
            select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        if 'frame' in self.mode:
            frame_input, frame_mask = self.get_visual_frames(idx)

        # Step 2, load title tokens
        if 'text' in self.mode:
            title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
            asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'])

            ocr=''
            for o in self.anns[idx]['ocr']:
                ocr=ocr+o['text']
            ocr_input, ocr_mask = self.tokenize_text(ocr)

            sep=torch.tensor(self.tokenizer.sep_token_id).view(1)
            att=torch.tensor(1).view(1)

            text_input=torch.cat([title_input,sep,asr_input[1:],sep,ocr_input[1:],sep])
            text_mask=torch.cat([title_mask,att,asr_mask[1:],att,ocr_mask[1:],att])
        
        # Step 3, summarize into a dictionary
        if 'frame' in self.mode:
            data = dict(
                frame_input=frame_input,
                frame_mask=frame_mask
            )
        if 'text' in self.mode:
            data = dict(
                text_input=text_input,
                text_mask=text_mask
            )
        
    
        # Step 4, load label if not test mode
        if 'train' in self.mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data

class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_frame_dir: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_frame_dir = zip_frame_dir

            
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        # we use the standard image transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']

        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)

        # Step 2, load title tokens
        title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'])
        
        ocr=''
        for o in self.anns[idx]['ocr']:
            ocr=ocr+o['text']
        ocr_input, ocr_mask = self.tokenize_text(ocr)
        
        sep=torch.tensor(self.tokenizer.sep_token_id).view(1)
        att=torch.tensor(1).view(1)
        
        text_input=torch.cat([title_input,sep,asr_input[1:],sep,ocr_input[1:],sep])
        text_mask=torch.cat([title_mask,att,asr_mask[1:],att,ocr_mask[1:],att])
        
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            text_input=text_input,
            text_mask=text_mask
        )
        
    
        # Step 4, load label if not test mode
        if not self.test_mode and 'unlabeled' not in self.zip_frame_dir:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
