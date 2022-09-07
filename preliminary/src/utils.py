import copy
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,random_split,Subset
from dataset import *
from sklearn.metrics import f1_score, accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from category_id_map import *
from config import Config
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
import copy
from tqdm.autonotebook import tqdm


def create_dataloaders(config: Config,mode):
    if mode=='pretrain1':
        dataset=MyDataset1(config,'pretrain')
    elif mode=='pretrain2':
        dataset=MyDataset2(config,'pretrain')
    elif mode=='train1':
        dataset=MyDataset1(config,'train')
    elif mode=='train2':
        dataset=MyDataset2(config,'train')

    if 'pre' in mode:
        bs=config.pretrain_batch_size
    else:
        bs=config.train_batch_size


    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                    batch_size=bs,
                                    sampler=train_sampler,
                                    drop_last=False)
    return train_dataloader

def build_optimizer(config: Config, model: nn.Module,mode):

    no_decay = ['bias', 'LayerNorm.weight']
    bert_params_ids = list(map(id,model.bert.parameters()))
    base_params = filter(
            lambda p: id(p) not in bert_params_ids, model.parameters())
    if mode=='train1':
        
        optimizer_grouped_parameters = [
            {'params':model.bert.parameters(),'lr':5e-6}, #改  #5e-5
            {'params': base_params,'lr':2e-4}  #3e-4
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,25000,1e-6)  #20000

    elif mode=='train2':
        optimizer_grouped_parameters = [
            {'params':model.bert.parameters(),'lr':5e-5}, #改  #5e-5
            {'params': base_params,'lr':2e-4}  #3e-4
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10000,1e-6)  #20000

    elif mode=='pretrain2':

        optimizer_grouped_parameters = [
            {'params':model.bert.parameters(),'lr':config.lr_bert},
            {'params': base_params,'lr':config.lr_base}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate, eps=config.adam_epsilon)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=config.max_steps)

    elif mode=='pretrain1':
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params':model.cls.parameters()}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate, eps=config.adam_epsilon)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=config.max_steps)


    return optimizer, scheduler


def get_kfolder_dataloader(config: Config,mode):

    with open(config.annotation_path, 'r', encoding='utf8') as f:
        anns = json.load(f)
    n=len(anns)
    labels=np.zeros((n,))
    for i,ann in enumerate( anns):
        label = category_id_to_lv2id(ann['category_id'])
        labels[i] = label
    
    skf=StratifiedKFold(n_splits=config.k_folder,shuffle=True,random_state=config.seed)

    if mode=='train1':
        dataset=MyDataset1(config)
    elif mode=='train2':
        dataset=MyDataset2(config)
        
    train_loader=[]
    val_loader=[]
    for train_index,test_index in skf.split(np.zeros((n,1)),labels):
        sub_train=Subset(dataset,train_index)
        sub_val=Subset(dataset,test_index)
        train_sampler = RandomSampler(sub_train)
        val_sampler = SequentialSampler(sub_val)
        sub_train_loader = DataLoader(sub_train,
                                  batch_size=config.train_batch_size,
                                  sampler=train_sampler,
                                  drop_last=False
                                  )
        sub_val_loader = DataLoader(sub_val,
                                batch_size=config.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False
                                )
        train_loader.append(sub_train_loader)
        val_loader.append(sub_val_loader)
        
    return train_loader,val_loader




class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for fname in os.listdir(base_dir):
        if 'model' in fname: 
            model_lists.append(base_dir+'/'+fname)

    model_lists = sorted(model_lists)
    
    return model_lists[:]

def SWA(model,base_dir='./check',k_folder=None):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(base_dir)
    print(f'mode_list:{model_path_list}')

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list:
            model.load_state_dict(torch.load(_ckpt))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash

    if k_folder is None:
        model_name=base_dir+'/swa_train.pth'
    else:
        model_name=base_dir+f'/swa_{k_folder}_train.pth'
    torch.save(swa_model.state_dict(), model_name)

    return swa_model

def SWA_Kfolder(model,k=5,base_dir='./check'):
    models=[]
    range_tqdm=tqdm(range(k)) 
    for i in range_tqdm:
        range_tqdm.set_description(f'swa folder {i}')
        swa_model=SWA(model,base_dir+f'/k_folder/folder{i}',k_folder=i)
        models.append(swa_model)
    return models