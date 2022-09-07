import logging
import random
import os
import copy
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_cosine_schedule_with_warmup
from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)  #weight decay + epoch 10 + warm up 0.6956 (第4轮)
                                                                 #  weight decay + epoch 5 + warm up 0.6949 (第4轮)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    return optimizer, scheduler
    
    
def build_optimizer_pretrain(args,model):
    bert_params_ids = list(map(id,model.bert.parameters()))
    visual_params_ids=list(map(id,model.visual_backbone.parameters()))
    bert_visual_params_ids=bert_params_ids+visual_params_ids
    base_params = filter(
            lambda p: id(p) not in bert_visual_params_ids, model.parameters())
    optimizer_grouped_parameters = [
            {'params':model.bert.parameters(),'lr':5e-5}, #改  #5e-5
            {'params':model.visual_backbone.parameters(),'lr':1e-5},
            {'params': base_params,'lr':2e-4}  #3e-4
        ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    
    return optimizer, scheduler

def build_optimizer_single_train(args,model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    return optimizer, scheduler
    
def build_optimizer_visual_pertrain(args,model):
    visual_params_ids=list(map(id,model.visual_backbone.parameters()))
    base_params = filter(
            lambda p: id(p) not in visual_params_ids, model.parameters())
    optimizer_grouped_parameters = [
            {'params':model.visual_backbone.parameters(),'lr':1e-5},
            {'params': base_params,'lr':2e-4}  #3e-4
        ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    
    return optimizer, scheduler

def build_optimizer_text_pertrain(args,model):
    bert_params_ids = list(map(id,model.bert.parameters()))
    base_params = filter(
            lambda p: id(p) not in bert_params_ids, model.parameters())
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params':model.bert.parameters(),'lr':2e-5}, #改  #5e-5
            {'params': base_params,'lr':2e-4}  #3e-4
        ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results


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
        if 'epoch_3' in fname or 'epoch_4' in fname: 
            model_lists.append(base_dir+fname)

    model_lists = sorted(model_lists)
    
    return model_lists[:]

def SWA(model,base_dir='./save/'):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(base_dir)
    print(f'mode_list:{model_path_list}')

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list:
            model.load_state_dict(torch.load(_ckpt)['model_state_dict'])
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash


    torch.save(swa_model.state_dict(),base_dir+"swa_model.pth")

    return swa_model