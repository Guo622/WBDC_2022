import torch
import numpy as np


class Config():
    best_scores=0.5
    def __init__(self) -> None:
        self.annotation_path = './data/annotations/labeled.json'
        self.feature_path = './data/zip_feats/labeled.zip'
        self.bert_path="hfl/chinese-roberta-wwm-ext-large"
        self.max_title_len = 80
        self.max_asr_len = 250
        self.max_video_T = 32
        self.max_ocr_T = 15
        self.max_ocr_len = 50
        self.max_ocr_concatlen=200
        self.max_title_asr_len=256
        self.max_concatall_len=384
        self.seed=2022
        self.train_batch_size=16  #16   
        self.pretrain_batch_size=48
        self.val_batch_size=64
        self.video_features = 768
        self.embedding_size = 1024  #改########
        self.lstm_video_hidden_size = 1024  # 1024
        self.lstm_ocr_hidden_size = 768  # 512
        self.batch_size = 32
        self.epochs = 10
        self.lr_bert = 5e-6  #1e-6 
        self.lr_base = 2e-4   #1e-4
        self.learning_rate=5e-5    #5e-5  3e-5
        self.weight_decay = 0.01 
        self.max_steps = 50000
        self.adam_epsilon = 1e-6
        self.warmup_steps = 1000  #2000  3000
        self.device = \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k_folder=10
        self.vocab_size=21128
    @staticmethod
    def seed_torch(seed=2022):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
