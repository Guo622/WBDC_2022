import math
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
from config import Config
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertConfig,BertPreTrainedModel, BertEmbeddings, BertEncoder,BertOnlyMLMHead
) 

class CrossAttention(nn.Module):
    def __init__(self, text_features, input_features) -> None:
        super().__init__()
        self.fc = nn.Linear(text_features+input_features, 1)  #有待改进
        self.sigmoid=nn.Sigmoid()
        self.text_features = text_features

    def forward(self, text: torch.Tensor, input: torch.Tensor):
        n, t, _ = input.shape
        text_expand = text.unsqueeze(1).expand([n, t, self.text_features])
        w = self.sigmoid(self.fc(torch.concat([text_expand, input], dim=2))) 
        r = w*input
        return r.sum(dim=1)

class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.senet = nn.Sequential(
            nn.Linear(in_features=channels,
                      out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio,
                      out_features=channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.senet(x)
        x = torch.mul(x, gates)
        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout=0.3):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)
        return embedding

class MaskVideo():
    def __init__(self, mlm_probability=0.25):
        self.mlm_probability = mlm_probability
        
    def torch_mask_frames(self, video_feature:torch.Tensor, video_mask:torch.Tensor):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2)) 

        return inputs, video_labels_index


class MaskLM():
    def __init__(self ,mlm_probability=0.25):
        self.mlm_probability = mlm_probability
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large",use_fast=True)
        
    def torch_mask_tokens(self, inputs:torch.Tensor, special_tokens_mask = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class ShuffleVideo():
    def __init__(self):
        pass
    
    def torch_shuf_video(self, video_feature,video_label):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        video_label=video_label[shuf_index]
        return video_feature, label,video_label

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Model1(nn.Module):
    def __init__(self, config: Config,mode='train') -> None:
        super().__init__()
        self.mode=mode
        self.embedding_size = config.embedding_size
        self.vocab_size=config.vocab_size
        self.device = config.device
        self.max_ocr_T = config.max_ocr_T
        self.max_ocr_len = config.max_ocr_len
        self.lstm_video_hidden_size = config.lstm_video_hidden_size
        self.lstm_ocr_hidden_size = config.lstm_ocr_hidden_size
        self.bert = BertModel.from_pretrained(config.bert_path)   
        self.lstm1 = nn.LSTM(config.video_features, config.lstm_video_hidden_size, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.1)

        self.cls=BertOnlyMLMHead(BertConfig.from_pretrained(config.bert_path))
        self.lm=MaskLM()
        self.attention1 = CrossAttention(
            self.embedding_size, self.lstm_video_hidden_size*2)

        self.dense=ConcatDenseSE(self.embedding_size+ self.lstm_video_hidden_size*2,1024,4)

        self.fc = nn.Linear(1024,200)
        
        

    def init_hidden(self, n, hidden_size):
        return torch.zeros((2*2, n, hidden_size)).to(self.device),\
            torch.zeros((2*2, n, hidden_size)).to(self.device)

    def forward(self, video, text_input_ids, text_mask):
        

        if self.mode=='pretrain':
            input_ids_mask,lm_label=self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids_mask=input_ids_mask.to(text_input_ids.device)
            lm_label=lm_label[:,1:].to(text_input_ids.device)
            encoder_outputs=self.bert(input_ids=text_input_ids_mask, attention_mask=text_mask)[0]
            lm_prediction_scores=self.cls(encoder_outputs)[:,1:,:]
            return lm_prediction_scores.contiguous().view(-1,self.vocab_size),lm_label.contiguous().view(-1)

        n = len(video)
        text = self.bert(
            input_ids=text_input_ids, attention_mask=text_mask)[0]  #换mean_pool   换 last4

        mask_expanded=text_mask.unsqueeze(-1).expand(text.size()).float()
        sum_embeddings=torch.sum(text*mask_expanded,1)
        sum_mask=mask_expanded.sum(1).clamp(min=1e-9)
        text_mean=sum_embeddings/sum_mask

        self.hidden = self.init_hidden(n, self.lstm_video_hidden_size)
        self.lstm1.flatten_parameters()
        video, _ = self.lstm1(video, self.hidden)

        video = self.attention1(text_mean, video)

        out=self.dense([video, text_mean])
        return self.fc(out)


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class Model2(nn.Module):
    def __init__(self,config:Config, mode='train'):
        super().__init__()
        self.mode=mode
        self.bert=BertModel.from_pretrained(config.bert_path)
        self.video_projection =nn.Sequential(
            nn.Linear(config.video_features, config.embedding_size),  
            nn.ReLU()
        ) 
        self.classifier=nn.Linear(config.embedding_size,200)
        if mode=='pretrain':
            self.bert_config=BertConfig.from_pretrained(config.bert_path)
            self.lm_cls=BertOnlyMLMHead(self.bert_config)
            self.lm=MaskLM()
            self.vm=MaskVideo()
            self.vm_header=VisualOnlyMLMHead(self.bert_config)
            self.sv=ShuffleVideo()
            self.itm_fc=nn.Linear(self.bert_config.hidden_size,1)

        self.vocab_size=config.vocab_size

    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        
        if self.mode=='pretrain':
            input_ids,lm_label=self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids=input_ids.to(text_input_ids.device)
            lm_label=lm_label[:,1:].to(text_input_ids.device)

            vm_input=video_feature
            input_feature,video_label=self.vm.torch_mask_frames(video_feature.cpu(),video_mask.cpu())
            video_feature=input_feature.to(video_feature.device)
            video_label=video_label.to(video_feature.device)

            input_feature,video_text_match_label,video_label=self.sv.torch_shuf_video(video_feature.cpu(),video_label.cpu())
            video_feature=input_feature.to(video_feature.device)
            video_text_match_label=video_text_match_label.to(video_feature.device)
            video_label=video_label.to(video_feature.device)


        text_emb = self.bert.embeddings(input_ids=text_input_ids)
        
        video_feature = self.video_projection(video_feature)
        video_emb=self.bert.embeddings(inputs_embeds=video_feature)

        embedding_concat=torch.concat([text_emb,video_emb],dim=1)

        mask = torch.concat([text_mask,video_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * (-10000.0)
        
        encoder_outputs = self.bert.encoder(embedding_concat, attention_mask=mask)['last_hidden_state']

        if self.mode=='pretrain':
            text_len=text_emb.shape[1]
            lm_prediction_scores=self.lm_cls(encoder_outputs)[:,1:text_len,:]
            pred=lm_prediction_scores.contiguous().view(-1,self.vocab_size)
            lm_loss=nn.CrossEntropyLoss()(pred,lm_label.contiguous().view(-1))

            vm_output=self.vm_header(encoder_outputs[:,text_len:,:])
            vm_loss=self.calculate_mfm_loss(vm_output,vm_input,video_mask,video_label,normalize=False)

            pred_itm=self.itm_fc(encoder_outputs[:,0,:])
            itm_loss=nn.BCEWithLogitsLoss()(pred_itm.view(-1),video_text_match_label.view(-1))


            return lm_loss/1.25/3+vm_loss/3/3+itm_loss/3,lm_loss,vm_loss,itm_loss

        # features_mean = torch.sum(encoder_outputs, 1)/torch.sum(mask,1,keepdim=True)
        mask_=torch.concat([text_mask,video_mask], 1)
        mask_expanded=mask_.unsqueeze(-1).expand(encoder_outputs.size()).float()
        sum_embeddings=torch.sum(encoder_outputs*mask_expanded,1)
        sum_mask=mask_expanded.sum(1).clamp(min=1e-9)
        features_mean=sum_embeddings/sum_mask

        output= self.classifier(features_mean)
        # output=self.classifier(encoder_outputs[:,0])

        return output

    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = F.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = F.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
