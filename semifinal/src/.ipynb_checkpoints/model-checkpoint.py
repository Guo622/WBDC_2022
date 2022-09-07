import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer
from transformers.models.bert.modeling_bert import (
    BertConfig,BertPreTrainedModel, BertEmbeddings, BertEncoder,BertOnlyMLMHead
) 
from swin import swin_tiny
from category_id_map import CATEGORY_ID_LIST
import math

class NCELoss(nn.Module):
    def __init__(self, t=0.07) -> None:
        super().__init__()
        self.t = t

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        scores = z1 @ z2.t()
        labels = torch.arange(len(z1), device=z1.device)
        return F.cross_entropy(scores/self.t, labels)
    
class CrossAttention(nn.Module):
    def __init__(self, text_features, input_features) -> None:
        super().__init__()
        self.fc = nn.Linear(text_features+input_features, 1)  #有待改进
        self.sigmoid=nn.Sigmoid()
        self.text_features = text_features

    def forward(self, text: torch.Tensor, input: torch.Tensor):
        n, t, _ = input.shape
        text_expand = text.unsqueeze(1).expand([n, t, self.text_features])
        w = self.sigmoid(self.fc(torch.cat([text_expand, input], dim=2))) 
        r = w*input
        return r.sum(dim=1)
    

    
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
    def __init__(self ,args,mlm_probability=0.25):
        self.mlm_probability = mlm_probability
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        
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
    
    def torch_shuf_video(self, video_feature,video_label,video_mask):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        video_label=video_label[shuf_index]
        video_mask=video_mask[shuf_index]
        return video_feature, label,video_label,video_mask

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

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
    
class VisualBackbone(nn.Module):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.args=args
        self.mode=mode
        self.visual_backbone=swin_tiny(args.swin_pretrained_path)
        
        bert_output_size = 768
        bert_config=BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        if mode == 'pretrain':
            self.vm=MaskVideo()
            self.vm_header=VisualOnlyMLMHead(bert_config)
            
        self.video_projection =nn.Sequential(
            nn.Linear(args.frame_embedding_size, args.bert_embedding_size),  
            nn.ReLU()
        )
        
        self.classifier=nn.Linear(bert_output_size,len(CATEGORY_ID_LIST))
        
    def forward(self,inputs):
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        if self.mode == 'pretrain':
            device =inputs['frame_input'].device
            vm_input=inputs['frame_input']
            input_feature,video_label=self.vm.torch_mask_frames(inputs['frame_input'].cpu(),inputs['frame_mask'].cpu())
            inputs['frame_input']=input_feature.to(device)
            video_label=video_label.to(device)
        
        video_feature = self.video_projection(inputs['frame_input'])
        
        if self.mode == 'pretrain':
            vm_output=self.vm_header(video_feature)
            vm_loss=self.calculate_mfm_loss(vm_output,vm_input,inputs['frame_mask'],video_label,normalize=False)
            return vm_loss
        
        

        mask_expanded=inputs['frame_mask'].unsqueeze(-1).expand(video_feature.size()).float()
        sum_embeddings=torch.sum(video_feature*mask_expanded,1)
        sum_mask=mask_expanded.sum(1).clamp(min=1e-9)
        features_mean=sum_embeddings/sum_mask
        
        prediction= self.classifier(features_mean)

        return self.cal_loss(prediction, inputs['label'])
    
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
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

class TextBackbone(nn.Module):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.args=args
        self.mode=mode
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        
        bert_output_size = 768
        bert_config=self.bert.config
        if mode == 'pretrain':
            self.lm_cls=BertOnlyMLMHead(bert_config)
            self.lm=MaskLM(args)
            
        self.vocab_size=bert_config.vocab_size
        
        self.classifier=nn.Linear(bert_output_size,len(CATEGORY_ID_LIST))
        
    def forward(self,inputs):
        
        if self.mode == 'pretrain':
            device =inputs['text_input'].device
            input_ids_mask,lm_label=self.lm.torch_mask_tokens(inputs['text_input'].cpu())
            inputs['text_input']=input_ids_mask.to(device)
            lm_label=lm_label[:,1:].to(device)
            
        text_emb = self.bert.embeddings(input_ids=inputs['text_input'])
        
        mask = inputs['text_mask'][:, None, None, :]
        mask = (1.0 - mask) * (-10000.0)
        
        encoder_outputs = self.bert.encoder(text_emb, attention_mask=mask)['last_hidden_state'] 
        
        if self.mode == 'pretrain':
            text_len=text_emb.shape[1]
            lm_prediction_scores=self.lm_cls(encoder_outputs)[:,1:text_len,:]
            pred=lm_prediction_scores.contiguous().view(-1,self.vocab_size)
            lm_loss=nn.CrossEntropyLoss()(pred,lm_label.contiguous().view(-1))
            return lm_loss
        
        mask_expanded=inputs['text_mask'].unsqueeze(-1).expand(encoder_outputs.size()).float()
        sum_embeddings=torch.sum(encoder_outputs*mask_expanded,1)
        sum_mask=mask_expanded.sum(1).clamp(min=1e-9)
        features_mean=sum_embeddings/sum_mask
        
        prediction= self.classifier(features_mean)

        return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    
class MultiModal(nn.Module):
    def __init__(self, args,mode='train'):
        super().__init__()
        self.args=args
        
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.visual_backbone =swin_tiny(args.swin_pretrained_path) 
        bert_output_size = 768
        bert_config=self.bert.config
        self.mode=mode
        
        self.video_projection =nn.Sequential(
            nn.Linear(args.frame_embedding_size, args.bert_embedding_size),  
            nn.ReLU()
        )
        self.classifier=nn.Linear(bert_output_size,len(CATEGORY_ID_LIST))
        if mode=='pretrain':
            self.lm_cls=BertOnlyMLMHead(bert_config)
            self.lm=MaskLM(args)
            self.vm=MaskVideo()
            self.vm_header=VisualOnlyMLMHead(bert_config)
            self.sv=ShuffleVideo()
            self.itm_fc=nn.Linear(bert_config.hidden_size,1)
        
        self.vocab_size=bert_config.vocab_size


    def init_hidden(self, n, hidden_size):
        return torch.zeros((2*2, n, hidden_size)).cuda(),\
            torch.zeros((2*2, n, hidden_size)).cuda()

    def forward(self, inputs, inference=False):
        
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        
        if self.mode=='pretrain':
            device =inputs['frame_input'].device
            
            input_ids_mask,lm_label=self.lm.torch_mask_tokens(inputs['text_input'].cpu())
            inputs['text_input']=input_ids_mask.to(device)
            lm_label=lm_label[:,1:].to(device)

            vm_input=inputs['frame_input']
            input_feature,video_label=self.vm.torch_mask_frames(inputs['frame_input'].cpu(),inputs['frame_mask'].cpu())
            inputs['frame_input']=input_feature.to(device)
            video_label=video_label.to(device)
            
            input_feature,video_text_match_label,video_label,video_mask=self.sv.torch_shuf_video(inputs['frame_input'].cpu(),video_label.cpu(),inputs['frame_mask'].cpu())
            inputs['frame_input']=input_feature.to(device)
            inputs['frame_mask']=video_mask.to(device)
            video_text_match_label=video_text_match_label.to(device)
            video_label=video_label.to(device)

        # text=self.bert(input_ids=inputs['text_input'], attention_mask=inputs['text_mask'])[0]
        
        text_emb = self.bert.embeddings(input_ids=inputs['text_input'])
        
        video_feature = self.video_projection(inputs['frame_input'])
        video_emb=self.bert.embeddings(inputs_embeds=video_feature)

        embedding_concat=torch.cat([text_emb,video_emb],dim=1)

        mask = torch.cat([inputs['text_mask'],inputs['frame_mask']], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * (-10000.0)
        
        encoder_outputs = self.bert.encoder(embedding_concat, attention_mask=mask)['last_hidden_state'] 
        
        if self.mode=='pretrain':
            text_len=text_emb.shape[1]
            lm_prediction_scores=self.lm_cls(encoder_outputs)[:,1:text_len,:]
            pred=lm_prediction_scores.contiguous().view(-1,self.vocab_size)
            lm_loss=nn.CrossEntropyLoss()(pred,lm_label.contiguous().view(-1))

            vm_output=self.vm_header(encoder_outputs[:,text_len:,:])
            vm_loss=self.calculate_mfm_loss(vm_output,vm_input,inputs['frame_mask'],video_label,normalize=False)

            pred_itm=self.itm_fc(encoder_outputs[:,0,:])
            itm_loss=nn.BCEWithLogitsLoss()(pred_itm.view(-1),video_text_match_label.view(-1))

            return lm_loss/1.25/3+vm_loss/3/3+itm_loss/3,lm_loss,vm_loss,itm_loss
        

        mask_=torch.cat([inputs['text_mask'],inputs['frame_mask']], 1)
        mask_expanded=mask_.unsqueeze(-1).expand(encoder_outputs.size()).float()
        sum_embeddings=torch.sum(encoder_outputs*mask_expanded,1)
        sum_mask=mask_expanded.sum(1).clamp(min=1e-9)
        features_mean=sum_embeddings/sum_mask
        
        prediction= self.classifier(features_mean)
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])
        

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
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


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
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
