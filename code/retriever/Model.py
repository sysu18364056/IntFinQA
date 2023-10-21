import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from config import parameters as conf

if conf.pretrained_model == "bert":
    from transformers import BertModel
elif conf.pretrained_model == "roberta":
    from transformers import RobertaModel


class Bert_model(nn.Module):

    def __init__(self, hidden_size, dropout_rate):

        super(Bert_model, self).__init__()

        self.hidden_size = hidden_size

        if conf.pretrained_model == "bert":
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)
        elif conf.pretrained_model == "roberta":
            self.bert = RobertaModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir)

        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)

        self.cls_final = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, is_training, input_ids, input_mask, segment_ids,intent_input_ids,intent_input_mask,intent_segment_ids,device):
        
        # bert_outputs = self.bert(
        #     input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        # bert_sequence_output = bert_outputs.last_hidden_state

        # bert_pooled_output = bert_sequence_output[:, 0, :]
        
        # pooled_output = self.cls_prj(bert_pooled_output)
        # pooled_output = self.cls_dropout(pooled_output)

        # logits = self.cls_final(pooled_output)
        

        
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        bert_sequence_output = bert_outputs.last_hidden_state


        bert_output=self.bert(input_ids=intent_input_ids,attention_mask=intent_input_mask,token_type_ids=intent_segment_ids)
        bert_seq_output=bert_output.last_hidden_state
        bert_pool_output=bert_seq_output[:,0,:]
        
        
        bert_pooled_output=bert_sequence_output
        
        #new code part: attention 
        bert_question_output=bert_sequence_output.transpose(-1,-2)
        #print(bert_pooled_output.size())
        #print(bert_pooled_output.size())  #[4,768]
        #print(bert_question_output.size())  #[512,768]
        
        #sim_scores=nn.functional.cosine_similarity(bert_pooled_output,bert_question_output)
        sim_scores=torch.matmul(bert_pooled_output,bert_question_output)
        attn_weights=torch.softmax(sim_scores,dim=-1)
        context_vector=torch.matmul(attn_weights,bert_pooled_output)
        
        #bert_pooled_output=context_vector[:,0]+context_vector[:,1]
        #print(bert_pool_output.size())
        #print(context_vector[:,0].size())
        if bert_pool_output.size()!=context_vector[:,0].size():
            print(bert_pool_output.size())
            print(context_vector[:,0].size())
            bert_pool_output=bert_pool_output.resize_(context_vector[:,0].size())
            print(bert_pool_output.size())
            print(context_vector[:,0].size())
        bert_pooled_output=context_vector[:,0]+context_vector[:,1]+bert_pool_output  



        #attn_weights=nn.functional.softmax(sim_scores,dim=0)
        #attn_weights=nn.functional.softmax(sim_scores,dim=-1)
        #attn_weights=attn_weights.unsqueeze(-1).expand(bert_question_output.size())
        #bert_pooled_output=(attn_weights*bert_question_output).sum(0)

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output) 

        return logits
