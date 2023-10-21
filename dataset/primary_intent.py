from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import scipy.io as sio
import nltk
import pdb
import random
import csv
import string
import contractions
import json
import random
import torch
from torch import nn

from transformers import BertModel,BertTokenizer,BertConfig



def save_data(data, file_path):
    # save data to disk
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return 


def consine_similarity(x,y): 
    intent2que=[]
    for i in x:
        score=[]
        for j in y:
            res=sum(i*j)
            res_x=np.sqrt(sum(i*i))
            res_y=np.sqrt(sum(j*j))
            cos=res/(res_x*res_y)
            score.append(cos)  
        #print(score)   #13083
        res=[k for k,n in enumerate(score) if n==max(score)]
        res=res[0]
        intent2que.append(res)
    return intent2que
       

def display_data1(data,intent):
    datasetName = 'BANKING77'
    data = data[datasetName]
    dict={}
    for domain in data:
        for d in data[domain]:
            if d[1][0] in intent:
                if  d[1][0] not in dict:
                    dict[d[1][0]]=[]
                    dict[d[1][0]].append(d[0])
                else:
                    dict[d[1][0]].append(d[0])
    
    return dict


def display_data2(data):
    question=[]
    for domain in data:
        question.append(domain['qa']['question'])
    return question



def read_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


def cal_the_mean(matrix_intent,intent_num):
    matrix=[]
    pos=0
    for k,num in enumerate(intent_num):
        if k==0:
            res=matrix_intent[pos:num]
            matrix.append(res.mean(axis=0))
            pos=pos+num
        else:
            res=matrix_intent[pos:pos+num]
            matrix.append(res.mean(axis=0))
            pos=pos+num  
    return matrix

def bert_train(sentences):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    done_sentences=[]
    for i in sentences:
        i="[ClS] "+i+" [SEP]"
        done_sentences.append(i)

    tokens,segments,input_masks=[],[],[]
    for text in done_sentences:
        tokenized_text=tokenizer.tokenize(text)
        indexed_tokens=tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0]*len(indexed_tokens))
        input_masks.append([1]*len(indexed_tokens))
    max_len=max([len(single) for single in tokens])
    
    for j in range(len(tokens)):
        padding=[0]*(max_len-len(tokens[j]))
        tokens[j]+=padding
        segments[j]+=padding
        input_masks[j]+=padding
    
    tokens_tensor=torch.tensor(tokens)
    segments_tensor=torch.tensor(segments)
    input_masks_tensors=torch.tensor(input_masks)
        
    textNet = BertTextNet(code_length=64)
    text_hashcodes=textNet(tokens_tensor,segments_tensor,input_masks_tensors)
    return text_hashcodes
    
    
class BertTextNet(nn.Module):
    def __init__(self,code_length):
        super(BertTextNet, self).__init__()
 
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.textExtractor = BertModel.from_pretrained(
            'bert-base-uncased', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size
 
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings
 
 



    

intentname=['card payment fee charged','card payment wrong exchange rate','country support','disposable card limits','exchange charge','exchange rate','extra charge on statement','fiat currency support','pending card payment','supported cards and currencies','terminate account','top up limits','transfer timing',
            'visa or mastercard','wrong amount of cash received','wrong exchange rate for cash withdrawal']


dataPath1 = "bank77/dataset.json"
dataPath2="FinQA/dataset/train.json"
print("Loading data ...", dataPath1)
data1 = read_data(dataPath1)
data2 = read_data(dataPath2)
dict_intent=display_data1(data1,intentname)
question_sentence=display_data2(data2)


intent_sentence=[]
intent_num=[]
intents=[]
for intent in dict_intent:
    print(intent)
    intents.append(intent)
    intent_sentence+=dict_intent[intent]
    intent_num.append(len(dict_intent[intent]))

intent_sentence_vector=bert_train(intent_sentence)
question_sentence_vector=bert_train(question_sentence)

intent_sentence_vector=intent_sentence_vector.detach().numpy()
question_sentence_vector=question_sentence_vector.detach().numpy()
intent_sentence_vector=cal_the_mean(intent_sentence_vector,intent_num) 


intent2que=consine_similarity(question_sentence_vector,intent_sentence_vector)


for k,domain in enumerate(data2):
   domain['qa']['intent']=intents[intent2que[k]]
   if k==0:
    print(intent2que[k])
    print(intents[intent2que[k]])
    print(domain['qa']['question'])

filename='train_primary.json'


with open(filename,'w') as f: 
    json.dump(data2,f,indent=4)

print("Display.. done") 


