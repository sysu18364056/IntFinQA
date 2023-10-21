
from importlib.metadata import distribution
import re
import json
import nltk
import numpy as np
import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk import word_tokenize
from nltk.corpus import stopwords
import time



import openai
import openpyxl
import pandas as pd
import os
from IPython.display import Markdown

def read_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

def display_data(data):
    question=[]
    for domain in data:
        question.append(domain['qa']['question'])
    return question


dataPath1="train.json"
print("Loading data ...", dataPath1)
data_all1 = read_data(dataPath1)
question_sentence1=display_data(data_all1)
len_1=len(question_sentence1)

dataPath2="dev.json"
print("Loading data ...", dataPath2)
data_all2 = read_data(dataPath2)
question_sentence2=display_data(data_all2)
len_2=len(question_sentence2)

dataPath3="test.json"
print("Loading data ...", dataPath3)
data_all3 = read_data(dataPath3)
question_sentence3=display_data(data_all3)
len_3=len(question_sentence3)


#step1:load the data from FinQA
stop_words = set(stopwords.words('english'))

texts=[]
for question in question_sentence1:
    question=re.sub(r'[^\w\s]','',question)
    question = re.sub(r'\d+', '', question)
    question=word_tokenize(question)
    question=[w for w in question if w not in stop_words]
    texts.append(question)

for question in question_sentence2:
    question=re.sub(r'[^\w\s]','',question)
    question = re.sub(r'\d+', '', question)
    question=word_tokenize(question)
    question=[w for w in question if w not in stop_words]
    texts.append(question)

for question in question_sentence3:
    question=re.sub(r'[^\w\s]','',question)
    question = re.sub(r'\d+', '', question)
    question=word_tokenize(question)
    question=[w for w in question if w not in stop_words]
    texts.append(question)



# step2:construct the dictionary
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# step 3:prepare the LDA model
num_topics = 7
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)


def perplexity_visible_model(topic_num,corpus):
        x_list = []
        y_list = []
        for i in range(1,topic_num):
            lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i)
            try:
                perplexity = lda_model.log_perplexity(corpus)
                print(perplexity)
                x_list.append(i)
                y_list.append(perplexity)
            except Exception as e:
                print(e)
        plt.plot(x_list,y_list)
        plt.xlabel('num topics')
        plt.ylabel('perplexity score')
        plt.legend(('perplexity_values'), loc='best')
        plt.savefig('perplexity.jpg')



def visible_model(topic_num,dictionary,texts):
        x_list = []
        y_list = []
        for i in range(1,topic_num):
            lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary,num_topics=i)
            cv_tmp = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
            x_list.append(i)
            y_list.append(cv_tmp.get_coherence())
        plt.plot(x_list, y_list)
        plt.xlabel('num topics')
        plt.ylabel('coherence score')
        plt.legend(('coherence_values'), loc='best')
        plt.show()
        plt.savefig('conherence.jpg')


def visibel(lda_model):
        
        lda = lda_model
        vis_data = gensimvis.prepare(lda, corpus, dictionary)
        pyLDAvis.show(vis_data, open_browser=False, local=False)




# step 4: show the topic distribution
topic_words=[]
for topic_id, topic_keywords in lda_model.show_topics(num_topics=num_topics, formatted=False):
    print(f"Topic {topic_id}:")
    topic_word=[]
    for word, probability in topic_keywords:
        topic_word.append(word)
        print(f"- {word} (prob: {probability:.2f})")
    topic_words.append(topic_word)




#step 5: get the intent description from the LDA model
openai.api_key = " "   #enter your api_key
intents=[]
print(len(topic_words))
for topic_word in topic_words:
    intent=[]
    result=topic_word
    data=[str(item) for item in result ]
    result=",".join(data)

    prompt="A LDA model was used to process a bunch of interrogative sentences in the financial domain, where the words under one of the themes were"+result+"Please use as few phrases as possible rather than questions to indicate the intent of the questions in the themes, an example of phrases is given here, ask about changes in bank ratios, just return the phrases like the example"
    completion = openai.Completion.create(
        engine="text-davinci-003",  
        prompt=prompt,
        max_tokens=800,  
         temperature=0.2
    )
    generated_text=completion.choices[0].text.strip()
    answer_index=generated_text.find("。")+1
    answer=generated_text[answer_index:].strip()
    intent.append(answer)
    intents.append(intent)
    time.sleep(20)

print(topic_words[0])
print(intents[0])


# step 6: get the topic of each sentence
indexs=[]
for text in texts:
    text_bow = dictionary.doc2bow(text)
    topic_distribution = lda_model.get_document_topics(text_bow)
    distribution=[]
    for dis in topic_distribution:
        distribution.append(dis[1])
    index=np.argmax(distribution)
    indexs.append(index)


#perplexity_visible_model(topic_num=10,corpus=corpus)
#visible_model(topic_num=7,dictionary=dictionary,texts=texts)


#step 7: choose the implied intent for each sentence 
for k,domain in enumerate(data_all1):
   index=int(indexs[k])
   domain['qa']['intent']=intents[index][0]

for k,domain in enumerate(data_all2):
   index=int(indexs[len_1+k])
   domain['qa']['intent']=intents[index][0]

for k,domain in enumerate(data_all3):
   index=int(indexs[len_1+len_2+k])
   domain['qa']['intent']=intents[index][0]


filename1='train_temp.json'
filename2='dev_temp.json'
filename3='test_temp.json'


with open(filename1,'w') as f: 
    json.dump(data_all1,f,indent=4)

with open(filename2,'w') as f: 
    json.dump(data_all2,f,indent=4)

with open(filename3,'w') as f: 
    json.dump(data_all3,f,indent=4)

print("Display.. done") 
visibel(lda_model=lda_model)