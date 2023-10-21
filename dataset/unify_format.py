import json
import re
# we use this file to unify the format of the descriptions from ChatGPT 


def read_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data
    

dataPath="train_temp.json"
print("Loading data ...", dataPath)
data_all = read_data(dataPath)


for domain in data_all:
    intent_origin=domain['qa']['intent']
    intent_origin = re.sub(r'\d+', '', intent_origin)
    #print(type(intent_origin))
    print(intent_origin)
    intent_origin=intent_origin.replace(';','')
    intent_origin=intent_origin.replace('\n','').replace('\r','').replace('.',';').replace('-',';').replace(',',';')
    domain['qa']['intent']=intent_origin.strip(':; ')

filename='train_implied.json'

with open(filename,'w') as f: 
    json.dump(data_all,f,indent=4)

print("Display.. done") 
