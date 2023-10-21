import json
# we use this file to concat the primary intent and the implied intent

def read_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data


filename1='train_primary.json'
filename2='train_implied.json'


data1=read_data(filename1)
data2=read_data(filename2)

intents=[]
for domain in data2:
    intents.append(domain['qa']['intent'])

for k,domain in enumerate(data1):
    temp=domain['qa']['intent']
    domain['qa']['intent']=temp+'/'+intents[k]


filename1='train_intent.json'


with open(filename1,'w') as f: 
    json.dump(data1,f,indent=4)
