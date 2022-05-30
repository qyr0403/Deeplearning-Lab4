import os
import jieba
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
skip=True
datapath="datasets/online_shopping_10_cats.csv"
label_id=0
label_map={}
length={}
with open(datapath,'r') as f:
    for item in tqdm(f):
        if skip:
            skip=False
            continue
        item=item.replace('\ufeff','')
        first_split=item.find(',')
        text_begin=first_split+3
        class_name=item[:first_split]
        text=item[text_begin:].strip('\n')

        if class_name not in label_map:
            label_map[class_name]=label_id
            label_id+=1
        #target_id=label_map[class_name]
        #text=jieba.lcut(text)
        t_length=len(jieba.lcut(text))
        if t_length not in length:
            length[t_length]=0
        else:
            length[t_length]+=1

lengths=sorted(length.keys())
frequencys=[length[id] for id in lengths]
plt.plot(lengths,frequencys)
plt.xlabel("sentence length")
plt.ylabel("number of sentences")
plt.savefig("frequency.jpg")
#print(max(length))
#print(label_map)
        #print(class_name)


with open("label_map.tsv",'w') as f:
    for k,v in label_map.items():
        f.write(k+'\t'+str(v)+'\n')