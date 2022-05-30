from typing import List, Tuple, Dict, Any
import torch
from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import jieba
import torch
# this is the class for dog-breed-recognition
class Shopping(Dataset):
    def __init__(self,word2id,datapath,mode) -> None:
        self.classes=['书籍','平板','手机',	'水果',	'洗发水',	'热水器',	'蒙牛',	'衣服',	'计算机','酒店']
        self.word2id=word2id
        self.datapath=datapath
        samples=[]
        skip=True
        with open(self.datapath,'r') as f:
            for item in tqdm(f,desc='load {} dataset'.format(mode)):
                if skip:
                    skip=False
                    continue
                item=item.replace('\ufeff','')
                first_split=item.find(',')
                text_begin=first_split+3
                class_name=item[:first_split]
                text=item[text_begin:].strip('\n')
                class_id=self.classes.index(class_name)
                sentence_ids=self.wordtoid(text)
                samples.append({
                    "class_id":class_id,
                    "sentence_ids":sentence_ids
                })
        self.train_samples=[]
        self.dev_samples=[]
        self.test_samples=[]
        for i in range(len(samples)):
            if i % 5==4:
                self.dev_samples.append(samples[i])
            elif i%5==0:
                self.test_samples.append(samples[i])
            else:
                self.train_samples.append(samples[i])
        if mode == 'train':
            self.samples=self.train_samples
        if mode == 'test':
            self.samples=self.test_samples
        if mode == 'dev':
            self.samples=self.dev_samples
        




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.samples[index]


    def __len__(self) -> int:
        return len(self.samples)


    def collate(self, batch):
        class_id=torch.tensor(
            [item["class_id"] for item in batch]
        )
        sentence_ids=torch.tensor(
            [item["sentence_ids"] for item in batch]
        )
        return class_id,sentence_ids

    def wordtoid(self,sentence):
        word_pieces=jieba.lcut(sentence)
        word_ids=[]
        for word in word_pieces:
            if not (word in self.word2id):
                word_ids.append(0)
                continue
            word_ids.append(self.word2id[word])
        word_ids=word_ids+250*[0]
        #word_ids=[self.word2id[word] for word in word_pieces]
        return word_ids[:250]
