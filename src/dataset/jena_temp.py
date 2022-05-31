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
class jena_temp(Dataset):
    def __init__(self,datapath,mode="all") -> None:
        skip=True
        temp=[]
        self.mean=9.4559
        self.std=8.4214
        with open(datapath,'r') as f:
            for item in f:
                if skip:
                    skip=False
                    continue
                citem=item.strip('\n').split(',')
                temp.append(
                    (float(citem[2])-9.4559)/8.4214
                    )
        days=[]
        for id in range(len(temp)):
            if id%144==0:
                days.append([])
            days[-1].append(temp[id])
        days=days[:2919]
        weeks=[]
        for id in range(len(days)):
            if id % 7==0:
                weeks.append([])
            weeks[-1].append(days[id])
        if mode=="train":
            self.weeks=weeks[:263]
        elif mode=="dev":
            self.weeks=weeks[263:313]
        elif mode=="test":
            self.weeks=weeks[313:]
        else:
            self.weeks=weeks
        # weeks :  [weeks,7,144]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.weeks[index]


    def __len__(self) -> int:
        return len(self.weeks)


    def collate(self, batch):
        first_5=torch.tensor([
            item[:5]for item in batch
        ])
        next_2=torch.tensor([
            item[5:]for item in batch
        ])

        return first_5,next_2

    
