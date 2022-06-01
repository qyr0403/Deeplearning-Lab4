from email.policy import strict
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.ShoppingLSTM import ShoppingLSTM
from dataset.shopping import Shopping
from tqdm import tqdm
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from util import set_seed
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def dev(dataloader,model):
    model.eval()
    num_correct = 0
    total_num=0
    with torch.no_grad():
        for class_id,sentence_ids in tqdm(dataloader,desc="inferencing develop set"):
            class_id, sentence_ids = class_id.to(device), sentence_ids.to(device)  
            logits = model(sentence_ids)  
            _, pred = torch.max(logits, dim=1)  
            num_correct += (pred == class_id).sum().item()  
            total_num+=len(class_id)
    acc=num_correct / total_num
    model.train()
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default=None, type=str,help="to load data files")
    parser.add_argument("--checkpoint_save_folder", default=None, type=str,help="to save checkpoint")
    parser.add_argument("--word_embedding_path", required=True,type=str,default=None)
    parser.add_argument("--log_dir", type=str,default=None)
    parser.add_argument("--results_file",type=str,default=None)
    parser.add_argument("--trained_model_name_or_path",type=str,default=None)
    parser.add_argument("--batch_size", default=64, type=int)
    #parser.add_argument("--model", default="vgg11", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    #parser.add_argument("--pre_process", default="resize", type=str)
    parser.add_argument("--optim", default="SGD", type=str)
    parser.add_argument("--hidden_sz", default=200, type=int)
    parser.add_argument("--output_sz", default=10, type=int)
    parser.add_argument("--embedding_name", default="weibo", type=str)
    
    #seed_list=[13,21,87,100,42]
    args = parser.parse_args()
    # build log file
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
        tb = writer

    # set random seed to keep the experiment reproduceble
    set_seed(args.seed)

    ## read embedding
    word_embedding=[]   #store word embedding
    #word=[]     #store word
    word2id={}  # map word->id
    skip=True
    id=0
    with open(args.word_embedding_path,'r') as f:
        for item in tqdm(f,desc="load word embeddings from {}".format(args.word_embedding_path)):
            if skip:
                skip=False
                continue
            split_id=item.find(' ')
            word=item[:split_id]
            str_embedding=item[split_id+1:].strip().split(' ')# list of str, for example ['-0.11222','-0.61']
            word_embedding.append(str_embedding)
            word2id[word]=id
            id+=1
    
    word_embedding=torch.tensor([
        [float(str_number) for str_number in str_embedding] for str_embedding in word_embedding
    ])
    
    model=ShoppingLSTM(
        hidden_sz=args.hidden_sz,
        output_sz=args.output_sz,
        embedding=word_embedding
        )
    
    if args.trained_model_name_or_path is not None:
        print("load pretrained weight...")
        st=torch.load(args.trained_model_name_or_path)
        # can not very strict because the architecture is not the same
        model.load_state_dict(st,strict=True) 
    
    train_dataset=Shopping(
            word2id=word2id,datapath=args.datapath,mode="train"
        )
    dev_dataset=Shopping(
            word2id=word2id,datapath=args.datapath,mode="dev"
        )
    test_dataset=Shopping(
            word2id=word2id,datapath=args.datapath,mode="test"
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,collate_fn=train_dataset.collate)
    dev_loaders= DataLoader(dev_dataset, batch_size=256, shuffle=False, num_workers=10,collate_fn=dev_dataset.collate) 
    test_loaders= DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=10,collate_fn=test_dataset.collate) 
    # load model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optim=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                        weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    global_training_steps=0
    best_acc=0.0
    scheduler=get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs*len(train_loader)
    )
    best_acc=0.0
    for epoch in range(args.epochs):
        model.train()
        for class_id,sentence_ids in tqdm(train_loader,desc="Training Epoch{}".format(epoch)):
            global_training_steps+=1
            class_id, sentence_ids = class_id.to(device), sentence_ids.to(device)  

            logits = model(sentence_ids)  

            loss = loss_func(logits, class_id) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()
            tb.add_scalar("loss",loss.item(),global_training_steps)
        acc=dev(dev_loaders,model)
        print("acc:",acc)
        if acc>best_acc:
            best_acc=acc
            st=model.state_dict(
            )
            torch.save(st,os.path.join(args.checkpoint_save_folder,"best_epoch{}_lr{}_embedding{}.bin".format(args.epochs,args.lr,args.embedding_name)))
            #_, pred = torch.max(logits, dim=1)  
            #num_correct += (pred == label).sum().item()  


        #mean_dev_acc=[dev(d_l,model) for d_l in dev_loaders]
        #print("dev_acc ",mean_dev_acc)
        #mean_dev_acc=sum(mean_dev_acc)/len(mean_dev_acc)
        #print("mean_dev_acc ",mean_dev_acc)
        #tb.add_scalar("mean_dev_acc",mean_dev_acc,epoch)
        #if mean_dev_acc>best_acc:
        
        #    best_acc=mean_dev_acc
        #    print("save model at epoch {}".format(epoch))
        #    torch.save(model.state_dict(), '{}/{}_epoch{}_bz{}_lr{}_optim{}_aug{}_best.pth'.format(
        #        args.checkpoint_save_folder,args.model,args.epochs,args.batch_size,args.lr,args.optim,args.pre_process
        #    ))
    if args.epochs ==0:
        model.load_state_dict(
            torch.load(
                args.trained_model_name_or_path
            )
        )
        test_acc=dev(test_loaders,model)
        print(test_acc)

    