import argparse
import os
import shutil
import tables

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import joblib

import tqdm
from prompt_config import get_template_by_dataset
import torch.utils.data as data_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class RawTextData(data_.Dataset):

    def __init__(self, text, node_id):
        self.text = text
        self.node_id = node_id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return (self.text[idx],self.node_id[idx])

def save_model_head(model):
    lm_head_path = "./pretrain_models/head/"
    if os.path.exists(lm_head_path):
        shutil.rmtree(lm_head_path, True)
    os.makedirs(lm_head_path)
    joblib.dump(model.lm_head.to('cpu'),open(f'{lm_head_path}lm_head.pkl','wb'))


def pretrain_collate_fn(data_tuple):
    seq = [torch.tensor(sq[0]) for sq in data_tuple]
    node_id = [sq[1] for sq in data_tuple]
    seq = pad_sequence(seq, batch_first=True, padding_value=tokenizer.pad_token_id)
    node_id = torch.tensor(node_id).view(-1, 1)
    node_id = node_id.repeat(1, seq.shape[1])
    return seq, node_id

def build_pretrain_data_by_tables(model, tokenizer, x_text, save_path, template_l_id, device, args):

    template_l_id = tokenizer.encode(template_l)[0:]
    template_l_id = torch.tensor(template_l_id).view(1, -1)

    print(template_l_id)

    token_embedding_path = save_path + 'token_embeddings.h5'
    f = tables.open_file(token_embedding_path, mode='w')
    atom = tables.Float16Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 5120))
    f.close()

    sentence_embedding_path = save_path + 'sentence_embeddings.h5'
    f = tables.open_file(sentence_embedding_path, mode='w')
    atom = tables.Float16Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 5120))
    f.close()

    token_node_ids_path = save_path + 'token_node_ids.h5'
    f = tables.open_file(token_node_ids_path, mode='w')
    atom = tables.IntAtom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1))
    f.close()

    token_label_path = save_path + 'token_labels.h5'
    f = tables.open_file(token_label_path, mode='w')
    atom = tables.IntAtom()
    array_c = f.create_earray(f.root, 'data', atom, (0, 1))
    f.close()

    model.to(device)
    feature_ls=[]
    test_max = 0
    for text in list(x_text):
        feature_ls.append(text)
    print('total node: ', len(feature_ls))

    feature_ls_ids = []
    for f in tqdm.tqdm(feature_ls):
        feature_ls_ids.append(tokenizer(f,padding=True,truncation=True)['input_ids'])
    nodedata_ = RawTextData(feature_ls_ids,list(range(len(feature_ls))))
    print(nodedata_)
    node_data_loader = DataLoader(nodedata_, batch_size=args.batch_size, shuffle=False,collate_fn=pretrain_collate_fn)

    token_node_ids_ls = []
    labels_ls = []
    embeddings_ls = []
    word_num_ls = []
    cls_embeddings_ls = []

    for i in range(1):
        for(text, node_id) in tqdm.tqdm(node_data_loader):
            with torch.no_grad():
                mlm_text_id, labels = text, text[..., 1:].contiguous()

                print(mlm_text_id)
                print(labels)

                mlm_text_id = mlm_text_id[:,1:]
                labels = labels[:,1:]
                node_id = node_id[:,1:]




                prompt_l = template_l_id.repeat(mlm_text_id.shape[0],1)#.to(device)
                prompt_labels = torch.zeros_like(prompt_l)
                node_id = torch.cat((prompt_labels-1,node_id),dim=1)
                mlm_text_id = torch.cat((prompt_l,mlm_text_id),dim=1)
                labels = torch.cat((prompt_labels,labels),dim=1)

                print(mlm_text_id)
                print(labels)
                print(node_id)
                break


    return token_embedding_path, sentence_embedding_path, token_node_ids_path, token_label_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing graph to generate require embeddings')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='instagram',
                        choices=['arxiv', 'instagram', 'reddit'])
    parser.add_argument('--plm_path', type=str, default='/data/pretrain_models/llama-2-13b-hf', help='path of llama 2')
    parser.add_argument('--pretrain_save_path', type=str, default='./token_embedding/', help='path of saving pretrain data')
    parser.add_argument('--prompt_save_path', type=str, default='./prompt_embedding/', help='path of saving prompt embedding')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of llama 2')
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    device = args.device
    save_path = args.pretrain_save_path + args.dataset_name + '/'

    print(device)
    print(save_path)

    model = AutoModelForCausalLM.from_pretrained(args.plm_path, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
        device)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path, use_fast=False)
    tokenizer.pad_token = '[PAD]'

    save_model_head(model)

    x_text = np.load(f'./datasets/{args.dataset_name}/x_text.npy')

    if os.path.exists(save_path):
        shutil.rmtree(save_path, True)
    os.makedirs(save_path)

    template_l, template_r = get_template_by_dataset(args.dataset_name)
    print("template_l:", template_l)
    print()
    print("template_r", template_r)
    token_embedding_path, sentence_embedding_path, token_node_ids_path, token_label_path = build_pretrain_data_by_tables(
        model, tokenizer, x_text, save_path, template_l, args.device, args)
    # convert_tables_to_npy(save_path)

