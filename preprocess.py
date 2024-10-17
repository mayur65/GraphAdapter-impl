import argparse
import os
import shutil
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import joblib
from prompt_config import get_template_by_dataset

def save_model_head(model):
    lm_head_path = "./pretrain_models/head/"
    if os.path.exists(lm_head_path):
        shutil.rmtree(lm_head_path, True)
    os.makedirs(lm_head_path)
    joblib.dump(model.lm_head.to('cpu'),open(f'{lm_head_path}lm_head.pkl','wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing graph to generate require embeddings')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='instagram',
                        choices=['arxiv', 'instagram', 'reddit'])
    parser.add_argument('--plm_path', type=str, default='/data/pretrain_models/llama-2-13b-hf', help='path of llama 2')
    parser.add_argument('--pretrain_save_path', type=str, default='./token_embedding/', help='path of saving pretrain data')
    parser.add_argument('--prompt_save_path', type=str, default='./prompt_embedding/', help='path of saving prompt embedding')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
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

    # save_lm_head(model)
    #
    # x_text = np.load(f'./datasets/{args.dataset_name}/x_text.npy')
    #
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path, True)
    # os.makedirs(save_path)
    #
    # template_l, template_r = get_template_by_dataset(args.dataset_name)
    # print("template_l:", template_l)
    # print()
    # print("template_r", template_r)
    # token_embedding_path, sentence_embedding_path, token_node_ids_path, token_label_path = build_pretrain_data_by_tables(
    #     model, tokenizer, x_text, save_path, template_l, args.device, args)
    # convert_tables_to_npy(save_path)

