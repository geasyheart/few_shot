# -*- coding: utf8 -*-
#
import json
import pathlib
from typing import Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def read_convert():
    with open(pathlib.Path(__file__).parent.joinpath('fewclue_tnews').joinpath('tnews.json'), 'r') as f:
        return json.load(f)


def build_sample(tokenizer, data: Dict):
    src_ids = tokenizer.encode(data['sentence'], add_special_tokens=False)
    # 构造prompt
    # 注意，tnews.json的label的长度都转换为固定长度（2），所以此处就默认值了

    # 超参，此处先只用1个，我试了下
    # ['[unused1]']
    # ['[unused1]', '[unused2]']
    # ['[unused1]', '[unused2]', '[unused3]']
    # 其中第二种是最好的，达到了0.667
    p_tokens = ['[unused1]', '[unused2]', '[unused3]']
    src_ids = tokenizer.convert_tokens_to_ids(p_tokens) + \
              [tokenizer.cls_token_id] + \
              [tokenizer.mask_token_id] * len(data['label_desc']) + \
              src_ids + \
              [tokenizer.sep_token_id]
    mask_positions = [
        index + 1 + len(p_tokens)
        for index in range(len(data['label_desc']))
    ]
    # label
    label_ids = tokenizer.convert_tokens_to_ids(list(data['label_desc']))
    return torch.tensor(src_ids, dtype=torch.long), torch.tensor(mask_positions, dtype=torch.long), torch.tensor(
        label_ids, dtype=torch.long)


class TNEWSTransform(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.label_convert = read_convert()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        line = self.data[item]
        line = {k: v for k, v in line.items()}
        line['label_desc'] = self.label_convert[line['label_desc']]
        return build_sample(tokenizer=self.tokenizer, data=line)

    def collate_fn(self, batch_data):
        input_ids, token_type_ids, attention_mask = [], [], []
        for input_id, mask_position, label in batch_data:
            input_ids.append(input_id)
            token_type_ids.append(torch.zeros(input_id.shape, dtype=torch.long))
            attention_mask.append(torch.ones(input_id.shape).bool())
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(
            self.device)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(
            self.device)

        return (input_ids, token_type_ids, attention_mask), torch.stack([i[1] for i in batch_data]).to(
            self.device), torch.stack([i[2] for i in batch_data]).to(self.device)

    def to_dl(self, batch_size, shuffle):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    def read_data(name):
        data = []
        with open(pathlib.Path(__file__).parent.joinpath('fewclue_tnews').joinpath(name), 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    for f in ('dev.jsonl', 'train.jsonl', 'public_test.jsonl'):
        train_dl = TNEWSTransform(data=read_data(f), tokenizer=tokenizer).to_dl(
            batch_size=8, shuffle=True)
        for _ in train_dl:
            print(_)
