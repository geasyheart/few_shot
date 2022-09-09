# -*- coding: utf8 -*-
#
import json
import pathlib
from typing import Dict

import torch
from transformers import BertForMaskedLM, BertTokenizer
from utrainer import UTrainer

from transform import TNEWSTransform, read_convert
from torch.functional import F
from metric import ACCMetric
LABEL_TOKEN_IDS = []


def get_label_idx(labels):
    batch_label_idx = []
    for batch_index in range(labels.shape[0]):
        label = labels[batch_index]
        for l in range(LABEL_TOKEN_IDS.shape[0]):
            if (LABEL_TOKEN_IDS[l] == label).all():
                batch_label_idx.append(l)
                continue
    return torch.tensor(batch_label_idx, dtype=torch.long).to(labels.device)


class PTuningTrainer(UTrainer):
    def train_steps(self, batch_idx, batch_data) -> Dict:
        (input_ids, token_type_ids, attention_mask), mask_position, labels = batch_data
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        mask_output = output.gather(dim=1, index=mask_position.unsqueeze(-1).expand(-1, -1, output.size(-1)))

        # 最难理解的地方到了哦
        # 比如
        # mask_output.shape: torch.Size([8, 2, 21128])
        # LABEL_TOKEN_IDS.shape: torch.Size([15, 2])
        label_logits = torch.ones(mask_output.shape[0], LABEL_TOKEN_IDS.shape[0]).to(self.device)
        for i in range(2):
            # 直接相乘
            label_logits *= mask_output[:, i, :][:, LABEL_TOKEN_IDS[:, i]]
        # 这样拿到了label所对应的概率,你可以通过下面代码进行验证
        # for batch_index in range(mask_output.shape[0]):
        #     batch_index_matrix = mask_output[batch_index]
        #     for label_index in range(LABEL_TOKEN_IDS.shape[0]):
        #         w1_index, w2_index = LABEL_TOKEN_IDS[label_index]
        #         assert (label_logits[batch_index][label_index] == batch_index_matrix[0][w1_index] * batch_index_matrix[1][w2_index])
        label_idx = get_label_idx(labels=labels)
        loss = F.cross_entropy(torch.softmax(label_logits, dim=-1), label_idx)
        print()
        return {"loss": loss}

    def evaluate_steps(self, batch_idx, batch_data):
        (input_ids, token_type_ids, attention_mask), mask_position, labels = batch_data
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        mask_output = output.gather(dim=1, index=mask_position.unsqueeze(-1).expand(-1, -1, output.size(-1)))

        # 最难理解的地方到了哦
        # 比如
        # mask_output.shape: torch.Size([8, 2, 21128])
        # LABEL_TOKEN_IDS.shape: torch.Size([15, 2])
        label_logits = torch.ones(mask_output.shape[0], LABEL_TOKEN_IDS.shape[0]).to(self.device)
        for i in range(2):
            # 直接相乘
            label_logits *= mask_output[:, i, :][:, LABEL_TOKEN_IDS[:, i]]

        label_idx = get_label_idx(labels=labels)
        return label_logits.argmax(-1), label_idx


def read_data(name):
    data = []
    with open(pathlib.Path(__file__).parent.joinpath('fewclue_tnews').joinpath(name), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    trainer = PTuningTrainer()
    trainer.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_dl = TNEWSTransform(data=read_data('train.jsonl'), tokenizer=tokenizer, device=trainer.device).to_dl(
        batch_size=8, shuffle=True)
    dev_dl = TNEWSTransform(data=read_data('dev.jsonl'), tokenizer=tokenizer, device=trainer.device).to_dl(
        batch_size=8, shuffle=False)
    test_dl = TNEWSTransform(data=read_data('public_test.jsonl'), tokenizer=tokenizer, device=trainer.device).to_dl(
        batch_size=8, shuffle=False)

    #
    for label in read_convert().values():
        label_to_char = list(label)
        LABEL_TOKEN_IDS.append(tokenizer.convert_tokens_to_ids(label_to_char))
    LABEL_TOKEN_IDS = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long).to(trainer.device)
    trainer.fit(
        train_dl=train_dl,
        dev_dl=dev_dl,
        epochs=30,
        metric_cls=ACCMetric
    )
