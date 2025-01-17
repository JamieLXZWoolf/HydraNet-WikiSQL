import numpy as np
import json
import os
import utils
import pandas as pd
import random
import torch.utils.data as torch_data
from wikisql_gendata import SQLExample
from collections import defaultdict
from typing import List
from utils import filter_content_one_column

stats = defaultdict(int)


class InputFeature(object):
    def __init__(self,
                 question,
                 table_id,
                 tokens,
                 word_to_char_start,
                 word_to_subword,
                 subword_to_word,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.question = question
        self.table_id = table_id
        self.tokens = tokens
        self.word_to_char_start = word_to_char_start
        self.word_to_subword = word_to_subword
        self.subword_to_word = subword_to_word
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.columns = None
        self.agg = None
        self.select = None
        self.where_num = None
        self.where = None
        self.op = None
        self.value_start = None
        self.value_end = None

    def output_SQ(self, agg = None, sel = None, conditions = None, return_str=True):
        agg_ops = ['NA', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        cond_ops = ['=', '>', '<', 'OP']

        if agg is None and sel is None and conditions is None:
            sel = np.argmax(self.select)
            agg = self.agg[sel]
            conditions = []
            for i in range(len(self.where)):
                if self.where[i] == 0:
                    continue
                conditions.append((i, self.op[i], self.value_start[i], self.value_end[i]))

        agg_text = agg_ops[agg]
        select_text = self.columns[sel]
        cond_texts = []
        for wc, op, vs, ve in conditions:
            column_text = self.columns[wc]
            op_text = cond_ops[op]
            word_start, word_end = self.subword_to_word[wc][vs], self.subword_to_word[wc][ve]
            char_start = self.word_to_char_start[word_start]
            char_end = len(self.question) if word_end + 1 >= len(self.word_to_char_start) else self.word_to_char_start[word_end + 1]
            value_span_text = self.question[char_start:char_end]
            cond_texts.append(column_text + op_text + value_span_text.rstrip())

        if return_str:
            sq = agg_text + ", " + select_text + ", " + " AND ".join(cond_texts)
        else:
            sq = (agg_text, select_text, set(cond_texts))

        return sq


class HydraFeaturizer(object):
    def __init__(self, config):
        self.config = config
        self.tokenizer = utils.create_tokenizer(config)
        self.colType2token = {
            "string": "[unused1]",
            "real": "[unused2]"}

    def get_input_feature(self, example: SQLExample, config):
        max_total_length = int(config["max_total_length"])

        input_feature = InputFeature(
            example.question,
            example.table_id,
            [],
            example.word_to_char_start,
            [],
            [],
            [],
            [],
            []
        )
        
        use_content = "use_content" in config.keys() and config["use_content"] == "True"
        filter_content = use_content and "filter_content" in config.keys() and config["filter_content"] == True
        contents = example.columns if use_content else None
        
        for column, col_type, _ in example.column_meta:
            # get query tokens
            tokens = []
            word_to_subword = []
            subword_to_word = []
            content = contents[column] if use_content else None
            for i, query_token in enumerate(example.tokens):
                if self.config["base_class"] == ["roberta", "grappa"]:
                    sub_tokens = self.tokenizer.tokenize(query_token, add_prefix_space=True)
                else:
                    sub_tokens = self.tokenizer.tokenize(query_token)
                cur_pos = len(tokens)
                if len(sub_tokens) > 0:
                    word_to_subword += [(cur_pos, cur_pos + len(sub_tokens))]
                    tokens.extend(sub_tokens)
                    subword_to_word.extend([i] * len(sub_tokens))
            if filter_content:
                content = [filter_content_one_column(self.tokenizer, example.tokens, content, 0.9)]
            if config["base_class"] == "tapas":
                data = {col_type + " " + column: content} if use_content else {col_type + " " + column: []}
                tokenize_result = self.tokenizer(
                    table=pd.DataFrame.from_dict(data),
                    queries=" ".join(tokens),
                    max_length=max_total_length,
                    truncation_strategy="longest_first",
                    padding="max_length",
                    return_token_type_ids=True,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = tokenize_result["input_ids"][0].tolist()
                segment_ids = utils.convert_tapas_token_type_ids(tokenize_result["token_type_ids"])
                input_mask = tokenize_result["attention_mask"][0].tolist()
                subword_to_word = [0] + subword_to_word
                for i in range(len(input_ids) - len(subword_to_word)):
                    subword_to_word += [0]
                word_to_subword = [(pos[0]+1, pos[1]+1) for pos in word_to_subword]
            else:
                column_input = col_type + " " + column + " " + " ".join(content) if use_content else col_type + " " + column
                if config["base_class"] == "grappa":
                    tokenize_result = self.tokenizer.encode_plus(
                        column_input,
                        tokens,
                        padding="max_length",
                        max_length=max_total_length,
                        truncation=True,
                    )
                elif config["base_class"] == "roberta":
                    tokenize_result = self.tokenizer.encode_plus(
                        column_input,
                        tokens,
                        padding="max_length",
                        max_length=max_total_length,
                        truncation=True,
                        add_prefix_space=True
                    )
                else:
                    tokenize_result = self.tokenizer.encode_plus(
                        column_input,
                        tokens,
                        max_length=max_total_length,
                        truncation_strategy="longest_first",
                        pad_to_max_length=True,
                    )

                input_ids = tokenize_result["input_ids"]
                input_mask = tokenize_result["attention_mask"]
                column_token_length = 0
                for i, token_id in enumerate(input_ids):
                    if token_id == self.tokenizer.sep_token_id:
                        column_token_length = i + 2
                        break
                segment_ids = [0] * max_total_length
                for i in range(column_token_length, max_total_length):
                    if input_mask[i] == 0:
                        break
                    segment_ids[i] = 1
                subword_to_word = [0] * column_token_length + subword_to_word
                word_to_subword = [(pos[0]+column_token_length, pos[1]+column_token_length) for pos in word_to_subword]

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            assert len(input_ids) == max_total_length
            assert len(input_mask) == max_total_length
            assert len(segment_ids) == max_total_length

            input_feature.tokens.append(tokens)
            input_feature.word_to_subword.append(word_to_subword)
            input_feature.subword_to_word.append(subword_to_word)
            input_feature.input_ids.append(input_ids)
            input_feature.input_mask.append(input_mask)
            input_feature.segment_ids.append(segment_ids)

        return input_feature

    def fill_label_feature(self, example: SQLExample, input_feature: InputFeature, config):
        max_total_length = int(config["max_total_length"])

        columns = [c[0] for c in example.column_meta]
        col_num = len(columns)
        input_feature.columns = columns

        input_feature.agg = [0] * col_num
        input_feature.agg[example.select] = example.agg
        input_feature.where_num = [len(example.conditions)] * col_num

        input_feature.select = [0] * len(columns)
        input_feature.select[example.select] = 1

        input_feature.where = [0] * len(columns)
        input_feature.op = [0] * len(columns)
        input_feature.value_start = [0] * len(columns)
        input_feature.value_end = [0] * len(columns)

        for colidx, op, _ in example.conditions:
            input_feature.where[colidx] = 1
            input_feature.op[colidx] = op
        for colidx, column_meta in enumerate(example.column_meta):
            if column_meta[-1] == None:
                continue
            se = example.value_start_end[column_meta[-1]]
            try:
                s = input_feature.word_to_subword[colidx][se[0]][0]
                input_feature.value_start[colidx] = s
                e = input_feature.word_to_subword[colidx][se[1]-1][1]-1
                input_feature.value_end[colidx] = e

                assert s < max_total_length and input_feature.input_mask[colidx][s] == 1
                assert e < max_total_length and input_feature.input_mask[colidx][e] == 1

            except:
                print("value span is out of range")
                return False

        # feature_sq = input_feature.output_SQ(return_str=False)
        # example_sq = example.output_SQ(return_str=False)
        # if feature_sq != example_sq:
        #     print(example.qid, feature_sq, example_sq)
        return True

    def load_data(self, data_paths, config, include_label=False):
        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids"]}
        if include_label:
            for k in ["agg", "select", "where_num", "where", "op", "value_start", "value_end"]:
                model_inputs[k] = []
         
        column_sample = "column_sample" in config.keys() and config["column_sample"] == "True"
        if column_sample:
            model_inputs["table_id"] = []
        pos = []
        input_features = []
        for data_path in data_paths.split("|"):
            cnt = 0
            for line in open(data_path, encoding="utf8"):
                example = SQLExample.load_from_json(line)
                if not example.valid and include_label == True:
                    continue

                input_feature = self.get_input_feature(example, config)
                if include_label:
                    success = self.fill_label_feature(example, input_feature, config)
                    if not success:
                        continue

                # sq = input_feature.output_SQ()
                input_features.append(input_feature)

                cur_start = len(model_inputs["input_ids"])
                cur_sample_num = len(input_feature.input_ids)
                pos.append((cur_start, cur_start + cur_sample_num))

                model_inputs["input_ids"].extend(input_feature.input_ids)
                model_inputs["input_mask"].extend(input_feature.input_mask)
                model_inputs["segment_ids"].extend(input_feature.segment_ids)
                if include_label:
                    model_inputs["agg"].extend(input_feature.agg)
                    model_inputs["select"].extend(input_feature.select)
                    model_inputs["where_num"].extend(input_feature.where_num)
                    model_inputs["where"].extend(input_feature.where)
                    model_inputs["op"].extend(input_feature.op)
                    model_inputs["value_start"].extend(input_feature.value_start)
                    model_inputs["value_end"].extend(input_feature.value_end)
                if column_sample:
                    table_ids = [input_feature.table_id for i in range(len(input_feature.input_ids))]
                    model_inputs["table_id"].extend(table_ids)

                cnt += 1
                if cnt % 5000 == 0:
                    print(cnt)

                if "DEBUG" in config and cnt > 100:
                    break

        for k in model_inputs:
            if k != "table_id":
                model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)
            else:
                model_inputs[k] = np.array(model_inputs[k])

        return input_features, model_inputs, pos
    
class SQLDataset(torch_data.Dataset):
    def __init__(self, data_paths, config, featurizer, include_label=False):
        self.config = config
        self.featurizer = featurizer
        self.input_features, self.model_inputs, self.pos = self.featurizer.load_data(data_paths, config, include_label)

        print("{0} loaded. Data shapes:".format(data_paths))
        for k, v in self.model_inputs.items():
            print(k, v.shape)

    def __len__(self):
        return self.model_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.model_inputs.items()}

class MetaSQLDataset(torch_data.Dataset):
    def __init__(self, model_inputs):
        self.model_inputs = model_inputs

    def __len__(self):
        return self.model_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.model_inputs.items()}
        
def sample_column_wise_meta_data(original_dataset: SQLDataset, config):
    k_shot = int(config["k_shot"])
    n_way = int(config["n_way"])
    n_tasks = int(config["n_tasks"])
    datas = {}
    for i in range(len(original_dataset)):
        item = original_dataset[i]
        if item["table_id"] not in datas:
            datas[item["table_id"]] = [item]
        else:
            datas[item["table_id"]].append(item)
    
    spt_sets = []
    qry_sets = []
    
    def convert_one_dataset(dataset_array):
        model_inputs = {k: [] for k in ["input_ids", "input_mask", "segment_ids", "agg", "select", "where_num", "where", "op", "value_start", "value_end"]}
        for item in dataset_array:
            for k in item:
                if k != "table_id":
                    model_inputs[k].append(item[k])
        for k in model_inputs:
            model_inputs[k] = np.array(model_inputs[k], dtype=np.int64)
        return MetaSQLDataset(model_inputs)
    for i in range(n_tasks):
        spt = []
        tables_spt = random.sample(datas.keys(), n_way)
        for t in tables_spt:
            spt.extend(random.sample(datas[t], min(k_shot, len(datas[t]))))
        qry = []
        tables_qry = []
        while len(tables_qry) < n_way:
            t = random.sample(datas.keys(), 1)
            while t[0] in tables_spt:
                t = random.sample(datas.keys(), 1)
            tables_qry += t
        for t in tables_qry:
            qry.extend(random.sample(datas[t], min(k_shot, len(datas[t]))))
        spt_sets.append(convert_one_dataset(spt))
        qry_sets.append(convert_one_dataset(qry))
    return spt_sets, qry_sets

if __name__ == "__main__":
    vocab = "vocab/baseTrue.txt"
    config = {}
    for line in open("conf/wikisql.conf", encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split()
        config[fields[0]] = fields[1]
    # config["DEBUG"] = 1

    featurizer = HydraFeaturizer(config)
    # train_data = SQLDataset(config["train_data_path"], config, featurizer, True)
    # train_data_loader = torch_data.DataLoader(train_data, batch_size=128, shuffle=True, pin_memory=True)
    # for batch_id, batch in enumerate(train_data_loader):
    #     print(batch_id, {k: v.shape for k, v in batch.items()})

    # for k, v in stats.items():
    #     print(k, v)
    meta_datas = featurizer.load_meta_data(config["train_data_path"], config, True)
