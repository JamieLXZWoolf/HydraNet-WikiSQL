import os
import json
import transformers
import torch

pretrained_weights = {
    ("bert", "base"): "bert-base-uncased",
    ("bert", "large"): "bert-large-uncased-whole-word-masking",
    ("roberta", "base"): "roberta-base",
    ("roberta", "large"): "roberta-large",
    ("albert", "xlarge"): "albert-xlarge-v2",
    ("grappa", "large"): "Salesforce/grappa_large_jnt",
    ("tapas", "base"): "google/tapas-base"
}


def read_jsonl(jsonl):
    for line in open(jsonl, encoding="utf8"):
        sample = json.loads(line.rstrip())
        yield sample

def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "" or line[0] == "#":
             continue
        fields = line.strip().split()
        config[fields[0]] = fields[1]
    config["train_data_path"] =  os.path.abspath(config["train_data_path"])
    config["dev_data_path"] =  os.path.abspath(config["dev_data_path"])

    return config

def create_base_model(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertModel.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaModel.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertModel.from_pretrained(weights_name)
    elif config["base_class"] == "grappa":
        return transformers.AutoModel.from_pretrained(weights_name)
    elif config["base_class"] == "tapas":
        return transformers.TapasModel.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

def create_tokenizer(config):
    weights_name = pretrained_weights[(config["base_class"], config["base_name"])]
    if config["base_class"] == "bert":
        return transformers.BertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "roberta":
        return transformers.RobertaTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "albert":
        return transformers.AlbertTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "grappa":
        return transformers.RobertaTokenizer.from_pretrained(weights_name)
    elif config["base_class"] == "tapas":
        return transformers.TapasTokenizer.from_pretrained(weights_name)
    else:
        raise Exception("base_class {0} not supported".format(config["base_class"]))

def convert_tapas_token_type_ids(token_type_ids):
    res = token_type_ids[0, :, 0]
    res = (1 - res).tolist()
    flag = False
    for i in range(len(res) - 1):
        if not flag and res[i + 1] == 0:
            res[i] = 0
            flag = True
            continue
        if flag and res[i] == 1:
            res[i] = 2
    res[0] = 0
    return res

def convert_tapas_segment_ids(segment_ids):  # (N, len)
    orig = segment_ids + 0
    for i in range(orig.shape[0]):
        orig[i, 0] = 1
        flag = 0
        for j in range(orig.shape[1]):
            if flag == 0 and orig[i, j] == 0:
                orig[i, j] = 1
                flag = 1
                continue
            if flag == 1 and orig[i, j] == 2:
                orig[i, j] = 1
    orig = 1 - orig
    res = torch.zeros(orig.shape[0], orig.shape[1], 7)
    res[:, :, 0] = orig
    res[:, :, 1] = orig
    res = res.to(dtype=int)
    try:
        assert convert_tapas_token_type_ids(
            res[0].unsqueeze(0)) == segment_ids[0].tolist()
    except:
        print("conv: ", convert_tapas_token_type_ids(res[0].unsqueeze(0)))
        print("orig: ", segment_ids[0])
    return res

def edit_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for i in range(len2 + 1):
        dp[0][i] = i
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp,
                           min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def longest_common_subsequence(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len1][len2]

def literal_exact_match(q_str, c_str):
    if q_str.find(c_str) != -1:
        return True
    return False

def literal_score_match(q_tok_wp, c_tok_wp):
    q_len = len(q_tok_wp)
    c_str = " ".join(c_tok_wp)
    c_str_len = len(c_str)
    max_score = -1
    st, ed = -1, -1
    for n in range(len(c_tok_wp), 0, -1):
        for i in range(q_len):
            if i + n > q_len:
                break
            q_str = " ".join(q_tok_wp[i: i + n])
            q_str_len = len(q_str)
            lcs = longest_common_subsequence(q_str, c_str)
            assert q_str_len > 0 and c_str_len > 0

            score = (lcs * 1.0 / q_str_len + lcs * 1.0 / c_str_len) / 2.0
            if score > max_score:
                max_score = score
                st = i
                ed = i + n
                if max_score == 1.0:
                    return max_score, st, ed
    return max_score, st, ed

def filter_content_one_column(tokenizer, q_tok_cn, cells, threshold):
    q_str_cn = "".join(q_tok_cn).lower()
    q_tok_wp = []
    for tok in q_tok_cn:
        sub_toks = tokenizer.tokenize(tok.lower())
        for sub_tok in sub_toks:
            q_tok_wp.append(sub_tok)
    matching = "#NONE#"
    for cell in cells:
        content = str(cell).lower()
        if q_str_cn.find(re.compile(' ').sub('', content)) == -1:
            continue
        c_tok_wp = tokenizer.tokenize(content)
        max_score, _, _ = literal_score_match(q_tok_wp, c_tok_wp)
        if max_score > threshold:
            matching = str(cell)
            threshold = max_score
    return matching

if __name__ == "__main__":
    qtokens = ['Tell', 'me', 'what', 'the', 'notes', 'are', 'for', 'South', 'Australia']
    column = "string School/Club Team"

    tokenizer = create_tokenizer({"base_class": "roberta", "base_name": "large"})

    qsubtokens = []
    for t in qtokens:
        qsubtokens += tokenizer.tokenize(t, add_prefix_space=True)
    print(qsubtokens)
    result = tokenizer.encode_plus(column, qsubtokens, add_prefix_space=True)
    for k in result:
        print(k, result[k])
    print(tokenizer.convert_ids_to_tokens(result["input_ids"]))



