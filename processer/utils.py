import json
import torch
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader
from processer.classes import InputFeatures
from hyperparameter import args
from transformers import BertTokenizer


def read_json(json_dir=args.train_file_path, do_lower_case=True):

    with open(json_dir, encoding='utf-8') as f:

        text_lists = []
        order_tag_lists = []
        entity_tag_lists = []
        for line in f:
            line = json.loads(line)
            
            token_list = []
            for token in line['text']:
                if token == '”' or token == '“' or token == '‘' or token == '’':
                    token = '"'
                if do_lower_case:
                    token_list.append(token.lower())
                else:
                    token_list.append(token)
                    print(token)

            text_lists.append(token_list)

            order_tags = ['O' for _ in range(len(line['text']))]
            entity_tags = ['O' for _ in range(len(line['text']))]
            labels = line['label']
            for tag, value in labels.items():

                for entity, indexs in value.items():
                    for index in indexs:
                        order_tags[index[0]] = 'B'
                        entity_tags[index[0]] = tag
                        for i in range(index[0]+1, index[1]+1):
                            order_tags[i] = 'I'
                            entity_tags[i] = tag

            order_tag_lists.append(order_tags)
            entity_tag_lists.append(entity_tags)

    return text_lists, order_tag_lists, entity_tag_lists


def build_corpus(order_tag_lists, entity_tag_lists):

    ordertag2id = {tag: i for i, tag in enumerate(args.id2ordertag)}
    entitytag2id = {tag: i for i, tag in enumerate(args.id2entitytag)}
    ordertag_ids, entitytag_ids = [], []

    for orders, entitys in zip(order_tag_lists, entity_tag_lists):

        ordertag_id, entitytag_id = [], []
        for order, entity in zip(orders, entitys):

            ordertag_id.append(ordertag2id[order])
            entitytag_id.append(entitytag2id[entity])

        ordertag_ids.append(ordertag_id)
        entitytag_ids.append(entitytag_id)

    return ordertag_ids, entitytag_ids

def eval_update(out_order_ids, pre_order, out_entity_ids, pre_entity):
    id2entity = args.id2entitytag
    id2order = args.id2ordertag
    tag2id = {tag: i for i, tag in enumerate(args.id2tag)}

    true_result, pre_result = [], []
    for orders_true, orders_pre, entities_true, entities_pre in zip(out_order_ids, pre_order, out_entity_ids, pre_entity):
        true_temp, pre_temp = [], []
        for order_true, order_pre, entitiy_true, entitiy_pre in zip(orders_true, orders_pre, entities_true, entities_pre):
            if id2order[order_true] in ['X', 'O', '[CLS]', '[SEP]', 'I']:
                tag_true = tag2id[id2order[order_true]]
            else:
                tag = id2order[order_true] + '-' + id2entity[entitiy_true]
                tag_true = tag2id[tag]
            true_temp.append(tag_true)

            if id2order[order_pre] in ['X', 'O', '[CLS]', '[SEP]', 'I']:
                tag_pre = tag2id[id2order[order_pre]]
            elif id2entity[entitiy_pre] in ['X', 'O', '[CLS]', '[SEP]']:
                tag_pre = tag2id[id2entity[entitiy_pre]]
            else:
                tag = id2order[order_pre] + '-' + id2entity[entitiy_pre]
                tag_pre = tag2id[tag]
            pre_temp.append(tag_pre)

        true_result.append(true_temp)
        pre_result.append(pre_temp)

    return true_result, pre_result


def convert_examples_to_features(token_list, order_ids, entity_ids,  id2order, id2entity, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        >>  token_list = [['关', '于', '存', '量', '客', '户', '的', '房', '贷', '利', '率', '是', '否', '调', '整', '，', '交', '行', '正', '在', '研', '究'],
                          ['约', '维', '蒂', '奇', '有', '望', '与', '吉', '拉', '蒂', '诺', '搭', '档', '锋', '线', '。', '2', '0']]
            tag_list = [[1, 2, 2, 2, 4, 4, 4, 4, 4, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                        [4, 4, 4, 4, 8, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

           convert_examples_to_features(token_list, tag_list, args.tag2id, 128, tokenizer)
    """
    order2id = {tag: i for i, tag in enumerate(id2order)}
    entity2id = {tag: i for i, tag in enumerate(id2entity)}
    features = []
    for tokens, orders, entitys in zip(token_list, order_ids, entity_ids):

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            orders = orders[: (max_seq_length - special_tokens_count)]
            entitys = entitys[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        orders += [order2id['[SEP]']]
        entitys += [entity2id['[SEP]']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            orders += [order2id['[CLS]']]
            entitys += [entity2id['[CLS]']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            orders = [order2id['[CLS]']] + orders
            entitys = [entity2id['[CLS]']] + entitys
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(orders)
        assert input_len == len(entitys)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            orders = ([pad_token] * padding_length) + orders
            entitys = ([pad_token] * padding_length) + entitys
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += ([pad_token_segment_id] * padding_length)
            orders += ([pad_token] * padding_length)
            entitys += ([pad_token] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(orders) == max_seq_length
        assert len(entitys) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, input_len=input_len,
                                      segment_ids=segment_ids, order_ids=orders, entity_ids=entitys))
    return features


def build_dataset(tokenizer, mode=True):

    if mode:
        # on train set
        text_lists, order_tag_lists, entity_tag_lists = read_json(args.train_file_path, do_lower_case=True)
        ordertag_ids, entitytag_ids = build_corpus(order_tag_lists, entity_tag_lists)
        features = convert_examples_to_features(text_lists, ordertag_ids, entitytag_ids, args.id2ordertag,
                                                args.id2entitytag, args.train_max_seq_length, tokenizer)
    else:
        # on dev set
        text_lists, order_tag_lists, entity_tag_lists = read_json(args.dev_file_path, do_lower_case=True)
        ordertag_ids, entitytag_ids = build_corpus(order_tag_lists, entity_tag_lists)
        features = convert_examples_to_features(text_lists, ordertag_ids, entitytag_ids, args.id2ordertag,
                                                args.id2entitytag, args.eval_max_seq_length, tokenizer)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_order_ids = torch.tensor([f.order_ids for f in features], dtype=torch.long)
    all_entity_ids = torch.tensor([f.entity_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_order_ids, all_entity_ids)

    return dataset

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_order_ids, all_entity_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_orders = all_order_ids[:, :max_len]
    all_entitys = all_entity_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_orders, all_entitys, all_lens


if __name__ == '__main__':
    # text_lists, tag_lists = read_json()
    # tag_ids = build_corpus(tag_lists)
    # print(text_lists[100:110])
    # print(tag_ids[:10])

    tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
    # token_list = [['关', '于', '存', '量', '客', '户', '的', '房', '贷', '利', '率', '是', '否', '调', '整', '，', '交', '行', '正', '在', '研', '究'],
    #               ['约', '维', '蒂', '奇', '有', '望', '与', '吉', '拉', '蒂', '诺', '搭', '档', '锋', '线', '。', '2', '0']]
    # tag_list = [[1, 2, 2, 2, 4, 4, 4, 4, 4, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    #             [4, 4, 4, 4, 8, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]

    train_dataset = build_dataset(tokenizer, True)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    batch = next(iter(train_dataloader))
    print(batch)