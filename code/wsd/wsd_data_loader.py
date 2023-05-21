from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from utils import load_data, load_wn_senses
from wsd_eval_config import Config

config = Config()
bert_model_dir = '/home/scratch/wse/models/bert-base-uncased/'


class WSDDataLoader:
    def __init__(self, batch_size, shuffle, max_length, use_train_data_as_dev=False):

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.tokenizer_kwargs = {
            'max_length': self.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_offsets_mapping': True,
            'return_tensors': 'pt',
        }

        self.wn_senses = load_wn_senses(config.WN)
        self.use_train_data_as_dev = use_train_data_as_dev
        self._dataloaders = {}
        # targetword_vocab = []
        for mode in ['train', 'dev', 'test']:
            data = self._load_raw_data(mode)
            self._build_dataloader(data, mode)
            # targetword_vocab += [x[1] for x in data]
        # targetword_vocab = list(set(targetword_vocab))
        # self.targetword2id = {w: i for i, w in enumerate(targetword_vocab)}

    def _load_raw_data(self, mode):
        data_ = []
        if mode == 'train':
            data = load_data(*config.SEMCOR)

        elif mode == 'dev':
            if self.use_train_data_as_dev:
                data = load_data(*config.SEMCOR)
            else:
                data = load_data(*config.SE07)
        else:
            data = load_data(*config.ALL)

        for sent in data:
            original_text_tokens = list(map(list, zip(*sent)))[0]

            for offset, (_, stem, pos, examplekey, sensekey) in enumerate(sent):
                if sensekey != -1:
                    text_tokens = original_text_tokens[:]

                    encoded = self.tokenizer.encode_plus(' '.join(text_tokens), **self.tokenizer_kwargs)

                    start, end = get_subtoken_indecies(text_tokens, encoded['offset_mapping'][0].tolist(), offset)
                    if start >= end:
                        continue
                    if end >= (self.max_length // 2 - 1):
                        # cut the original text tokens
                        half = 20
                        assert offset > half
                        new_text_tokens = text_tokens[offset - half:offset + half + 1]
                        new_encoded = self.tokenizer.encode_plus(' '.join(new_text_tokens), **self.tokenizer_kwargs)
                        new_start, new_end = get_subtoken_indecies(new_text_tokens,
                                                                   new_encoded['offset_mapping'][0].tolist(), half)
                        if new_end >= (self.max_length // 2 - 1):
                            continue
                        data_.append((new_text_tokens, stem + '+' + config.pos_map[pos], examplekey, sensekey, half,
                                      [new_start, new_end]))
                    else:
                        data_.append(
                            (text_tokens, stem + '+' + config.pos_map[pos], examplekey, sensekey, offset, [start, end]))

        if mode == 'train':
            self.task_to_classes = {}
            for text_tokens, targetword, examplekey, sensekey, offset, span in data_:
                example = (text_tokens, targetword, examplekey, sensekey, offset, span)

                if targetword not in self.task_to_classes:
                    self.task_to_classes[targetword] = OrderedDict({sensekey: [example]})
                elif targetword in self.task_to_classes and sensekey not in self.task_to_classes[targetword]:
                    self.task_to_classes[targetword][sensekey] = [example]
                elif targetword in self.task_to_classes and sensekey in self.task_to_classes[targetword]:
                    self.task_to_classes[targetword][sensekey].append(example)
                else:
                    raise ValueError('Should not happen.')
            self.training_words = set([word for word in self.task_to_classes.keys()])
            self.task_freq = {task: sum([len(examples) for examples in sense_to_examples.values()])
                              for task, sense_to_examples in self.task_to_classes.items()}

            self.sense_freq = defaultdict(int)
            for task, sense_to_examples in self.task_to_classes.items():
                for sensekey, examples in sense_to_examples.items():
                    self.sense_freq[sensekey] += len(examples)
        return data_

    def _build_dataloader(self, data, mode):

        dataset = WSDDataset(data, self.tokenizer, self.wn_senses, self.max_length)
        collate_fn = dataset.collater

        self._dataloaders[mode] = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            worker_init_fn=np.random.seed(0),
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f'[{mode}] dataloader (iterator) built.')

    @property
    def train(self):
        return self._dataloaders['train']

    @property
    def dev(self):
        return self._dataloaders['dev']

    @property
    def test(self):
        return self._dataloaders['test']


class WSDDataset(Dataset):
    def __init__(self, data, tokenizer, wn_senses, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.wn_senses = wn_senses
        self.all_senses = [sensekey for _, sensekeys in self.wn_senses.items() for sensekey in sensekeys]
        self.sensekey_to_id = {sensekey: i for i, sensekey in enumerate(self.all_senses)}
        self.max_length = max_length

    def __getitem__(self, index):
        # batch: text_tokens, targetword, examplekey, sensekey, offset, span
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collater(self, batch):
        texts, targetwords, examplekeys, sensekeys, offsets, spans = list(map(list, zip(*batch)))

        sense_ids = []
        n_senses = []
        for word in targetwords:
            sense_ids_i = [self.sensekey_to_id[sensekey] for sensekey in self.wn_senses[word]]
            sense_ids.append(sense_ids_i)
            n_senses.append(len(sense_ids_i))
        for i in range(len(sense_ids)):
            sense_ids[i] += [-1] * (max(n_senses) - len(sense_ids[i]))
        sense_ids = torch.tensor(sense_ids).long()

        target_ids = torch.tensor([
            self.wn_senses[targetword].index(sensekey)
            for targetword, sensekey in zip(targetwords, sensekeys)
        ]).long()

        encoded = self.tokenizer.batch_encode_plus(
            [' '.join(text) for text in texts],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt',
        )

        # print('dataloader')
        # print('context_targetwords: ', targetwords)
        # print('context_wn_senses: ', [self.wn_senses[w] for w in targetwords])
        # print('number of senses: ', [len(self.wn_senses[w]) for w in targetwords])
        # print('sensekeys: ', sensekeys)
        # print('target_ids: ', target_ids)
        # print('*' * 30)

        return {
            'context_ids': encoded['input_ids'],
            'context_texts': texts,
            'context_targetwords': targetwords,
            'context_examplekeys': examplekeys,
            'context_sensekeys': sensekeys,
            'context_offsets': offsets,
            'context_spans': spans,
            'target_ids': target_ids,
            'sense_ids': sense_ids,
            'batch_size': target_ids.size(0),
            'batch_idx': torch.arange(target_ids.size(0)).long()
        }


def get_subtoken_indecies(text_tokens, offset_mapping, offset):
    if offset == 0:
        start = 0
        end = len(text_tokens[offset])
    else:
        start = len(' '.join(text_tokens[:offset])) + 1
        end = start + len(text_tokens[offset])

    start_token_pos = 0
    end_token_pos = 0
    for i, (s, e) in enumerate(offset_mapping):
        if start == s:
            start_token_pos = i
        if end == e:
            end_token_pos = i + 1
    if offset == 0:
        start_token_pos = 1
    return start_token_pos, end_token_pos


def extend_span_offset(paired_spans, offset_mappings):
    """
    Extend span offset for query.
    """
    paired_spans_ = []
    for paired_span, offset_mapping in zip(paired_spans, offset_mappings):
        loc = []  # indices for [CLS], first [SEP], and second [SEP]
        for i, (s, e) in enumerate(offset_mapping):
            if s == 0 and e == 0:
                loc.append(i)
        s_paired_span = list(paired_span[0])
        q_paired_span = [paired_span[1][0] + loc[1], paired_span[1][1] + loc[1]]
        paired_spans_.append([s_paired_span, q_paired_span])

    return torch.tensor(paired_spans_).long()
