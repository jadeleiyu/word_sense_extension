import random
import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    def __init__(self, df, encoded_sentences, tokenizer, n_novel_token_class,
                 mask_prob=0.15):
        self.df = df
        self.encoded_sentences = encoded_sentences
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer
        self.n_novel_token_class = n_novel_token_class

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        inputs = torch.clone(self.encoded_sentences[self.df['encoded sentence id'][i]])
        target_words_seq_idx = self.df['target word idx in encoded sentence'][i]
        child_tokens_vocab_idx = torch.tensor(self.df['partitioned child token id'][i]) + len(
            self.tokenizer)  # vocab_id = class_id + old_vocab_size
        inputs[target_words_seq_idx] = child_tokens_vocab_idx

        labels = torch.tensor([-100] * len(inputs))
        non_pad_token_idx = (inputs != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        n_masked_tokens = int(self.mask_prob * len(non_pad_token_idx))
        randomly_masked_token_idx = non_pad_token_idx[
            random.sample(range(len(non_pad_token_idx)), k=n_masked_tokens)
        ]
        labels[randomly_masked_token_idx] = inputs[randomly_masked_token_idx]
        inputs[randomly_masked_token_idx] = self.tokenizer.mask_token_id

        return {
            'inputs': inputs,
            'labels': labels
        }


class WSELearningDataset(Dataset):
    def __init__(self, vocab_df_by_parent, usage_df_by_child, encoded_sentences, tokenizer,
                 n_shot_train=10, n_shot_test=100, partition_parent_tokens=True, usage_subsample=False):
        self.vocab_df_by_parent = vocab_df_by_parent
        self.usage_df_by_child = usage_df_by_child
        self.encoded_sentences = encoded_sentences
        self.tokenizer = tokenizer
        self.n_shot_train = n_shot_train
        self.n_shot_test = n_shot_test
        self.partition_parent_tokens = partition_parent_tokens
        self.usage_subsample = usage_subsample

    def __len__(self):
        if self.usage_subsample:  # train and val phase of wse
            return self.vocab_df_by_parent.shape[0]
        else:  # computing usage embeddings for categorization models
            return self.usage_df_by_child.shape[0]

    def __getitem__(self, i):

        if not self.usage_subsample:
            usage_row = self.usage_df_by_child.iloc[i]
            # child_token_id = usage_row['partitioned child token id'][0]
            encoded_sent_idx = usage_row['encoded sentence id']
            encoded_sents = torch.clone(self.encoded_sentences[encoded_sent_idx])
            encoded_sents = torch.cat(
                [encoded_sents,
                 torch.zeros(self.n_shot_test - len(encoded_sents), encoded_sents.shape[-1], dtype=torch.long)]
            )
            child_seq_idx = usage_row['target word idx in encoded sentence']
            child_seq_idx += [-1] * (self.n_shot_test - len(child_seq_idx))

            return {
                'partitioned_token_id': usage_row['partitioned child token id'],
                'inputs': encoded_sents,  # shape (n_shot_test, seq_len, h_dim)
                'child_seq_idx': torch.tensor(child_seq_idx)  # shape (n_shot_test)
            }

        else:
            source_child_id, target_child_id = random.sample(
                self.vocab_df_by_parent['partitioned child token id'][i], k=2)

            source_usages_row = self.usage_df_by_child.loc[
                self.usage_df_by_child['partitioned child token id'] == source_child_id
                ].reset_index(drop=True)
            target_usages_row = self.usage_df_by_child.loc[
                self.usage_df_by_child['partitioned child token id'] == target_child_id
                ].reset_index(drop=True)

            sampled_source_support_idx = random.choices(
                range(len(source_usages_row['encoded sentence id'][0])), k=self.n_shot_train
            )
            sampled_target_support_idx = random.choices(
                range(len(target_usages_row['encoded sentence id'][0])), k=self.n_shot_train
            )

            encoded_source_sents = torch.clone(self.encoded_sentences[
                                                   [source_usages_row['encoded sentence id'][0][k] for k in
                                                    sampled_source_support_idx]
                                               ])
            encoded_target_sents = torch.clone(self.encoded_sentences[
                                                   [target_usages_row['encoded sentence id'][0][k] for k in
                                                    sampled_target_support_idx]
                                               ])

            child_seq_idx_source = [source_usages_row['target word idx in encoded sentence'][0][k]
                                    for k in sampled_source_support_idx]
            child_seq_idx_target = [target_usages_row['target word idx in encoded sentence'][0][k]
                                    for k in sampled_target_support_idx]

            if self.partition_parent_tokens:
                for j in range(len(child_seq_idx_source)):
                    encoded_source_sents[j][child_seq_idx_source[j]] = len(self.tokenizer) + source_child_id
                for j in range(len(child_seq_idx_target)):
                    encoded_target_sents[j][child_seq_idx_target[j]] = len(self.tokenizer) + target_child_id

            return {
                'inputs_source': encoded_source_sents,
                'inputs_target': encoded_target_sents,
                'child_seq_idx_source': torch.tensor(child_seq_idx_source),
                'child_seq_idx_target': torch.tensor(child_seq_idx_target)
            }


class WSETestDataset(Dataset):
    def __init__(self, vocab_df, exemplars, child_ids, n_exemplars,
                 sibling_lookup, n_parent_samples=99):
        self.vocab_df = vocab_df
        self.exemplars = exemplars
        self.n_exemplars = n_exemplars
        self.child_ids = child_ids
        self.n_parent_samples = n_parent_samples
        self.sibling_lookup = sibling_lookup
        self.token_id2col_id = {int(child_id): i for i, child_id in enumerate(child_ids.tolist())}
        self.n_parent_words = self.vocab_df.shape[0]

    def __len__(self):
        return len(self.child_ids)

    def __getitem__(self, i):

        n_exemplars_query = self.n_exemplars[i]
        exemplars_query = self.exemplars[i][:n_exemplars_query]  # (n_query_i, h_dim)
        child_id_query = self.child_ids[i]

        sibling_token_idx = self.sibling_lookup[int(child_id_query)]
        n_siblings = len(sibling_token_idx)
        sibling_col_idx = torch.tensor(
            [self.token_id2col_id[int(token_id)] for token_id in sibling_token_idx])

        exemplars_source = []
        for sibling_col_id in sibling_col_idx:
            n_exemplars_sib = self.n_exemplars[sibling_col_id]
            sampled_exemplars_sib = self.exemplars[sibling_col_id][:n_exemplars_sib]
            exemplars_source.append(sampled_exemplars_sib)
        exemplars_source = torch.cat(exemplars_source, 0)
        prototype_source = exemplars_source.mean(0)

        return {
            'child_id_query': child_id_query,
            'n_exemplars_query': n_exemplars_query,
            'exemplars_query': exemplars_query,
            'prototype_source': prototype_source,
            'exemplars_source': exemplars_source
        }


def test_loader_collate_fn(batch):
    n_exemplars_query = torch.tensor([x['n_exemplars_query'] for x in batch])
    n_exemplars_source = torch.tensor([len(x['exemplars_source']) for x in batch])
    child_id_query = torch.tensor([x['child_id_query'] for x in batch])
    prototype_source = torch.stack([x['prototype_source'] for x in batch])

    exemplars_query = []
    exemplars_source = []
    max_n_exemplars_query = n_exemplars_query.max()
    max_n_exemplars_source = n_exemplars_source.max()
    h_dim = prototype_source.shape[-1]
    for i in range(len(batch)):
        exemplars_query_i = batch[i]['exemplars_query']
        exemplars_source_i = batch[i]['exemplars_source']
        exemplars_query_i = torch.cat(
            [exemplars_query_i, torch.ones(max_n_exemplars_query - n_exemplars_query[i], h_dim)], 0
        )
        exemplars_source_i = torch.cat(
            [exemplars_source_i, -torch.inf * torch.ones(max_n_exemplars_source - n_exemplars_source[i], h_dim)], 0
        )
        exemplars_query.append(exemplars_query_i)
        exemplars_source.append(exemplars_source_i)

    return {
        'child_id_query': child_id_query,
        'n_exemplars_query': n_exemplars_query,
        'exemplars_query': torch.stack(exemplars_query),
        'prototype_source': prototype_source,
        'exemplars_source': torch.stack(exemplars_source)
    }


class WSENegativeSupportsSampler:
    def __init__(self, vocab_df, exemplars, child_ids, n_exemplars):
        self.vocab_df = vocab_df
        self.exemplars = exemplars
        self.n_exemplars = n_exemplars
        self.child_ids = child_ids
        self.token_id2col_id = {int(child_id): i for i, child_id in enumerate(child_ids.tolist())}
        self.n_parent_words = self.vocab_df.shape[0]

    def sample(self, n_parent_samples=99):

        sampled_exemplars = []
        sampled_prototypes = []
        sampled_parent_ids = random.sample(list(range(self.vocab_df.shape[0])), k=n_parent_samples)

        for j in sampled_parent_ids:
            child_token_idx = self.vocab_df['partitioned child token id'][j]
            child_col_idx = torch.tensor(
                [self.token_id2col_id[int(token_id)] for token_id in child_token_idx])
            sampled_exemplars_j = []
            for child_col_id in child_col_idx:
                n_exemplars_jk = self.n_exemplars[child_col_id]
                sampled_exemplars_jk = self.exemplars[child_col_id][:n_exemplars_jk]
                sampled_exemplars_j.append(sampled_exemplars_jk)
            sampled_exemplars_j = torch.cat(sampled_exemplars_j, 0)
            sampled_exemplars.append(sampled_exemplars_j)
            sampled_prototypes.append(sampled_exemplars_j.mean(0))

        sampled_prototypes = torch.stack(sampled_prototypes)
        h_dim = sampled_prototypes.shape[-1]
        n_sampled_exemplars = torch.tensor([len(x) for x in sampled_exemplars])
        max_n_sampled_exemplars = n_sampled_exemplars.max()
        for i in range(len(sampled_exemplars)):
            sampled_exemplars[i] = torch.cat([
                sampled_exemplars[i], -torch.inf * torch.ones(max_n_sampled_exemplars - n_sampled_exemplars[i], h_dim)
            ])
        sampled_exemplars = torch.stack(sampled_exemplars)  # (n_neg_class, max_n_exemplars_per_class, h_dim)

        return sampled_prototypes, sampled_exemplars
