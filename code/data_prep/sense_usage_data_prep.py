import pickle
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from supersense_usage_data_prep import sentence_encode, mlm_train_val_split

model_dir = '/home/jingyihe/scratch/wsg/models/bert-base-nli-mean-tokens/'
data_dir = '/home/jingyihe/scratch/wsg/data/'
wsd_dir = '/home/jingyihe/scratch/wsg/data/wikitext-103/wsd/'
n_usage_per_sense = 100


def build_individual_novel_vocab(sampled_usage_df):
    novel_token_vocab_df = sampled_usage_df.groupby(
        ['target verb lemma', 'target verb wordnet sense']
    ).size().reset_index(name='count')
    novel_token_vocab_df['novel token class id'] = range(novel_token_vocab_df.shape[0])
    novel_token_vocab_df = novel_token_vocab_df.drop(['count'], axis=1)

    vs2novel_token_class_id = {
        (row['target verb lemma'], row['target verb wordnet sense']): row['novel token class id']
        for _, row in novel_token_vocab_df.iterrows()}
    novel_token_class_idx_col = [
        vs2novel_token_class_id[(row['target verb lemma'], row['target verb wordnet sense'])]
        for _, row in sampled_usage_df.iterrows()
    ]
    return novel_token_vocab_df, novel_token_class_idx_col


def build_loo_novel_vocab(sampled_usage_df):
    grouped_df_by_v = sampled_usage_df.groupby(
        ['target verb lemma']
    ).agg({'target verb wordnet sense': set}).reset_index()

    novel_token_vocab_df = sampled_usage_df.groupby(
        ['target verb lemma', 'target verb wordnet sense']
    ).size().reset_index(name='count')

    vs2novel_token_class_id = {}
    i = 0
    for _, row in tqdm(grouped_df_by_v.iterrows(), total=grouped_df_by_v.shape[0]):
        sense_list_row = list(row['target verb wordnet sense'])
        random.shuffle(sense_list_row)
        test_sense = sense_list_row[0]
        support_senses_list = sense_list_row[1:]
        vs2novel_token_class_id[(row['target verb lemma'], test_sense)] = i
        i += 1
        for sense in support_senses_list:
            vs2novel_token_class_id[(row['target verb lemma'], sense)] = i
        i += 1

    loo_vocab_col = [
        vs2novel_token_class_id[(row['target verb lemma'], row['target verb wordnet sense'])]
        for _, row in sampled_usage_df.iterrows()
    ]
    novel_token_vocab_df['novel token class id'] = [
        vs2novel_token_class_id[(row['target verb lemma'], row['target verb wordnet sense'])]
        for _, row in novel_token_vocab_df.iterrows()
    ]
    novel_token_vocab_df = novel_token_vocab_df.drop(['count'], axis=1)
    novel_token_vocab_df.append(novel_token_vocab_df)

    return novel_token_vocab_df, loo_vocab_col


def main(tokenizer, topn_verb=2000, topn_noun=5000, max_len=64, min_n_sense=2):
    usage_df = pd.read_csv(data_dir + 'dataframes/usages_all.csv')

    # step 1: choose usages with most common verbs, rels and noun objects
    verb_counts_df = usage_df.groupby(['target verb lemma']).size().reset_index(name='verb count')
    verb_counts_df = verb_counts_df.sort_values(['verb count'], ascending=False).reset_index(drop=True)
    verb_vocab = set(verb_counts_df.head(topn_verb)['target verb lemma'])

    noun_counts_df = usage_df.groupby(['object lemma']).size().reset_index(name='noun object count')
    noun_counts_df = noun_counts_df.sort_values(['noun object count'], ascending=False).reset_index(drop=True)
    noun_vocab = set(noun_counts_df.head(topn_noun)['object lemma'])

    usage_df = usage_df.loc[
        (usage_df['target verb lemma'].isin(verb_vocab)) &\
        (usage_df['object lemma'].isin(noun_vocab))
        ].reset_index(drop=True)

    # step 2: choose (verb, sense) pairs with at least 10 usages
    verb_sense_count_df = usage_df.groupby(
        ['target verb lemma', 'target verb wordnet sense']
    ).agg({'sentence id': lambda x: len(list(x))}).reset_index()
    valid_row_idx = [
        i for i, row in verb_sense_count_df.iterrows()
        if row['target verb wordnet sense'][:2] != 'NA'
           and row['target verb lemma'].isalpha()
           and row['sentence id'] >= 10
    ]
    verb_sense_count_df = verb_sense_count_df.iloc[valid_row_idx].reset_index(drop=True)
    valid_verb_sense_pairs = set([
        (row['target verb lemma'], row['target verb wordnet sense'])
        for _, row in verb_sense_count_df.iterrows()
    ])
    valid_usage_row_idx = [
        i for i, row in tqdm(usage_df.iterrows(), total=usage_df.shape[0])
        if (row['target verb lemma'], row['target verb wordnet sense']) in valid_verb_sense_pairs
    ]
    usage_df = usage_df.iloc[valid_usage_row_idx].reset_index(drop=True)
    usage_df['row idx'] = range(usage_df.shape[0])

    # step 3: sample n_usage_per_sense usages per sense
    grouped_usage_df = usage_df.groupby([
        'target verb lemma', 'target verb wordnet sense'
    ]).agg({'row idx': list}).reset_index()
    sampled_usage_row_idx = []
    for _, row in tqdm(grouped_usage_df.iterrows(), total=grouped_usage_df.shape[0]):
        sampled_usage_row_idx += random.choices(row['row idx'], k=n_usage_per_sense)

    sampled_usage_row_idx = list(set(sampled_usage_row_idx))
    sampled_usage_df = usage_df.iloc[sampled_usage_row_idx].reset_index(drop=True)

    # step 4: encode sampled sentences and remove rows where target verb is not detected by the tokenizer
    encoded_sentences = []
    verb_seq_idx = []
    valid_row_idx = []
    for i, row in tqdm(sampled_usage_df.iterrows(), total=len(sampled_usage_df['relation'])):
        sent = row['sentence']
        rel = row['relation']
        if rel.startswith('pobj_'):
            prep = rel.split('_')[-1]
            sent = re.sub(' ' + prep + ' ', ' ', sent)
        verb_text = row['target verb text']
        encoded_sentence_i, verb_seq_idx_i = sentence_encode(sent, verb_text, tokenizer, max_len)
        if verb_seq_idx_i != -1:
            encoded_sentences.append(encoded_sentence_i)
            verb_seq_idx.append(int(verb_seq_idx_i))
            valid_row_idx.append(i)

    sampled_usage_df = sampled_usage_df.iloc[valid_row_idx].reset_index(drop=True)
    sampled_usage_df['target verb idx in encoded sentence'] = verb_seq_idx
    sampled_usage_df['encoded sentence idx'] = range(len(encoded_sentences))
    encoded_sentences = torch.stack(encoded_sentences)

    # step 5: choose frames with at least two object supersenses with sufficient number of object nouns
    sampled_usage_df['row index'] = range(sampled_usage_df.shape[0])
    valid_row_idx = []
    supersense_count_df = sampled_usage_df.groupby(
        ['target verb lemma']
    ).agg({'target verb wordnet sense': lambda x: len(set(x)), 'row index': list})
    for _, row in supersense_count_df.iterrows():
        if row['target verb wordnet sense'] >= min_n_sense:
            valid_row_idx += row['row index']
    sampled_usage_df = sampled_usage_df.iloc[valid_row_idx].reset_index(drop=True)
    sampled_usage_df = sampled_usage_df.drop(['row index'], axis=1)

    # step 6: assign class idx for novel tokens
    novel_token_vocab_df_ind, novel_token_class_idx_col_ind = build_individual_novel_vocab(sampled_usage_df)
    novel_token_vocab_df_loo, novel_token_class_idx_col_loo = build_loo_novel_vocab(sampled_usage_df)
    sampled_usage_df['novel token class id (individual)'] = novel_token_class_idx_col_ind
    sampled_usage_df['novel token class id (leave-one-out)'] = novel_token_class_idx_col_loo

    # step 7: train-val split
    sampled_usage_df['row index'] = range(sampled_usage_df.shape[0])
    is_train_inds_individual = mlm_train_val_split(sampled_usage_df, eval_type='individual')
    is_train_inds_loo = mlm_train_val_split(sampled_usage_df, eval_type='leave-one-out')
    sampled_usage_df['is mlm training example (individual)'] = is_train_inds_individual
    sampled_usage_df['is mlm training example (leave-one-out)'] = is_train_inds_loo

    return sampled_usage_df, encoded_sentences, \
           novel_token_vocab_df_ind, novel_token_vocab_df_loo


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    sampled_usage_df, encoded_sentences, novel_token_vocab_df_ind, novel_token_vocab_df_loo = main(tokenizer)

    sampled_usage_df.to_csv(data_dir + 'dataframes/verb-sense/usages_sampled.csv',
                            index=False)  # df obtained after filtering step 8

    novel_token_vocab_df_ind.to_csv(data_dir + 'dataframes/verb-sense/novel_token_vocab_individual.csv',
                                    index=False)
    novel_token_vocab_df_loo.to_csv(data_dir + 'dataframes/verb-sense/novel_token_vocab_leave-one-out.csv',
                                    index=False)
    torch.save(encoded_sentences,
               model_dir + 'encoded_sentences_verb-sense.pt')  # encoded sentence for each row in sampled_usage_df

