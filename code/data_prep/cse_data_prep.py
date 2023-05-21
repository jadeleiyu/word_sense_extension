import pickle
import numpy as np
import pandas as pd

data_dir = '/home/jingyihe/scratch/wsg/data/'


def contrastive_learning_data_prep(sampled_usage_df, novel_token_vocab_df, sense_type, eval_type, train_ratio=0.8):

    if sense_type == 'object-supersense':
        grouped_novel_token_vocab_df = novel_token_vocab_df.groupby(['target verb lemma', 'relation']).agg(
            {'novel token class id': list}
        ).reset_index()
    else:
        grouped_novel_token_vocab_df = novel_token_vocab_df.groupby(['target verb lemma']).agg(
            {'novel token class id': list}
        ).reset_index()

    grouped_sampled_usage_df = sampled_usage_df.groupby(['novel token class id ({})'.format(eval_type)]).agg(
        {'encoded sentence idx': list, 'target verb idx in encoded sentence': list}
    ).reset_index()
    novel_token_id2enc_sent_idx = {row['novel token class id ({})'.format(eval_type)]: row['encoded sentence idx']
                                   for _, row in grouped_sampled_usage_df.iterrows()}
    novel_token_id2target_token_seq_idx = {
        row['novel token class id ({})'.format(eval_type)]: row['target verb idx in encoded sentence']
        for _, row in grouped_sampled_usage_df.iterrows()}

    # cse train-val split: all sibling tokens under the same v-r frame must fall into the same learning category (train vs. val)
    is_train_vr_inds = np.zeros(grouped_novel_token_vocab_df.shape[0], dtype='int')
    is_train_vr_inds[:int(train_ratio * grouped_novel_token_vocab_df.shape[0])] = 1
    np.random.shuffle(is_train_vr_inds)
    is_train_token_lookup = {}
    sibling_tokens_lookup = {}
    for i, row in grouped_novel_token_vocab_df.iterrows():
        novel_token_class_idx_i = row['novel token class id']
        for j in range(len(novel_token_class_idx_i)):
            is_train_token_lookup[novel_token_class_idx_i[j]] = is_train_vr_inds[i]
            sibling_tokens_lookup[novel_token_class_idx_i[j]] \
                = [novel_token_class_idx_i[k] for k in range(len(novel_token_class_idx_i)) if k != j]

    contrastive_learning_df = {
        'novel token class id': [],
        'sibling novel token class idx': [],
        'encoded sentence idx': [],
        'target verb idx in encoded sentence': [],
        'is training example': []
    }

    for novel_token_class_id in grouped_sampled_usage_df['novel token class id ({})'.format(eval_type)].tolist():
        contrastive_learning_df['novel token class id'].append(novel_token_class_id)
        contrastive_learning_df['sibling novel token class idx'].append(
            sibling_tokens_lookup[novel_token_class_id]
        )
        contrastive_learning_df['encoded sentence idx'].append(
            novel_token_id2enc_sent_idx[novel_token_class_id]
        )
        contrastive_learning_df['target verb idx in encoded sentence'].append(
            novel_token_id2target_token_seq_idx[novel_token_class_id]
        )
        contrastive_learning_df['is training example'].append(is_train_token_lookup[novel_token_class_id])

    contrastive_learning_df = pd.DataFrame(contrastive_learning_df)
    is_cse_training_lookup = {
        row['novel token class id']: row['is training example']
        for _, row in contrastive_learning_df.iterrows()}

    return pd.DataFrame(contrastive_learning_df), sibling_tokens_lookup, is_cse_training_lookup


def main():
    for sense_type in ['object-supersense', 'verb-sense']:
        sampled_usage_df = pd.read_csv(data_dir + 'dataframes/{}/usages_sampled.csv'.format(sense_type))
        for eval_type in ['individual', 'leave-one-out']:
            novel_token_vocab_df = pd.read_csv(
                data_dir + 'dataframes/{}/novel_token_vocab_{}.csv'.format(sense_type, eval_type))

            contrastive_learning_df, sibling_tokens_lookup, is_cse_training_lookup = contrastive_learning_data_prep(
                sampled_usage_df, novel_token_vocab_df, sense_type, eval_type)

            sampled_usage_df['is cse training example ({})'.format(eval_type)] = [
                is_cse_training_lookup[row['novel token class id ({})'.format(eval_type)]]
                for _, row in sampled_usage_df.iterrows()
            ]
            contrastive_learning_df.to_csv(
                data_dir + 'dataframes/{}/contrastive_{}.csv'.format(sense_type, eval_type), index=False)
            pickle.dump(sibling_tokens_lookup, open(
                data_dir + 'sibling_tokens_lookup_{}_{}.p'.format(sense_type, eval_type), 'wb'
            ))
        sampled_usage_df.to_csv(data_dir + 'dataframes/{}/usages_sampled.csv'.format(sense_type), index=False)


if __name__ == '__main__':
    main()
