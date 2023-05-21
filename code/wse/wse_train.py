import sys
from ast import literal_eval
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from datasets import WSETestDataset, WSELearningDataset, WSENegativeSupportsSampler, test_loader_collate_fn
from utils import setup_logger

model_dir = '/home/scratch/wse/models/bert-base-uncased/'
data_dir = '/home/scratch/wse/data/'
log_dir = '/home/scratch/wse/results/'
batch_size_train = 32
batch_size_test = 16
lr = 5e-5
n_epochs_wse = 16
n_epochs_mlm = 18
n_novel_tokens = 32460
use_partitioned_tokens = True
n_shot_train = 20
n_shot_test = 100
bert_embedding_dim = 768
is_pretrained = 0


class WSETrainer:
    def __init__(self, args):
        self.wse_type = args['wse_type']
        self.device = args['device']
        self.tokenizer = args['tokenizer']
        self.bert_model = args['bert_model']
        # self.logging_dir = args['logging_dir']
        self.data_dir = args['data_dir']
        self.model_dir = args['model_dir']
        self.embedding_dir = args['model_dir'] + 'embeddings/wse-{}/'.format(self.wse_type)
        self.lr = args['lr']
        self.batch_size_train = args['batch_size_train']
        self.batch_size_test = args['batch_size_test']
        self.use_partitioned_tokens = args['use_partitioned_tokens']
        self.bert_embedding_dim = args['bert_embedding_dim']
        self.n_epochs_mlm = args['n_epochs_mlm']
        self.is_pretrained = args['is_pretrained']

        self.n_shot_train = args['n_shot_train']
        self.n_shot_test = args['n_shot_test']

        usage_df = pd.read_csv(
            self.data_dir + 'dataframes/usage-wse.csv',
            converters={
                'encoded sentence id': literal_eval,
                'target word idx in encoded sentence': literal_eval
            }
        )
        vocab_df = pd.read_csv(
            self.data_dir + 'dataframes/vocab-wse.csv',
            converters={
                'partitioned child token id': literal_eval,
            }
        )
        vocab_df = vocab_df.loc[
            vocab_df['partitioned child token id'].map(len) > 1
            ].reset_index(drop=True)
        self.vocab_df = vocab_df
        self.sibling_lookup = self.get_siblings_lookup()

        # todo: remove partitioned tokens with no siblings
        train_usage_df = usage_df.loc[
            (usage_df['is wse training example'] == 1) \
            & (usage_df['partitioned child token id']).isin(self.sibling_lookup)
            ].reset_index(drop=True)
        val_usage_df = usage_df.loc[
            (usage_df['is wse training example'] == 0) \
            & (usage_df['partitioned child token id']).isin(self.sibling_lookup)
            ].reset_index(drop=True)
        train_vocab_df = vocab_df.loc[
            vocab_df['is wse training example'] == 1
            ].reset_index(drop=True)
        val_vocab_df = vocab_df.loc[
            vocab_df['is wse training example'] == 0
            ].reset_index(drop=True)
        self.val_vocab_df = val_vocab_df

        encoded_sentences = torch.load(model_dir + 'encoded_sentences.pt').long()
        self.train_dataset = WSELearningDataset(train_vocab_df, train_usage_df, encoded_sentences,
                                                self.tokenizer, partition_parent_tokens=self.use_partitioned_tokens)
        self.val_dataset = WSELearningDataset(val_vocab_df, val_usage_df, encoded_sentences,
                                              self.tokenizer, partition_parent_tokens=self.use_partitioned_tokens)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size_train, shuffle=False)

        self.optimizer = AdamW(self.bert_model.parameters(), lr=self.lr)

    def get_siblings_lookup(self):
        sibling_lookup = {}
        for _, row in self.vocab_df.iterrows():
            siblings_i = row['partitioned child token id']
            if len(siblings_i) > 1:
                for child_token_id in siblings_i:
                    sibling_lookup[child_token_id] = [sibling for sibling in siblings_i if sibling != child_token_id]
        return sibling_lookup

    def train_epoch(self):
        self.bert_model.train()
        self.train_dataset.usage_subsample = True
        train_losses = []
        n_batches = int(len(self.train_dataset) / self.batch_size_train)
        for _, inputs in tqdm(enumerate(self.train_loader), total=n_batches):
            B, n_shot, seq_len = inputs['inputs_source'].shape
            hidden_states_source = self.bert_model(
                input_ids=inputs['inputs_source'].to(self.device).view(-1, seq_len),
                output_hidden_states=True
            )['hidden_states'][-1].view(B, n_shot, seq_len, -1)

            hidden_states_target = self.bert_model(
                input_ids=inputs['inputs_target'].to(self.device).view(-1, seq_len),
                output_hidden_states=True
            )['hidden_states'][-1].view(B, n_shot, seq_len, -1)
            if self.wse_type == 'prototype':
                loss = prototype_loss(hidden_states_source, hidden_states_target, inputs)
            else:
                loss = exemplar_loss(hidden_states_source, hidden_states_target, inputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.detach().cpu().mean())
        mean_train_loss = torch.stack(train_losses).mean()

        return mean_train_loss

    def evaluate(self):
        self.bert_model.eval()
        self.val_dataset.usage_subsample = True
        val_losses = []
        n_batches = int(len(self.val_dataset) / self.batch_size_train)
        for _, inputs in tqdm(enumerate(self.val_loader), total=n_batches):
            with torch.no_grad():
                B, n_shot, seq_len = inputs['inputs_source'].shape
                hidden_states_source = self.bert_model(
                    input_ids=inputs['inputs_source'].to(self.device).view(-1, seq_len),
                    output_hidden_states=True
                )['hidden_states'][-1].view(B, n_shot, seq_len, -1)

                hidden_states_target = self.bert_model(
                    input_ids=inputs['inputs_target'].to(self.device).view(-1, seq_len),
                    output_hidden_states=True
                )['hidden_states'][-1].view(B, n_shot, seq_len, -1)

                if self.wse_type == 'prototype':
                    loss = prototype_loss(hidden_states_source, hidden_states_target, inputs)
                else:
                    loss = exemplar_loss(hidden_states_source, hidden_states_target, inputs)
                val_losses.append(loss.cpu().mean())

        mean_val_loss = torch.stack(val_losses).mean()
        return mean_val_loss

    def save_bert_model(self, epoch):
        torch.save(self.bert_model.state_dict(),
                   self.model_dir + 'checkpoints/wse/pretrained_{}_mlm_{}_wse_{}_{}.pt'.format(
                       self.is_pretrained, self.n_epochs_mlm, self.wse_type, epoch + 1)
                   )

    def get_usage_embeddings(self, loader):

        self.bert_model.eval()
        loader.dataset.usage_subsample = False

        exemplars = []
        child_ids = []
        n_exemplars = []
        n_batches = int(len(loader.dataset) / loader.batch_size)

        for _, inputs in tqdm(enumerate(loader), total=n_batches):

            with torch.no_grad():
                B, n_shot, seq_len = inputs['inputs'].shape
                hidden_states = self.bert_model(
                    input_ids=inputs['inputs'].to(self.device).view(-1, seq_len),
                    output_hidden_states=True
                )['hidden_states'][-1].view(B, n_shot, seq_len, -1)

                for i in range(B):
                    exemplars_i = []
                    n_exemplars_i = 0
                    child_id_i = int(inputs['partitioned_token_id'][i])
                    if child_id_i in self.sibling_lookup:

                        for j in range(n_shot):
                            if inputs['child_seq_idx'][i][j] != -1:
                                exemplars_i.append(hidden_states[i][j][inputs['child_seq_idx'][i][j]].cpu())
                                n_exemplars_i += 1
                            else:
                                exemplars_i.append(torch.zeros(hidden_states.shape[-1]))

                        exemplars_i = torch.stack(exemplars_i)
                        child_ids.append(inputs['partitioned_token_id'][i].cpu())
                        exemplars.append(exemplars_i)
                        n_exemplars.append(n_exemplars_i)

        return torch.stack(exemplars), torch.tensor(child_ids), torch.tensor(n_exemplars)

    def build_test_loader(self, epoch_wse):

        try:
            exemplars_val = torch.load(self.embedding_dir + 'exemplars_val_{}.pt'.format(epoch_wse))
            child_ids_val = torch.load(self.embedding_dir + 'child_ids_val_{}.pt'.format(epoch_wse))
            n_exemplars_val = torch.load(self.embedding_dir + 'n_exemplars_val_{}.pt'.format(epoch_wse))
        except FileNotFoundError as e:
            exemplars_val, child_ids_val, n_exemplars_val = self.get_usage_embeddings(self.val_loader)
            torch.save(exemplars_val, self.embedding_dir + 'exemplars_val_{}.pt'.format(epoch_wse))
            torch.save(child_ids_val, self.embedding_dir + 'child_ids_val_{}.pt'.format(epoch_wse))
            torch.save(n_exemplars_val, self.embedding_dir + 'n_exemplars_val_{}.pt'.format(epoch_wse))

        test_ds = WSETestDataset(self.val_vocab_df, exemplars_val, child_ids_val,
                                 n_exemplars_val, self.sibling_lookup)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size_test,
                                 shuffle=False, collate_fn=test_loader_collate_fn)
        neg_sampler = WSENegativeSupportsSampler(self.val_vocab_df, exemplars_val, child_ids_val, n_exemplars_val)

        return test_loader, neg_sampler

    def test(self, test_loader, neg_supports_sampler, n_neg_parents=99):
        precisions, mrrs = [], []
        n_batches = int(len(test_loader.dataset) / test_loader.batch_size) + 1
        for _, batch_inputs in tqdm(enumerate(test_loader), total=n_batches):
            queries = batch_inputs['exemplars_query'].to(
                self.device)  # (B, n_query_per_token, h_dim), padded with torch.ones(h_dim)
            n_exemplars_query = batch_inputs['n_exemplars_query']  # (B)
            sampled_prototypes, sampled_exemplars = neg_supports_sampler.sample(n_neg_parents)
            # sampled_exemplars: (n_neg_class, n_exemplars_per_class, h_dim), padded with -inf * torch.ones(h_dim)

            if self.wse_type == 'prototype':
                source_prototypes = batch_inputs['prototype_source'].to(self.device).unsqueeze(-1)  # (B, h_dim, 1)
                logits_source = torch.bmm(queries, source_prototypes)  # (B, n_query_per_token, 1)
                logits_neg = torch.matmul(
                    queries, sampled_prototypes.to(self.device).T
                )  # (B, n_query_per_token, n_neg_class)
                logits = torch.cat([logits_source, logits_neg], -1)  # shape (B, n_query_per_token, n_class)

            else:
                source_exemplars = batch_inputs['exemplars_source'].to(
                    self.device)  # shape (B, n_exemplars_per_token, h_dim)

                dot_prods_source = torch.bmm(queries, torch.transpose(source_exemplars, 1,
                                                                      2))  # (B, n_query_per_token, n_exemplars_per_token)
                dot_prods_source = torch.nan_to_num(dot_prods_source, nan=-torch.inf)
                logits_source = torch.logsumexp(dot_prods_source, -1).unsqueeze(-1)  # (B, n_query_per_token, 1)

                B, n_query_per_token, h_dim = queries.shape
                dot_prods_neg = torch.matmul(
                    queries.view(-1, h_dim), sampled_exemplars.view(-1, h_dim).to(self.device).T
                ).view(B, n_query_per_token, -1,
                       n_neg_parents)  # (B, n_query_per_token, n_exemplars_per_class, n_neg_class)
                dot_prods_neg = torch.nan_to_num(dot_prods_neg, nan=-torch.inf)
                logits_neg = torch.logsumexp(dot_prods_neg, -2)  # (B, n_query_per_token, n_neg_class)
                logits = torch.cat([logits_source, logits_neg], -1)  # shape (B, n_query_per_token, n_class)

            sorted_idx = torch.argsort(logits, -1, descending=True).cpu()  # shape (B, n_query_per_token, n_class)
            # the first column is the ground-truth source
            for j in range(len(queries)):
                sorted_idx_j = sorted_idx[j, :n_exemplars_query[j]]  # (n_query_j, n_class)
                # print(logits[j, :n_exemplars_query[j]])
                precisions.append(1. * (sorted_idx_j[:, 0] == 0).sum() / n_exemplars_query[j])
                mrrs.append(torch.reciprocal(1. + (sorted_idx_j == 0).nonzero(as_tuple=True)[1]).mean())

        return {
            'precision': torch.stack(precisions).mean(),
            'mrr': torch.stack(mrrs).mean()
        }


def prototype_loss(hidden_states_source, hidden_states_target, inputs):
    prototypes_source = []
    exemplars_target = []
    child_seq_idx_source = inputs['child_seq_idx_source']
    child_seq_idx_target = inputs['child_seq_idx_target']

    for i in range(hidden_states_source.shape[0]):
        child_embs_source_i = []
        child_embs_target_i = []
        for j in range(hidden_states_source.shape[1]):
            child_embs_source_i.append(hidden_states_source[i][j][child_seq_idx_source[i][j]])
            child_embs_target_i.append(hidden_states_target[i][j][child_seq_idx_target[i][j]])
        prototypes_source.append(torch.stack(child_embs_source_i).mean(0))
        exemplars_target.append(torch.stack(child_embs_target_i))

    prototypes_source = torch.stack(prototypes_source)  # shape (B, h_dim)
    exemplars_target = torch.stack(exemplars_target)  # shape (B, n_shot, h_dim)

    B, h_dim = prototypes_source.shape
    logits = torch.matmul(
        exemplars_target.view(-1, h_dim), prototypes_source.T
    ).view(B, -1, B)  # shape (B, n_shot, B)
    log_probs = torch.log_softmax(logits, -1)
    losses = []
    for i in range(log_probs.shape[0]):
        losses.append(-log_probs[i, :, i].sum())
    return torch.stack(losses).mean()


def exemplar_loss(hidden_states_source, hidden_states_target, inputs):
    B, n_shot, _, h_dim = hidden_states_source.shape
    exemplars_source = []
    exemplars_target = []
    child_seq_idx_source = inputs['child_seq_idx_source']
    child_seq_idx_target = inputs['child_seq_idx_target']

    for i in range(hidden_states_source.shape[0]):
        exemplars_source_i = []
        exemplars_target_i = []
        for j in range(hidden_states_source.shape[1]):
            exemplars_source_i.append(hidden_states_source[i][j][child_seq_idx_source[i][j]])
            exemplars_target_i.append(hidden_states_target[i][j][child_seq_idx_target[i][j]])
        exemplars_source.append(torch.stack(exemplars_source_i))
        exemplars_target.append(torch.stack(exemplars_target_i))

    exemplars_source = torch.stack(exemplars_source).view(-1, h_dim)  # shape (B*n_shot, h_dim)
    exemplars_target = torch.stack(exemplars_target).view(-1, h_dim)  # shape (B*n_shot, h_dim)
    dot_prods = torch.matmul(exemplars_target, exemplars_source.T).view(B, n_shot, B,
                                                                        n_shot)  # shape (B, n_shot, B, n_shot)
    logits = torch.logsumexp(dot_prods, dim=-1)  # shape (B, n_shot, B)
    log_probs = torch.log_softmax(logits, -1)  # shape (B, n_shot, B)
    losses = []
    for i in range(log_probs.shape[0]):
        losses.append(-log_probs[i, :, i].sum())
    return torch.stack(losses).mean()


def main(job_id):
    if job_id == 0:
        wse_type = 'prototype'
    else:
        wse_type = 'exemplar'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    bert_config = BertConfig()
    bert_model = nn.DataParallel(BertForMaskedLM(bert_config))
    bert_model.module.resize_token_embeddings(len(tokenizer) + n_novel_tokens)
    bert_model.load_state_dict(torch.load(
        model_dir + 'checkpoints/mlm/pretrained_{}_mlm_{}.pt'.format(is_pretrained, n_epochs_mlm)))
    bert_model = nn.DataParallel(bert_model.module.bert)
    bert_model.to(device)

    args = {
        'device': device,
        'tokenizer': tokenizer,
        'bert_model': bert_model,
        'n_shot_train': n_shot_train,
        'n_shot_test': n_shot_test,
        'bert_embedding_dim': bert_embedding_dim,
        'use_partitioned_tokens': use_partitioned_tokens,
        'batch_size_train': batch_size_train,
        'batch_size_test': batch_size_test,
        'lr': lr,
        'data_dir': data_dir,
        'model_dir': model_dir,
        'is_pretrained': is_pretrained,
        'n_epochs_mlm': n_epochs_mlm,
        'wse_type': wse_type
    }
    wse_trainer = WSETrainer(args)

    test_loader, neg_supports_sampler = wse_trainer.build_test_loader(0)
    test_results = wse_trainer.test(test_loader, neg_supports_sampler)
    print('wse model type: {}\n'.format(wse_type))
    print('mean wse precision before wse training: {}'.format(test_results['precision']))
    print('mrr before wse training: {}\n\n'.format(test_results['mrr']))

    # for epoch in range(n_epochs_wse):
    #     mean_train_loss = wse_trainer.train_epoch()
    #     mean_val_loss = wse_trainer.evaluate()
    #     print('epoch {}, mean wse training loss: {}, mean wse validation loss: {}'.format(
    #         epoch + 1, mean_train_loss, mean_val_loss))
    #     wse_trainer.save_bert_model(epoch + 1)
    #
    # test_loader = wse_trainer.build_test_loader(n_epochs_wse)
    # precisions, mrrs = wse_trainer.test(test_loader)
    # print('\n\nmean wse precision after wse training: {}'.format(precisions.mean()))
    # print('mrr after wse training: {}\n\n'.format(mrrs.mean()))


if __name__ == '__main__':
    job_id = int(sys.argv[1])
    main(job_id)
