import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from ast import literal_eval
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from datasets import LanguageModelingDataset
from utils import setup_logger

model_dir = '/home/scratch/wse/models/bert-base-uncased/'
data_dir = '/home/scratch/wse/data/'
log_dir = '/home/scratch/wse/results/'
batch_size = 128
lr = 5e-5
n_novel_tokens = 32460


def main_mlm(mlm_config, logger):
    is_pretrained = mlm_config['is_pretrained'][0]
    n_epochs = mlm_config['n_epochs'][0]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig()
    model = BertForMaskedLM(config)
    if is_pretrained == 1:
        model = BertForMaskedLM.from_pretrained(model_dir)
    model = nn.DataParallel(model)
    model.module.resize_token_embeddings(len(tokenizer) + n_novel_tokens)

    usage_df = pd.read_csv(
        data_dir + 'dataframes/usage-mlm.csv',
        converters={
            'target word text': literal_eval,
            'target word lemma': literal_eval,
            'target word wordnet sense': literal_eval,
            'target word idx in encoded sentence': literal_eval,
            'partitioned child token id': literal_eval
        }
    )
    train_df = usage_df.loc[
        usage_df['is mlm training example'] == 1
    ].reset_index(drop=True)
    val_df = usage_df.loc[
        usage_df['is mlm training example'] == 0
    ].reset_index(drop=True)

    encoded_sentences = torch.load(model_dir + 'encoded_sentences.pt').long()
    train_dataset = LanguageModelingDataset(train_df, encoded_sentences, tokenizer, n_novel_tokens)
    val_dataset = LanguageModelingDataset(val_df, encoded_sentences, tokenizer, n_novel_tokens)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    n_train_batches = int(len(train_dataset) / batch_size)
    n_val_batches = int(len(val_dataset) / batch_size)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    torch.save(model.state_dict(),
               model_dir + 'checkpoints/mlm/pretrained_{}_mlm_{}.pt'.format(
                   is_pretrained, 0))

    message = '\nis pretrained model: {},\n\n\n'.format(is_pretrained)
    logger.info(message)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for _, inputs in tqdm(enumerate(train_loader), total=n_train_batches):
            outputs = model(
                input_ids=inputs['inputs'].to(device),
                labels=inputs['labels'].to(device)
            )

            loss = outputs.loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().mean())

        mean_train_loss = torch.stack(train_losses).mean()

        # evaluate at the end of every epoch
        model.eval()
        val_losses = []
        for _, inputs in tqdm(enumerate(val_loader), total=n_val_batches):
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['inputs'].to(device),
                    labels=inputs['labels'].to(device)
                )
            loss = outputs.loss.mean()
            val_losses.append(loss.cpu().mean())

        mean_val_loss = torch.tensor(val_losses).mean()

        message = '\nepoch {}, mean training mlm loss: {}, mean validation mlm loss: {}\n\n\n'.format(
            epoch + 1, mean_train_loss, mean_val_loss)
        logger.info(message)

        torch.save(model.state_dict(),
                   model_dir + 'checkpoints/mlm/pretrained_{}_mlm_{}.pt'.format(
                       is_pretrained, epoch + 1))


if __name__ == '__main__':
    config_id = int(sys.argv[1])
    logging_file = os.path.join(log_dir, 'mlm/mlm_{}.log'.format(config_id))
    logger = setup_logger(logging_file=logging_file)
    mlm_configs = pd.read_csv(data_dir + 'dataframes/mlm_configs.csv')
    mlm_config = mlm_configs.iloc[[config_id]].reset_index()
    main_mlm(mlm_config, logger)
