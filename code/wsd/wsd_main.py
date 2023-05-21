from tqdm import tqdm
from transformers import BertModel

from utils import *
from wsd_models import *
from wsd_data_loader import *
from wse_train import WSETrainer
from torch.optim import AdamW

bert_model_dir = '/home/scratch/wse/models/bert-base-uncased/'
data_dir = '/home/scratch/wse/data/'
batch_size = 64
shuffle = True
max_length = 128
lr = 2e-5
n_epochs = 20


class WSDTrainer:
    def __init__(self, args):
        self.device = args['device']
        self.bert_wsd_model = args['bert_wsd_model']
        self.wsd_loaders = args['wsd_loaders']
        self.optimizer = AdamW(self.bert_wsd_model.parameters(), lr=args['lr'])

    def train_epoch(self):
        self.bert_wsd_model.train()
        train_loader = self.wsd_loaders.train
        n_batches = int(len(train_loader.dataset) / train_loader.batch_size)
        train_losses = []
        for _, batch_inputs in tqdm(enumerate(train_loader), total=n_batches):
            loss, _ = self.bert_wsd_model(batch_inputs, 'train')
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.detach().cpu())
        return torch.stack(train_losses).mean()

    def evaluate(self, val_set):

        if val_set == 'dev':
            val_loader = self.wsd_loaders.dev
        else:
            val_loader = self.wsd_loaders.test
        n_dev_examples = len(val_loader.dataset)
        n_batches = int(n_dev_examples / val_loader.batch_size)
        self.bert_wsd_model.eval()
        # n_correct = 0
        preds = []
        for _, batch_inputs in tqdm(enumerate(val_loader), total=n_batches):
            # print(batch_inputs)
            with torch.no_grad():
                batch_preds = self.bert_wsd_model(batch_inputs, 'eval')
                print('batch preds: ', batch_preds)
                # print(batch_correct)
                # print(batch_preds)
                # n_correct += batch_correct
                preds += batch_preds

        pred_key_path = os.path.join(config.TMP_DIR, 'tmp.key.txt')
        save_pred_file(preds, os.path.join(config.TMP_DIR, 'tmp.key.txt'))
        f1 = evaluate_from_pred_file(config.SE07_GOLD_KEY_PATH, pred_key_path)
        # acc = 1.0 * n_correct / n_dev_examples
        return {'f1': f1, 'predictions': preds}


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    bert_encoder = BertModel.from_pretrained(bert_model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_dir)
    context_encoder = ContextEncoder(bert_encoder, tokenizer)

    wsd_loaders = WSDDataLoader(batch_size, shuffle, max_length)
    bert_wsd_linear_model = CBERTLinear(wsd_loaders.wn_senses, wsd_loaders.training_words, context_encoder, device)
    bert_wsd_linear_model = nn.DataParallel(bert_wsd_linear_model).to(device)

    args = {
        'device': device,
        'tokenizer': tokenizer,
        'bert_encoder': bert_encoder,
        'data_dir': data_dir,
        'lr': lr,
        'batch_size': batch_size,
        'use_partitioned_tokens': False,
        'bert_wsd_model': bert_wsd_linear_model,
        'wsd_loaders': wsd_loaders
    }

    wsd_trainer = WSDTrainer(args)
    # wse_trainer = WSETrainer(args)

    for epoch in range(n_epochs):
        # wse_train_loss = wse_trainer.train_epoch()
        # wse_val_loss = wse_trainer.evaluate()
        # wsd_train_loss = wsd_trainer.train_epoch()
        wsd_val_f1 = wsd_trainer.evaluate('dev')['f1']
        wsd_test_f1 = wsd_trainer.evaluate('test')['f1']

        print('epoch {}'.format(epoch))
        # print('mean WSE training loss: {}, mean WSE evaluation loss: {}'.format(wse_train_loss, wse_val_loss))
        # print('mean WSD training loss: {}, mean WSD validation f1 score: {}, mean WSD f1 score loss: {}.\n\n\n'.format(
        #     wsd_train_loss, wsd_val_f1, wsd_test_f1))
        print('mean WSD validation f1 score: {}, mean WSD f1 score loss: {}.\n\n\n'.format(
            wsd_val_f1, wsd_test_f1))


if __name__ == '__main__':
    main()
    # gold_key_path = config.SE07_GOLD_KEY_PATH
    # pred_key_path = '/lustre04/scratch/wse/tmp/tmp.key.txt'
    # print(config.SCORER_DIR)
    # print(gold_key_path)
    # os.chdir(config.SCORER_DIR)
    # evaluate_from_pred_file(gold_key_path, pred_key_path)

    # java Scorer.java /home/scratch/wse/data/wsd/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt /lustre04/scratch/wse/tmp/tmp.key.txt
