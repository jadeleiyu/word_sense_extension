import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from wsd_data_loader import get_subtoken_indecies


class ContextEncoder(nn.Module):
    def __init__(self, bert_model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = bert_model  # an BertForMaskedLM model

    def forward(self, input_ids, spans):
        output_tensor = self.model(
            input_ids=input_ids,
            output_hidden_states=True
        )['hidden_states'][-1]
        outputs = []
        for span, output_tensor_slice in zip(spans, output_tensor):
            outputs.append(output_tensor_slice[span[0]:span[1], :].mean(dim=0))
        return torch.stack(outputs)

    def encode(self, text_tokens, offsets):
        """Take in a single text string  Return representations with subtokens merged."""
        reps = []
        encoded = self.tokenizer.encode_plus(' '.join(text_tokens), return_tensors='pt', return_offsets_mapping=True)

        with torch.no_grad():
            output_tensor = self.model(
                input_ids=encoded['input_ids'],
                output_hidden_states=True
            )['hidden_states'][-1]

        for offset in offsets:
            start, end = get_subtoken_indecies(text_tokens, encoded['offset_mapping'], offset)
            reps.append(output_tensor[:, start:end, :].mean(dim=1))
        return reps


class CBERTLinear(nn.Module):
    def __init__(self, wn_senses, training_words, context_encoder, device, bert_dim=768):
        super().__init__()
        self.wn_senses = wn_senses
        self.training_words = training_words
        self.device = device
        self.context_encoder = context_encoder

        self.all_senses = [sensekey for _, sensekeys in self.wn_senses.items() for sensekey in sensekeys]
        # self.sensekey_to_id = {sensekey: i for i, sensekey in enumerate(self.all_senses)}
        self.senseid_to_key = {i: sensekey for i, sensekey in enumerate(self.all_senses)}
        self.sensekey_linear = nn.Linear(bert_dim, len(self.all_senses))

    def forward(self, batch, model_mode='train'):
        if model_mode == 'train':

            # batch: an input dict generated from a WSDDataLoader with a WSDDataset
            context_ids = batch['context_ids'].to(self.device)
            spans = batch['context_spans']
            targetwords = batch['context_targetwords']
            target_ids = batch['target_ids']
            sense_ids = batch['sense_ids']

            # print('forward function')
            # print('targetwords: ', targetwords)
            # print('number of senses: ', [len(self.wn_senses[w]) for w in targetwords])
            # print('target_ids: ', target_ids)
            # print()

            reps = self.context_encoder(context_ids, spans)  # this step now fails
            batch_loss = None
            correct = []
            for rep, target_id, sense_ids_word in zip(reps, target_ids, sense_ids):
                # sense_ids = [self.sensekey_to_id[sensekey] for sensekey in self.wn_senses[word]]
                # print('len sense_ids: ', len(sense_ids))
                sense_ids_word = sense_ids_word[(sense_ids_word != -1).nonzero(as_tuple=True)[0]]
                linear_weights = self.sensekey_linear.weight[sense_ids_word, :]
                # print('shape of linear_weights: ', linear_weights.shape)
                linear_biases = self.sensekey_linear.bias[sense_ids_word]
                # print('shape of linear_biases: ', linear_biases.shape)
                out = linear_weights @ rep + linear_biases
                # print('shape of out: ', out.shape)
                pred = torch.argmax(out)
                # print('shape of pred: ', pred.shape)
                correct.append(pred == target_id)

                # print('out: ', out.unsqueeze(0))
                # print('target_id: ', target_id.unsqueeze(0))
                loss = F.cross_entropy(out.unsqueeze(0), target_id.unsqueeze(0))
                # print('loss: ', loss)
                # print()
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss += loss

            return batch_loss / len(targetwords), torch.stack(correct).detach()
        else:
            return self.forward_eval(batch)

    def forward_eval(self, batch):
        context_ids = batch['context_ids'].to(self.device)
        spans = batch['context_spans']
        targetwords = batch['context_targetwords']
        examplekeys = batch['context_examplekeys']
        target_ids = batch['target_ids']
        sense_ids = batch['sense_ids']
        batch_idx = batch['batch_idx']

        with torch.no_grad():
            reps = self.context_encoder(context_ids, spans)
            correct = []
            preds = []

            for batch_id, rep, target_id, sense_ids_word in zip(batch_idx, reps, target_ids, sense_ids):
                word = targetwords[int(batch_id)]
                examplekey = examplekeys[int(batch_id)]
                target_id = target_id.item()
                candidate_sensekeys = self.wn_senses[word]
                if word in self.training_words:
                    sense_ids_word = sense_ids_word[(sense_ids_word != -1).nonzero(as_tuple=True)[0]]
                    linear_weights = self.sensekey_linear.weight[sense_ids_word, :]
                    linear_biases = self.sensekey_linear.bias[sense_ids_word]
                    out = linear_weights @ rep + linear_biases
                    pred_idx = torch.argmax(out).item()

                else:
                    pred_idx = 0
                pred = f'{examplekey} {candidate_sensekeys[pred_idx]}'
                preds.append(pred)
            print('output preds by model before gathering: ', preds)
            return preds

    def forward_eval_return_rep(self, batch):
        assert len(batch['context_targetwords']) == 1

        context_ids = batch['context_ids']
        spans = batch['context_spans']
        targetword = batch['context_targetwords'][0]
        sensekey = batch['context_sensekeys'][0]

        with torch.no_grad():
            query_rep = self.context_encoder(context_ids, spans)
            return query_rep, sensekey, targetword


def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer


class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            # assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []
        for i in range(output.size(0)):
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output


class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        # load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):
        # encode gloss text
        if self.is_frozen:
            with torch.no_grad():
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        # training model to put all sense information on CLS token
        gloss_output = gloss_output[:, 0, :].squeeze(dim=1)  # now bsz*gloss_hdim
        return gloss_output


class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        # load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context

    def forward(self, input_ids, attn_mask, output_mask):
        # encode context
        if self.is_frozen:
            with torch.no_grad():
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        # average representations over target word(s)
        example_arr = []
        for i in range(context_output.size(0)):
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output


class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False):
        super(BiEncoderModel, self).__init__()

        # tying encoders for ablation
        self.tie_encoders = tie_encoders

        # load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss,
                                              tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)
