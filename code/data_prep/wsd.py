import sys
import numpy as np
import pandas as pd
import spacy
from spacy.symbols import nsubj, dobj, pobj, prep, VERB
from tqdm import tqdm
import torch
from ewiser.spacy.disambiguate import Disambiguator
from spacy import load


def wsd_annotate(sentences, sent_idx, nlp_wsd, out_fn, batch_size=16):
    with open(out_fn, 'w') as f:
        doc_generator = nlp_wsd.pipe(sentences, batch_size=batch_size)
        i = 0
        for doc in tqdm(doc_generator, total=len(sentences)):
            with torch.no_grad():
                sent_i = sent_idx[i]
                for j in range(len(doc)):
                    w = doc[j]
                    if w._.offset:
                        parsed_token = ' '.join([w.text, w.lemma_, w.pos_, str(j), str(sent_i), str(w._.offset)])
                        f.write(parsed_token + '\n')
                i += 1


def main_wsd(n_jobs=10):
    job_id = int(sys.argv[1])
    with open('/home/leiyu/scratch/wsg/wikitext-103-sentences.txt', 'r') as f:
        sentences = f.readlines()
    sent_idx_chunk = np.array_split(np.arange(len(sentences)), n_jobs)[job_id]
    sent_chunk = [sentences[i] for i in sent_idx_chunk]

    model_path = "/home/leiyu/scratch/dim_lm/models/ewiser/ewiser.semcor+wngt.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wsd = Disambiguator(model_path, lang='en', batch_size=5, save_wsd_details=False).eval().to(device)
    nlp_wsd = load("en_core_web_sm", disable=['parser', 'ner'])
    wsd.enable(nlp_wsd, "wsd")

    out_fn = '/home/leiyu/scratch/wsg/wikitext-103-wsd-annotated-tokens-{}.txt'.format(job_id)
    wsd_annotate(sent_chunk, sent_idx_chunk, nlp_wsd, out_fn, batch_size=16)


if __name__ == '__main__':
    main_wsd()
