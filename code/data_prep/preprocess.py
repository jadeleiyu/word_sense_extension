import spacy
from spacy.symbols import nsubj, dobj, pobj, prep, VERB
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

raw_data_dir = '/scratch/y/yangxu/leiyu/dim_lm/data/wikitext-103/raw/'
parsed_data_dir = '/scratch/y/yangxu/leiyu/dim_lm/data/wikitext-103/parsed/'
sent_data_dir = '/scratch/y/yangxu/leiyu/dim_lm/data/wikitext-103/sentencized/'


def parse_corpus(input_fn, output_fn, n_process=32, min_sent_length=30):
    with open(input_fn) as f:
        lines = f.readlines()

    docs = nlp.pipe(lines, n_process=n_process)
    with open(output_fn, 'w') as f:
        for doc in tqdm(docs, total=len(lines)):
            for sent in doc.sents:
                if len(sent.text) >= min_sent_length and sent.text.isascii():
                    parsed_sent = []
                    for token in sent:
                        parsed_token_str = '_'.join([token.text, token.lemma_, token.pos_, str(token.i), token.dep_,
                                                     str(token.head.i), str(token.is_alpha), str(token.is_stop)])
                        parsed_sent.append(parsed_token_str)
                    f.write(' '.join(parsed_sent) + '\n')


def main_sentencize():
    lines = []
    for s in ['train', 'validation', 'test']:
        with open('/scratch/y/yangxu/leiyu/dim_lm/data/wikitext-103/raw/{}.raw'.format(s), 'r') as f:
            lines += f.readlines()
    docs = nlp.pipe(lines, n_process=32)
    with open('/scratch/y/yangxu/leiyu/wsg/wikitext-103-sentences.txt', 'w') as f:
        for doc in docs:
            for sent in doc.sents:
                if len(sent.text) >= 30 and sent.text.isascii():
                    f.write(sent.text + '\n')


def main_parse():
    raw_fns = ['train.raw', 'validation.raw', 'test.raw']
    parsed_fns = ['train.parsed', 'validation.parsed', 'test.parsed']
    for i in range(len(raw_fns)):
        input_fn = raw_data_dir + raw_fns[i]
        output_fn = parsed_data_dir + parsed_fns[i]
        parse_corpus(input_fn, output_fn)


if __name__ == '__main__':
    main_sentencize()
