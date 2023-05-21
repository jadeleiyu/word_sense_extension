import spacy
from spacy.symbols import nsubj, dobj, pobj, prep, VERB
from tqdm import tqdm

data_dir = '/home/leiyu/scratch/wsg-wn-ss/data/'


def extract_vsor(doc):
    vi2subj = {}
    vi2rel_obj = {}
    for token in doc:
        if token.dep == nsubj and token.head.pos == VERB:
            vi2subj[token.head.i] = token.i
        if token.dep == dobj and token.head.pos == VERB:
            vi2rel_obj[token.head.i] = ('dobj', token.i)
        if token.dep == pobj and token.head.dep == prep and token.head.head.pos == VERB:
            vi2rel_obj[token.head.head.i] = ("pobj_{}".format(token.head.lemma_), token.i)

    vsor_tuples = []
    # 4-tuples of (verb, subj, obj, vo-relation)
    vis = set(vi2subj.keys()).union(set(vi2rel_obj.keys()))
    for vi in vis:
        # each word has form "token_lemma_pos_index"
        verb = '_'.join([doc[vi].text, doc[vi].lemma_, doc[vi].tag_, str(vi)])
        vsor_tuple = [verb, 'N/A', 'N/A', 'N/A']
        if vi in vi2rel_obj:
            obji = vi2rel_obj[vi][1]
            rel = vi2rel_obj[vi][0]
            obj = '_'.join([doc[obji].text, doc[obji].lemma_, doc[obji].tag_, str(obji)])
            vsor_tuple[2] = obj
            vsor_tuple[3] = rel
        if vi in vi2subj:
            subji = vi2subj[vi]
            subj = '_'.join([doc[subji].text, doc[subji].lemma_, doc[subji].tag_, str(subji)])
            vsor_tuple[1] = subj
        vsor_tuples.append(vsor_tuple)

    return vsor_tuples


def main_vsor():
    nlp = spacy.load("en_core_web_sm")
    with open(data_dir + 'wikitext-103/wikitext-103-sentences.txt', 'r') as f:
        lines = f.readlines()
    sents = nlp.pipe(lines, n_process=32)
    with open(data_dir + 'wikitext-103/wikitext-103-vsor.txt', 'w') as f_vsor:
        for i, sent in tqdm(enumerate(sents), total=len(lines)):
            vsor_tuples = extract_vsor(sent)
            for vsor_tuple in vsor_tuples:
                f_vsor.write(','.join(vsor_tuple + [str(i)]) + '\n')


if __name__ == '__main__':
    main_vsor()
