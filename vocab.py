# ""Vocabulary wrapper"""

import os
import json
import csv
import argparse
from collections import Counter
import pandas as pd

import nltk

# nltk.download()

# annotations = {
#     "coco_precomp": ["train_caps.txt", "dev_caps.txt"],
#     "f30k_precomp": ["train_caps.txt", "dev_caps.txt"],
#     "cc152k_precomp": ["train_caps.tsv", "dev_caps.tsv"],
# }


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d["word2idx"] = vocab.word2idx
    d["idx2word"] = vocab.idx2word
    d["idx"] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d["word2idx"]
    vocab.idx2word = d["idx2word"]
    vocab.idx = d["idx"]
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, "r") as f:
        for line in f:
            captions.append(line.strip())
    return captions


def from_tsv(tsv):
    captions = []
    img_ids = []
    with open(tsv) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for line in tsvreader:
            captions.append(line[1])
            img_ids.append(line[0])
    return captions


def build_vocab(data_path, data_name, threshold):

    counter = Counter()
    # for path in caption_file[data_name]:
    if data_name == "train" or data_name=='test':
        df=pd.read_csv(data_path)
        captions=df['caption'].tolist()

    else:
        raise NotImplementedError("Not support!")

    for i, caption in enumerate(captions):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, threshold=4)
    serialize_vocab(vocab, "./%s_vocab.json" % data_name)
    print("Saved vocabulary file to ", "./%s_vocab.json" % data_name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", default="/data/RR/data")
    # parser.add_argument(
    #     "--data_name", default="cc152k_precomp", help="{coco,f30k,cc152k}_precomp"
    # )
    # opt = parser.parse_args()
    main(r'E:\CODE\VIT\pvqa\images\train.csv', 'train')  #data_path,data_name
