import re
import nltk

pattern = re.compile(r'[-_/]+')

def my_split(s):
    text = []
    iter = re.finditer(pattern, s)
    start = 0
    for i in iter:
        if start != i.start():
            text.append(s[start: i.start()])
        text.append(s[i.start(): i.end()])
        start = i.end()
    if start != len(s):
        text.append(s[start: ])
    return text

def my_tokenize(txt):
    tokens1 = nltk.word_tokenize(txt.replace('"', " "))  # replace due to nltk transfer " to other character, see https://github.com/nltk/nltk/issues/1630
    tokens2 = []
    for token1 in tokens1:
        token2 = my_split(token1)
        tokens2.extend(token2)
    return tokens2


class FoxTokenizer:
    white_char = set()
    # white char
    white_char.add(' ')
    white_char.add('\f')
    white_char.add('\n')
    white_char.add('\r')
    white_char.add('\t')
    white_char.add('\v')

    # punctuation begin
    punc = set()
    punc.add('`')
    punc.add('~')
    punc.add('!')
    punc.add('@')
    punc.add('#')
    punc.add('$')
    punc.add('%')
    punc.add('&')
    punc.add('*')
    punc.add('(')
    punc.add(')')
    punc.add('-')
    punc.add('_')
    punc.add('+')
    punc.add('=')
    punc.add('{')
    punc.add('}')
    punc.add('|')
    punc.add('[')
    punc.add(']')
    punc.add('\\')
    punc.add(':')
    punc.add(';')
    punc.add('\'')
    punc.add('"')
    punc.add('<')
    punc.add('>')
    punc.add(',')
    punc.add('.')
    punc.add('?')
    punc.add('/')

    # punctuation end

    # given a 'offset' and a string 's'
    # return a list of tokens, each element is a list (text, start, end)
    @classmethod
    def tokenize(self, offset, s, onlyText):
        sb = ''
        tokens = []
        i = 0

        while i < len(s):
            ch = s[i]

            if ch in self.white_char or ch in self.punc:
                if len(sb) != 0:

                    if onlyText == False:
                        token = []
                        token.append(sb)
                        token.append(offset - len(sb))
                        token.append(offset)
                        tokens.append(token)
                    else:
                        tokens.append(sb)
                    sb = ''

                if ch not in self.white_char:

                    if onlyText == False:
                        token = []
                        token.append(ch)
                        token.append(offset)
                        token.append(offset + 1)
                        tokens.append(token)
                    else:
                        tokens.append(ch)
            else:
                sb += ch

            offset += 1
            i += 1

        if len(sb) != 0:

            if onlyText == False:
                token = []
                token.append(sb)
                token.append(offset - len(sb))
                token.append(offset)
                tokens.append(token)
            else:
                tokens.append(sb)
            sb = ''

        return tokens


class Alphabet:
    def __init__(self, name, label=False):
        self.name = name
        self.UNKNOWN = "</unk>"
        self.PAD = "</pad>"
        self.label = label
        self.instance2index = {}
        self.instances = []
        self.next_index = 0

        if not self.label:
            self.add(self.PAD)
            self.add(self.UNKNOWN)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def size(self):
        return self.next_index

    def get_index(self, instance):
        if not self.label:
            if instance in self.instance2index:
                return self.instance2index[instance]
            else:
                return self.instance2index[self.UNKNOWN]
        else:
            if instance in self.instance2index:
                return self.instance2index[instance]
            else:
                raise RuntimeError("{} not exist".format(instance))

    def get_instance(self, index):
        if index >= 0 and index < self.next_index:
            return self.instances[index]
        else:
            raise RuntimeError("{} not exist".format(index))

    def iteritems(self):
        return self.instance2index.items()


from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

import numpy as np
import torch
def pad_sequence(x, max_len):

    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    padded_x = torch.LongTensor(padded_x)

    return padded_x

import os
import shutil
def makedir_and_clear(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)

import pickle
def load(obj, data_file):
    f = open(data_file, 'rb')
    tmp_dict = pickle.load(f)
    f.close()
    obj.__dict__.update(tmp_dict)

def save(obj, save_file):
    f = open(save_file, 'wb')
    pickle.dump(obj.__dict__, f, 2)
    f.close()