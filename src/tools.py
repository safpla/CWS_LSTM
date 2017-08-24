from collections import defaultdict
from collections import OrderedDict

import numpy as np
import gensim
import re
def initCemb(ndims, train_file, pre_trained, thr = 5.):
    f = open(train_file)
    train_vocab = defaultdict(float)
    for line in f.readlines():
        sent = line.split()
        for word in sent:
            for character in word:
                train_vocab[character] += 1
    f.close()
    character_vecs = {}
    if pre_trained is not None:
        pre_trained = gensim.models.Word2Vec.load(pre_trained)
        pre_trained_vocab = set([w for w in pre_trained.vocab.keys()])

    character_vecs['start'] = np.random.uniform(-0.5, 0.5, ndims)
    character_vecs['stop'] = np.random.uniform(-0.5, 0.5, ndims)
    for character in train_vocab:
        if train_vocab[character] < thr:
            continue
        character_vecs[character] = np.random.uniform(-0.5, 0.5, ndims)
    for character in pre_trained_vocab:
        character_vecs[character] = pre_trained[character]

    idx = 1

    Cemb = np.zeros(shape=(len(character_vecs)+1, ndims))
    character_idx_map = dict()

    for character in character_vecs:
        Cemb[idx] = character_vecs[character]
        character_idx_map[character] = idx
        idx += 1

    return Cemb, character_idx_map

def word2tags(word):
    tags = []
    for i in range(len(word)):
        tags.append('M')
    if len(word) == 1:
        tags[0] = 'S'
    else:
        tags[0] = 'B'
        tags[len(word)-1] = 'E'
    return tags

def prepareData(train_file, character_idx_map, config):
    # output:
    #   data: sentences in the form of character index
    #         shape:[num_sent, max_sent_len, feature_win]
    #   label: sentences in the form of tags
    #         shape:[num_sent, max_sent_len]

    seqs, tags = [], []
    feature_half_win = config.feature_half_win
    max_sent_len = config.max_sent_len
    f = open(train_file)
    seq_init, tag_init = [], []
    for i in range(feature_half_win):
        seq_init.append('start')
        tag_init.append('N')


    for line in f.readlines():
        sent = line.split()
        seq = seq_init.copy()
        tag = tag_init.copy()

        previous_word = 'pre'
        for word in sent:
            if len(re.sub('\W','',word,flags=re.U)) == 0:
                if len(re.sub('\W','',previous_word,flags=re.U)) == 0:
                    previous_word = word
                    continue
                for i in range(feature_half_win):
                    seq.append('stop')
                    tag.append('N')
                seqs.append(seq)
                tags.append(tag)
                seq = seq_init.copy()
                tag = tag_init.copy()
            else:
                seq.extend(list(word))
                tag.extend(list(word2tags(word)))
            previous_word = word

    data = np.zeros([len(seqs), max_sent_len, feature_half_win*2+1], np.int32)
    label = np.zeros([len(seqs), max_sent_len], int)
    tag2num = {'B':1, 'M':2, 'E':3, 'S':4}

    for iseqs in range(len(seqs)):
        seq = seqs[iseqs]
        count = 0
        for iseq in range(len(seq)):
            if iseq >= max_sent_len:
                break
            tag = tags[iseqs][iseq]
            if tag == 'N':
                continue
            X = seq[iseq-feature_half_win : iseq+feature_half_win+1]
            X = [character_idx_map[character] if character in character_idx_map else 0 for character in X]
            data[iseqs, count, :] = X
            label[iseqs, count] = tag2num[tag]
            count += 1
    return data, label

class BatchGenerator(object):
    """a batch data generator"""

    def __init__(self, X, y, shuffle=False):
        """
        :param X: all train data, should be ndarray or list like type
        :param y: all train labels, should be ndarray or list like type
        :param shuffle: shuffle or not
        """
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' example from this data set. """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            self._epochs_completed += 1
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]
