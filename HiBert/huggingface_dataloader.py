import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import logging
import numpy as np
import pickle

def data_handler():
    '''
    function for loading TCGA numpy arrays
    '''
    tasks = ['site', 'histology']

    y_trains = []
    y_vals = []
    y_tests = []
    num_classes = [25,117]

    y_trains = np.load(r'../data/npy/train_Y.npy')
    y_trains = [y_trains[:, i] for i in range(y_trains.shape[1])]

    y_vals = np.load(r'../data/npy/val_Y.npy')
    y_vals = [y_vals[:, i] for i in range(y_vals.shape[1])]

    y_tests = np.load(r'../data/npy/test_Y.npy')
    y_tests = [y_tests[:, i] for i in range(y_tests.shape[1])]

    # load data
    X_train = np.load('../data/npy/train_X.npy')
    X_val = np.load('../data/npy/val_X.npy')
    X_test = np.load('../data/npy/test_X.npy')

    # load word2id map
    with open('../data/word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    idx2word = {v: k for k, v in word2idx.items()}

    # convert tokens to strings
    print("\nconverting ids to strings")

    def convert_tokens(x):
        docs = []
        for i, doc in enumerate(x):
            words = []
            for id in doc:
                if not id==0:
                    words.append(idx2word[id])
            docs.append(' '.join(words))
            sys.stdout.write('processed %i documents        \r' % i)
            sys.stdout.flush()
        return docs

    X_train = convert_tokens(X_train)
    X_val = convert_tokens(X_val)
    X_test = convert_tokens(X_test)

    return X_train, X_val, X_test, y_trains, y_vals, y_tests, num_classes


class BertDatasetMemory(Dataset):
    '''
    Pytorch dataloader for generating dataset on the fly and keeping it in memory
    Designed for single-GPU implementation of HiBERT
    '''
    def __init__(self, text, labels, tokenizer_path='data/vocab.txt'):

        self.tasks = ['site', 'histology']
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tokens, self.segids, self.masks = self._process_text(text)
        self.labels = labels

    def _process_text(self, text, max_len=512, n_seg=10):

        tokens = []
        seg_ids = []
        masks = []

        for i, row in enumerate(text):

            tokenized_text = self.tokenizer.tokenize(row)
            text_segments = [tokenized_text[l:l + max_len - 2] for l in \
                             range(0, len(tokenized_text), int(max_len))]

            # clip texts too long to fit into gpu memory
            text_segments = text_segments[:n_seg]

            tokens_ = []
            seg_ids_ = []
            masks_ = []

            for j, text_segment in enumerate(text_segments):

                if len(text_segment) < 5:
                    continue

                text_segment = text_segment[:(max_len - 2)]
                text_segment = ['[CLS]'] + text_segment + ['[SEP]']
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(text_segment)
                l = len(indexed_tokens)
                l_pad = max_len - l
                indexed_tokens += [0] * l_pad
                tokens_.append(indexed_tokens)
                seg_ids_.append([0 for i in indexed_tokens])
                masks_.append([1] * l + [0] * l_pad)

            tokens.append(tokens_)
            seg_ids.append(seg_ids_)
            masks.append(masks_)

            print("processed %i lines        \r" % i)
            # sys.stdout.flush()

        print()
        return tokens, seg_ids, masks

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, idx):

        sample = {'tokens': torch.tensor(self.tokens[idx]),
                  'masks': torch.tensor(self.masks[idx]),
                  'seg_ids': torch.tensor(self.segids[idx])}

        for i, task in enumerate(self.tasks):
            sample['labels_%s' % task] = torch.tensor(self.labels[i, idx], dtype=torch.long)

        return sample


class BertDatasetMemoryMultiGPU(BertDatasetMemory):
    '''
    Pytorch dataloader for generating dataset on the fly and keeping it in memory
    Designed for multi-GPU implementation of HiBERT
    '''
    def __init__(self, text, labels, max_seg=10,
                 tokenizer_path='data/vocab.txt'):
        self.tasks = ['site', 'histology']
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tokens, self.segids, self.masks = self._process_text(text)
        self.labels = labels
        self.max_seg = max_seg

    def __getitem__(self, idx):
        n_segs = self.tokens[idx].shape[0]
        diff = self.max_seg - n_segs
        tokens = np.vstack((self.tokens[idx], np.zeros((diff, 512))))
        masks = np.vstack((self.masks[idx], np.zeros((diff, 512))))
        seg_ids = np.vstack((self.segids[idx], np.zeros((diff, 512))))

        sample = {'tokens': torch.tensor(tokens, dtype=torch.long),
                  'masks': torch.tensor(masks, dtype=torch.long),
                  'seg_ids': torch.tensor(seg_ids, dtype=torch.long),
                  'n_segs': torch.tensor(n_segs, dtype=torch.long)}

        for i, task in enumerate(self.tasks):
            sample['labels_%s' % task] = torch.tensor(self.labels[i, idx], dtype=torch.long)

        return sample


class BertDatasetToDisk(object):
    '''
    Pytorch dataloader for preparing dataset and saving outputs to disk
    Designed to be used with BertDatasetFromDisk and BertDatasetFromDiskMultiGPU classes
    '''
    def __init__(self, tokenizer_path='data/vocab.txt'):

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tasks = ['site', 'histology']

    def save_text(self, text, savepath, max_len=512, n_seg=10):

        for i, row in enumerate(text):

            tokenized_text = self.tokenizer.tokenize(row)
            text_segments = [tokenized_text[l:l + max_len - 2] for l in \
                             range(0, len(tokenized_text), int(max_len))]

            # clip texts too long to fit into gpu memory
            text_segments = text_segments[:n_seg]

            tokens = []
            seg_ids = []
            masks = []

            for j, text_segment in enumerate(text_segments):

                if j >= 8:
                    break

                text_segment = text_segment[:(max_len - 2)]
                text_segment = ['[CLS]'] + text_segment + ['[SEP]']
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(text_segment)
                l = len(indexed_tokens)
                l_pad = max_len - l
                indexed_tokens += [0] * l_pad
                tokens.append(indexed_tokens)
                seg_ids.append([0 for i in indexed_tokens])
                masks.append([1] * l + [0] * l_pad)

            tokens = np.array(tokens)
            seg_ids = np.array(seg_ids)
            masks = np.array(masks)
            np.save(os.path.join(savepath, '%i_tokens' % i), tokens)
            np.save(os.path.join(savepath, '%i_seg_ids' % i), seg_ids)
            np.save(os.path.join(savepath, '%i_masks' % i), masks)

            print("processed %i lines        \r" % i)
            # sys.stdout.flush()

        print()

    def save_labels(self, labels, savepath):

        np.save(os.path.join(savepath, 'labels.npy'), labels)


class BertDatasetFromDisk(Dataset):
    '''
    Pytorch dataloader for loading dataset from disk
    Designed to be used with BertDatasetToDisk class 
    Designed for single-GPU implementation of HiBERT
    '''
    def __init__(self, loadpath):
        self.loadpath = loadpath
        self.tasks = ['site', 'histology']
        self.labels = np.load(os.path.join(loadpath, 'labels.npy'))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, idx):
        tokens = np.load(os.path.join(self.loadpath, '%i_tokens.npy' % idx))
        masks = np.load(os.path.join(self.loadpath, '%i_masks.npy' % idx))
        seg_ids = np.load(os.path.join(self.loadpath, '%i_seg_ids.npy' % idx))

        sample = {'tokens': torch.tensor(tokens, dtype=torch.long),
                  'masks': torch.tensor(masks, dtype=torch.long),
                  'seg_ids': torch.tensor(seg_ids, dtype=torch.long)}

        for i, task in enumerate(self.tasks):
            sample['labels_%s' % task] = torch.tensor(self.labels[i, idx], dtype=torch.long)

        return sample


class BertDatasetFromDiskMultiGPU(Dataset):
    '''
    Pytorch dataloader for loading dataset from disk
    Designed to be used with BertDatasetToDisk class 
    Designed for multi-GPU implementation of HiBERT
    '''
    def __init__(self, loadpath, max_seg=10):
        self.loadpath = loadpath
        self.tasks = ['site', 'histology']
        self.labels = np.load(os.path.join(loadpath, 'labels.npy'))
        self.max_seg = max_seg

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, idx):
        tokens_ = np.load(os.path.join(self.loadpath, '%i_tokens.npy' % idx))
        masks_ = np.load(os.path.join(self.loadpath, '%i_masks.npy' % idx))
        seg_ids_ = np.load(os.path.join(self.loadpath, '%i_seg_ids.npy' % idx))

        n_segs = tokens_.shape[0]
        diff = self.max_seg - n_segs
        tokens = np.vstack((tokens_, np.zeros((diff, 512))))
        masks = np.vstack((masks_, np.zeros((diff, 512))))
        seg_ids = np.vstack((seg_ids_, np.zeros((diff, 512))))

        sample = {'tokens': torch.tensor(tokens, dtype=torch.long),
                  'masks': torch.tensor(masks, dtype=torch.long),
                  'seg_ids': torch.tensor(seg_ids, dtype=torch.long),
                  'n_segs': torch.tensor(n_segs, dtype=torch.long)}

        for i, task in enumerate(self.tasks):
            sample['labels_%s' % task] = torch.tensor(self.labels[i, idx], dtype=torch.long)

        return sample


if __name__ == "__main__":

    X_train, X_val, X_test, y_trains, y_vals, y_tests, num_classes = data_handler()

    if not os.path.exists('data/bert/train/'):
        os.makedirs('data/bert/train/')
    if not os.path.exists('data/bert/val/'):
        os.makedirs('data/bert/val/')
    if not os.path.exists('data/bert/test/'):
        os.makedirs('data/bert/test/')

    with open('data/bert/num_classes.pkl', 'wb') as f:
        pickle.dump(num_classes, f)

    datasaver = BertDatasetToDisk()
    datasaver.save_text(X_train, 'data/bert/train/')
    datasaver.save_labels(y_trains, 'data/bert/train/')
    datasaver.save_text(X_val, 'data/bert/val/')
    datasaver.save_labels(y_vals, 'data/bert/val/')
    datasaver.save_text(X_test, 'data/bert/test/')
    datasaver.save_labels(y_tests, 'data/bert/test/')
