import json
import re
import string

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import argparse
from numpy import random, vstack, save, zeros
from gensim.models import Word2Vec
# import logging
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for the deep classifiers',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', '-dr', default="data/features_full", type=str,
                        help='Provide the path for PathReports')
    parser.add_argument("--task", default="site", type=str, required=False, help="site or histology?")
    # Not required
    parser.add_argument('--use_site_info', action='store_true', default=True, help='uses site info in ground truth')
    # Not required
    parser.add_argument('--use_full_features', action='store_true', default=True, help='uses the full report')
    # TODO: pre-processing options; potentially LANL, MIMIC, API version
    # TODO: vectorization options: Prep_CNN (LANL Vector), Word2Vec, etc.

    args = parser.parse_args()
    print(args)

    return args


args = parse_arguments()

def get_split_docs(args,task):
    # x_train = np.load('./ground_truth/CNN/train_data.npy')
    # x_test = np.load('./ground_truth/CNN/test_data.npy')
    # train_df = pd.read_csv('./data/train_labels.csv', delimiter='\s+')
    # test_df = pd.read_csv('./data/test_labels.csv', delimiter='\s+')
    train_df = pd.read_csv('./data/split/train_labels.csv', delimiter=',')
    val_df = pd.read_csv('./data/split/val_labels.csv',delimiter=',')
    test_df = pd.read_csv('./data/split/test_labels.csv', delimiter=',')
    test_df.index = test_df.filename.values
    val_df.index = val_df.filename.values
    train_df.index = train_df.filename.values
    if args.task == 'site':
        y_train = train_df.site.values
        y_val = val_df.site.values
        y_test = test_df.site.values
    elif args.task == 'histology':
        y_train = train_df.histology.values
        y_val = val_df.histology.values
        y_test = test_df.histology.values
    # Assign train and test site info
    tr_site_info, te_site_info, tv_site_info = [], [], []
    if args.use_site_info:
        if task == 'site':
            # class_site_mapper = json.load(open('./data/class_site_mapper.json', 'r'))
            class_site_mapper = json.load(open('./data/mapper/site_class_mapper.json', 'r'))
            # Test site info
        #     te_site_info = []
        #     predicted_site = np.argmax(np.load('./new_val/CNN/TCGA/full_features/site/val_pred_scores.npy'), 1)
        #     if not len(predicted_site) == test_df.shape[0]:
        #         print("predicted site labels and original test dataset dimensions does not match")
        #         # pdb.set_trace()
        #         sys.exit(0)
        #     for psite in predicted_site:
        #         te_site_info.append(class_site_mapper[str(psite)])
        #     # Train site info
            tr_site_info = []
            for tsite in train_df.site.values:
                # print(tsite)
                tr_site_info.append(class_site_mapper[str(tsite).strip()])
            tv_site_info = []
            for vsite in val_df.site.values:
                tv_site_info.append(class_site_mapper[str(vsite).strip()])
            te_site_info = []
            for tesite in test_df.site.values:
                te_site_info.append(class_site_mapper[str(tesite).strip()])
        else:
            class_histology_mapper = json.load(open('./data/mapper/histology_class_mapper.json', 'r'))
            tr_site_info = []
            for tsite in train_df.histology.values:
                # print(tsite)
                tr_site_info.append(class_histology_mapper[str(tsite).strip()])
            tv_site_info = []
            for vsite in val_df.histology.values:
                tv_site_info.append(class_histology_mapper[str(vsite).strip()])
            te_site_info = []
            for tesite in test_df.histology.values:
                te_site_info.append(class_histology_mapper[str(tesite).strip()])

    tr_docs, _ = prep_splits_data(args, train_df, tr_site_info)
    te_docs, _ = prep_splits_data(args, test_df, te_site_info)
    tv_docs, _ = prep_splits_data(args, val_df, tv_site_info)

    return tr_docs, te_docs,tv_docs, tr_site_info, te_site_info,tv_site_info

# MTCNN Version
def get_split_docs2(args):
    # x_train = np.load('./ground_truth/CNN/train_data.npy')
    # x_test = np.load('./ground_truth/CNN/test_data.npy')
    # train_df = pd.read_csv('./data/train_labels.csv', delimiter='\s+')
    # test_df = pd.read_csv('./data/test_labels.csv', delimiter='\s+')
    train_df = pd.read_csv('./data/split/train_labels.csv', delimiter=',')
    val_df = pd.read_csv('./data/split/val_labels.csv',delimiter=',')
    test_df = pd.read_csv('./data/split/test_labels.csv', delimiter=',')
    test_df.index = test_df.filename.values
    val_df.index = val_df.filename.values
    train_df.index = train_df.filename.values
    if args.task == 'site':
        y_train = train_df.site.values
        y_val = val_df.site.values
        y_test = test_df.site.values
    elif args.task == 'histology':
        y_train = train_df.histology.values
        y_val = val_df.histology.values
        y_test = test_df.histology.values
    # Assign train and test site info
    tr_site_info, te_site_info, tv_site_info = [], [], []
    tr_histology_info, te_histology_info, tv_histology_info = [], [], []

    if args.use_site_info:

        # class_site_mapper = json.load(open('./data/class_site_mapper.json', 'r'))
        class_site_mapper = json.load(open('./data/mapper/site_class_mapper.json', 'r'))
        # Test site info
    #     te_site_info = []
    #     predicted_site = np.argmax(np.load('./new_val/CNN/TCGA/full_features/site/val_pred_scores.npy'), 1)
    #     if not len(predicted_site) == test_df.shape[0]:
    #         print("predicted site labels and original test dataset dimensions does not match")
    #         # pdb.set_trace()
    #         sys.exit(0)
    #     for psite in predicted_site:
    #         te_site_info.append(class_site_mapper[str(psite)])
    #     # Train site info
        tr_site_info = []
        for tsite in train_df.site.values:
            # print(tsite)
            tr_site_info.append(class_site_mapper[str(tsite).strip()])
        tv_site_info = []
        for vsite in val_df.site.values:
            tv_site_info.append(class_site_mapper[str(vsite).strip()])
        te_site_info = []
        for tesite in test_df.site.values:
            te_site_info.append(class_site_mapper[str(tesite).strip()])

        class_histology_mapper = json.load(open('./data/mapper/histology_class_mapper.json', 'r'))
        tr_histology_info = []
        for tsite in train_df.histology.values:
            # print(tsite)
            tr_histology_info.append(class_histology_mapper[str(tsite).strip()])
        tv_histology_info = []
        for vsite in val_df.histology.values:
            tv_histology_info.append(class_histology_mapper[str(vsite).strip()])
        te_histology_info = []
        for tesite in test_df.histology.values:
            te_histology_info.append(class_histology_mapper[str(tesite).strip()])

    tr_docs, _ = prep_splits_data(args, train_df, tr_site_info)
    te_docs, _ = prep_splits_data(args, test_df, te_site_info)
    tv_docs, _ = prep_splits_data(args, val_df, tv_site_info)

    tr_info = list(zip(tr_site_info,tr_histology_info))
    tv_info = list(zip(tv_site_info,tv_histology_info))
    te_info = list(zip(te_site_info,te_histology_info))

    return tr_docs, te_docs,tv_docs, tr_info, te_info,tv_info


def prep_splits_data(args, split_labels, site_info):
    documents = []
    # print(split_labels.head())
    # print(site_info.values())
    df = split_labels
    for i in range(df.shape[0]):
        if '//' in str(df.index[i]):
            filename = df.index[i].split('//')[1].strip() + '.txt.hstlgy'
        else:
            filename = df.index[i].strip()
        if args.use_full_features:
            fname = args.data_dir + "/" + filename.split('.hstlgy')[0].strip()
        else:
            fname = args.data_dir + "/" + filename
        if args.use_site_info and args.task == 'histology':
            # if not fname in site_info.keys(): print(fname)
            site_inform = site_info[i]
            doc = str(site_inform) + " " + open(fname, 'r', encoding="utf8").read().strip()
        else:
            # print(fname)
            doc = open(fname, 'r', encoding="utf8").read().strip()
            doc = clearup(doc)
        documents.append(doc)
        if args.task == 'histology':
            labels = df.histology.values
        else:
            labels = df.site.values
    return documents, labels


# Preprocessing

def clearup(document):
    document = document.translate(string.punctuation)
    # pdb.set_trace()
    numbers = re.search('[0-9]+', document)
    document = re.sub('\(\d+.\d+\)|\d-\d|\d', '', document) \
        .replace('.', '').replace(',', '').replace(',', '').replace(':', '').replace('~', '') \
        .replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('/', '') \
        .replace('%', '').replace('(', '').replace(')', '').replace('?', '') \
        .replace('—', '').replace(';', '').replace('&quot', '').replace('&lt', '') \
        .replace('^', '').replace('"', '').replace('{', '').replace('}', '').replace('\\', '').replace('+', '') \
        .replace('&gt', '').replace('&apos', '').replace('*', '').strip().lower().split()
    # document = document.translate(string.punctuation)
    # return re.sub('[l]+', ' ', str(document)).strip()
    # pdb.set_trace()
    return document


def size(alist):
    return len(alist)


def prep_data_CNN(documents):
    """
    Prepare the padded docs and vocab_size for CNN training
    """
    # pdb.set_trace()
    t = Tokenizer()
    docs = list(filter(None, documents))
    print("Size of the documents in prep_data {}".format(len(documents)))
    # with open("./data/features_full/prep_documents.txt", 'w') as f:
    #     f.write("\n".join(str(tr_d) for tr_d in docs))
    # sys.exit(0)
    t.fit_on_texts(docs)

    vocab_size = len(t.word_counts)
    print("Vocab size {}".format(vocab_size))
    encoded_docs = t.texts_to_sequences(docs)
    print("Size of the encoded documents {}".format(len(encoded_docs)))
    e_lens = []
    for i in range(len(encoded_docs)):
        e_lens.append(len(encoded_docs[i]))
    # print("Encoded docs lengths {}".format(e_lens))
    lens_edocs = list(map(size, encoded_docs))
    max_length = np.average(lens_edocs)
    sequence_length = 1500  # Can use this instead of the above average max_length value
    max_length = sequence_length
    # encoded_docs = [one_hot(d, vocab_size) for d in data_X]
    padded_docs = pad_sequences(
        encoded_docs, maxlen=int(max_length), padding='post')
    # print("encoded docs : {}".format(encoded_docs[:5]))
    # print("padded docs {}".format(padded_docs[:5]))
    print("Length of a padded row {}".format(padded_docs.shape))
    print("max_length {} and min_length {} and average {}".format(
        max_length, min(lens_edocs), np.average(lens_edocs)))
    return padded_docs, max_length, vocab_size, t.word_index

def word2Vec(docs,word_index):
    # data goes here!!!!
    # sentences = [['this', 'is', 'report', 'one'], \
    #              ['this', 'is', 'report', 'two'], \
    #              ['0', '5', '6', '4', '6']]
    # train word2vec
    sentences = docs
    model = Word2Vec(sentences, min_count=1, size=300, workers=4, iter=10)
    # save all word embeddings to matrix
    vocab = zeros((len(word_index) + 1, 300))
    word2idx = word_index
    # for key, val in model.wv.vocab.items():
    #     idx = val.__dict__['index'] + 1
    #     vocab[idx, :] = model[key]
    #     word2idx[key] = idx
    for key, val in model.wv.vocab.items():
        if key in word2idx:
            idx = word2idx[key]
            vocab[idx, :] = model[key]
    # add additional word embedding for unknown words
    unk = len(vocab)
    vocab = vstack((vocab, random.rand(1, 300) - 0.5))

    # normalize embeddings
    vocab -= vocab.mean()
    vocab /= (vocab.std() * 2.5)
    vocab[0, :] = 0
    max_len = 1500
    # convert words to indices
    text_idx = zeros((len(sentences), max_len))
    for i, sent in enumerate(sentences):
        idx = [word2idx[word] if word in model.wv.vocab else unk for word in sent][:max_len]
        l = len(idx)
        text_idx[i, :l] = idx
    # save data
    return text_idx,word2idx,vocab


if __name__ == '__main__':
    tr_docs, te_docs, tv_docs, y_train, y_test, y_val = get_split_docs2(args)
    prep_docs = tr_docs + tv_docs + te_docs

    padded_docs, max_length, vocab_size, word_index = prep_data_CNN(prep_docs)

    padded_docs, word2idx, vocab = word2Vec(prep_docs,word_index)

    with open('data/word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    save('data/vocab.npy', vocab)
    train_x = padded_docs[:len(tr_docs)]
    val_x = padded_docs[len(tr_docs):len(tr_docs) + len(tv_docs)]
    test_x = padded_docs[len(tr_docs) + len(tv_docs):]
    #
    np.save("data/npy/train_X.npy",train_x)
    np.save("data/npy/test_X.npy",test_x)
    np.save("data/npy/val_X.npy",val_x)
    np.save("data/npy/train_Y.npy",np.array(y_train))
    np.save("data/npy/test_Y.npy", np.array(y_test))
    np.save("data/npy/val_Y.npy", np.array(y_val))
# MtCNN => Multi-task labels
# HiSAN