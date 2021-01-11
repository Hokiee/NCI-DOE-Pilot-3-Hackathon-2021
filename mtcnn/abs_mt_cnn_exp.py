"""
DISCLAIMER
UT-BATTELLE, LLC AND THE GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES,
BOTH EXPRESSED AND IMPLIED.  THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY
OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY
PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL
ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY
OR DAMAGE.  THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS,
CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF, IN
WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
"""
"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from abs_keras_mt_shared_cnn import init_export_network
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
from keras.models import load_model
import math

import numpy as np
import time
import os

import urllib.request
import ftplib

from keras import backend as K
from keras.callbacks import LambdaCallback

from abs_funcs import modify_labels
from abs_funcs import print_abs_stats, write_abs_stats, adjust_alpha


def main():
    # mtcnn parameters
    wv_len = 300
    seq_len = 1500
    num_tasks = 2
    batch_size = 16
    epochs = 100
    filter_sizes = [3, 4, 5]
    num_filters = [100, 100, 100]
    concat_dropout_prob = 0.5
    emb_l2 = 0.001
    w_l2 = 0.01
    optimizer = 'adam'

    # build a dictionary here to consolidate all the settings
    # params for abstention
    params = {}
    params.update({'max_abs': [0.1, 0.4]})
    params.update({'min_acc': [0.99, 0.90]})
    params.update({'abs_gain': 1.0})
    params.update({'acc_gain': 1.0})
    params.update({'alpha_scale_factor': [0.8, 0.8]})
    params.update({'abs_tasks': [1, 1]})
    params.update({'alpha_init': [0.1, 0.1]})
    params.update({'n_iters': 1})

    print(params)

    # make alpha a Keras variable so it can be modified
    alpha = K.variable(params['alpha_init'])

    train_x = np.load(r'../data/npy/train_X.npy')#./data/train_X.npy')
    train_y = np.load(r'../data/npy/train_Y.npy')
    val_x = np.load(r'../data/npy/val_X.npy')
    val_y = np.load(r'../data/npy/val_Y.npy')
    test_x = np.load(r'../data/npy/test_X.npy')
    test_y = np.load(r'../data/npy/test_Y.npy')

    print('Data shapes')
    print('Train X ', train_x.shape)
    print('Train Y ', train_y.shape)
    print('Val X ', val_x.shape)
    print('Val Y ', val_y.shape)
    print('Test X ', test_x.shape)
    print('Test Y ', test_y.shape)

    max_classes = []
    for task in range(len(train_y[0, :])):
        cat1 = np.unique(train_y[:, task])
        print(task, len(cat1), cat1)
        cat2 = np.unique(val_y[:, task])
        print(task, len(cat2), cat2)
        cat3 = np.unique(test_y[:, task])
        print(task, len(cat3), cat3)
        cat12 = np.union1d(cat1, cat2)
        cat = np.union1d(cat12, cat3)
        print(task, len(cat), cat)
        train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
        test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]
        val_y[:, task] = [np.where(cat == x)[0][0] for x in val_y[:, task]]
        max_classes.append(len(cat))

    # for task in range(len(train_y[0, :])):
    #     cat = np.unique(train_y[:, task])
    #     train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
    #     test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]

    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn( int(max_vocab) + 1, wv_len ).astype( 'float32' ) * 0.1

    #num_classes = np.max(train_y) + 1

    num_classes = []
    new_train_y = []
    new_test_y = []
    new_val_y = []
    abs_index = []

    for i in range(num_tasks):
        #num_classes.append(np.max(train_y[:, i]) + 1 + params['abs_tasks'][i])
        num_classes.append(max_classes[i] + 1 + params['abs_tasks'][i])
        new_train_i, new_test_i, new_val_i = modify_labels(num_classes[i],
                                                           train_y[:, i],
                                                           test_y[:, i],
                                                           val_y[:, i])
        new_train_y.append(new_train_i)
        new_test_y.append(new_test_i)
        new_val_y.append(new_val_i)
        abs_index.append(num_classes[i])

    # num_classes.append(np.max(train_y[:,1]) + 1)
    # num_classes.append(np.max(train_y[:,2]) + 1)
    # num_classes.append(np.max(train_y[:,3]) + 1)

    print("Number of classes (including abstention): ", num_classes)

    cnn = init_export_network(
        num_classes=num_classes,
        alpha_init=params['alpha_init'],
        in_seq_len=seq_len,
        vocab_size=len(wv_mat),
        wv_space=len(wv_mat[0]),
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        concat_dropout_prob=concat_dropout_prob,
        emb_l2=emb_l2,
        w_l2=w_l2,
        optimizer=optimizer)

    model_name = 'abs_mt_cnn_model.h5'
    print("Model file: ", model_name)

    print(cnn.summary())

    print(cnn.output_names)

    val_labels = {}
    train_labels = []
    #task_names = ['Dense0', 'Dense1']
    task_names = cnn.output_names
    task_list = [0, 1]

    params.update({'task_names': task_names})
    params.update({'task_list': task_list})

    for i in range(train_y.shape[1]):
        if i in task_list:
            task_string = task_names[i]
            val_labels[task_string] = new_val_y[i]
            train_labels.append(np.array(new_train_y[i]))

    validation_data = ({'Input': val_x}, val_labels)

    print(val_labels, test_y)
    #validation_data = (
    #    { 'Input': np.array(test_x) },
    #    {
    #        'Dense0': test_y[:, 0],
    #        'Dense1': test_y[:, 1],
    #        # 'Dense2': test_y[:, 2],
    #        # 'Dense3': test_y[:, 3],
    #    }
    #)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    # initialize manual early stopping
    min_val_loss = np.Inf
    min_scale_norm = 0.01
    scale_norm = np.Inf
    p_count = 0
    patience = 10
    monitor = 'val_loss'

    for epoch in range(epochs):
        h = cnn.fit(x=np.array(train_x),
                    y=train_labels,
                    batch_size=batch_size,
                    initial_epoch=epoch,
                    epochs=epoch + params['n_iters'],
                    verbose=2,
                    validation_data=validation_data,
                    callbacks=[checkpointer, stopper])

        # manual early-stopping 
        if (h.history[monitor][0] < min_val_loss) and (scale_norm < min_scale_norm):
            print("Scaling improved from ", min_scale_norm, " to ", scale_norm)
            print(monitor, " improved from ", min_val_loss, " to ", h.history[monitor][0])
            print("Saving model")
            # set new min, reset patience
            min_val_loss = h.history[monitor][0]
            min_scale_norm = scale_norm
            cnn.save(model_name) # , save_weights_only=False)
            p_count = 0
        else:
            # increment patience
            p_count += 1

        if p_count >= patience: # patience limit
            if scale_norm < min_scale_norm: # scaling factors close to 1 (abstention and accuracy limit satisfied)
                return

        scales, alpha = adjust_alpha(params, val_x, val_y, val_labels, cnn, alpha, abs_index)
        scale_norm = np.linalg.norm(np.array(scales)-1.0)
        print(scale_norm)

    # TODO: Add performance on test set.

if __name__ == "__main__":
    main()
