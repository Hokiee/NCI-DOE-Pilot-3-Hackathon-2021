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
from abs_funcs import abstention_loss, save_results


def main():
    # mtcnn parameters
    wv_len = 300
    seq_len = 1500
    batch_size = 16
    epochs = 100
    filter_sizes = [3, 4, 5]
    num_filters = [100, 100, 100]
    concat_dropout_prob = 0.5
    emb_l2 = 0.001
    w_l2 = 0.01
    optimizer = 'adam'
    tasks = ['site', 'histology']
    num_tasks = len(tasks)

    train_x = np.load(r'../data/npy/train_X.npy')
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

    # build a dictionary here to consolidate all the settings
    # params for abstention
    params = {}
    params.update({'max_abs': [0.1, 0.4]})
    params.update({'min_acc': [0.98, 0.90]})
    params.update({'abs_gain': 0.5})
    params.update({'acc_gain': 0.5})
    params.update({'alpha_scale_factor': [0.8, 0.8]})
    params.update({'abs_tasks': [1, 1]})
    params.update({'alpha_init': [0.1, 0.1]})
    params.update({'n_iters': 1})
    params.update({'task_names': tasks})
    params.update({'task_list': [0, 1]})

    print(params)

    # make alpha a Keras variable so it can be modified
    alpha = K.variable(params['alpha_init'])

    max_classes = []
    for task in range(num_tasks):
        cat1 = np.unique(train_y[:, task])
        print(task, len(cat1), cat1)
        cat2 = np.unique(val_y[:, task])
        print(task, len(cat2), cat2)
        cat3 = np.unique(test_y[:, task])
        print(task, len(cat3), cat3)
        cat12 = np.union1d(cat1, cat2)
        cat = np.union1d(cat12, cat3)
        print(task, len(cat), cat)
        max_classes.append(len(cat))

    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn(int(max_vocab) + 1, wv_len).astype('float32') * 0.1

    num_classes = []
    new_train_y = []
    new_test_y = []
    new_val_y = []
    abs_index = []

    for i in range(num_tasks):
        num_classes.append(max_classes[i] + params['abs_tasks'][i])
        new_train_i, new_test_i, new_val_i = modify_labels(num_classes[i],
                                                           train_y[:, i],
                                                           test_y[:, i],
                                                           val_y[:, i])
        new_train_y.append(new_train_i)
        new_test_y.append(new_test_i)
        new_val_y.append(new_val_i)
        abs_index.append(num_classes[i])

    print("Number of classes (including abstention): ", num_classes)

    cnn = init_export_network(
        task_names=tasks,
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

    val_labels = {}
    train_labels = []

    for i in range(num_tasks):
        task_string = tasks[i]
        val_labels[task_string] = new_val_y[i]
        train_labels.append(np.array(new_train_y[i]))

    validation_data = ({'Input': val_x}, val_labels)

    print(val_labels, test_y)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    f = open('abs_stats.csv', 'w+')
    for k in range(num_tasks):
        f.write("%s," % ('alpha_' + tasks[k]))
    for k in range(num_tasks):
        f.write("%s," % ('acc_' + tasks[k]))
    for k in range(num_tasks):
        f.write("%s," % ('abs_' + tasks[k]))
    f.write("\n")
    f.close()

    # initialize manual early stopping
    min_val_loss = np.Inf
    min_comb_loss = 0.01
    min_scale_norm = 0.01
    scale_norm = np.Inf
    p_count = 0
    patience = 10
    monitor = 'val_loss'
    model_saved = False
    saved_epoch = 0

    for epoch in range(epochs):
        h = cnn.fit(x=np.array(train_x),
                    y=train_labels,
                    batch_size=batch_size,
                    initial_epoch=epoch,
                    epochs=epoch + params['n_iters'],
                    verbose=2,
                    validation_data=validation_data)

        scales, alpha = adjust_alpha(params, val_x, val_y, val_labels, cnn, alpha, abs_index)
        scale_norm = np.linalg.norm(np.array(scales) - 1.0, ord=2)

        # manual early-stopping
        combined_loss = h.history[monitor][0] * scale_norm
        if combined_loss < min_comb_loss:
            print("Combined loss improved from ", min_comb_loss, " to ", combined_loss)
            print("Scaling is ", scale_norm)
            print(monitor, " is: ", h.history[monitor][0])
            print("Saving model")
            # set new min, reset patience
            min_comb_loss = combined_loss
            cnn.save(model_name) # , save_weights_only=False)
            p_count = 0
            saved_epoch = epoch
            model_saved = True
        elif model_saved:
            # increment patience
            print('Stopping criterion did not improve from %.4f at epoch %d' % (min_comb_loss, saved_epoch))
            p_count += 1

            if p_count >= patience: # patience limit
                break
        else:
            print('Stopping criterion not yet satisfied %.4f > %.4f' % (combined_loss, min_comb_loss))

    # Predict on Test data
    trues = []
    falses = []
    abstains = []

    task_list = params['task_list']
    max_abs = params['max_abs']
    min_acc = params['min_acc']
    if model_saved:  # otherwise use latest
        print("Loading best saved model from epoch", saved_epoch)
        cnn.load_weights(model_name)
    else:
        print("Using latest model")
    pred_probs = cnn.predict(np.array(test_x))
    print('Prediction on test set')
    for t in range(len(tasks)):
        preds = [np.argmax(x) for x in pred_probs[t]]
        pred_max = [np.max(x) for x in pred_probs[t]]
        y_pred = preds
        y_true = test_y[:, t]
        y_prob = pred_max

        true = K.eval(K.sum(K.cast(K.equal(y_pred, y_true), 'int64')))
        false = K.eval(K.sum(K.cast(K.not_equal(y_pred, y_true), 'int64')))
        abstain = K.eval(K.sum(K.cast(K.equal(y_pred, num_classes[t] - 1), 'int64')))

        trues.append(true)
        falses.append(false)
        abstains.append(abstain)

        # generate the results on the base classes
        preds = np.array(y_pred)
        base_pred = preds[preds < num_classes[t] - 1]
        base_true = y_true[preds < num_classes[t] - 1]

        micro = f1_score(base_true, base_pred, average='micro')
        macro = f1_score(base_true, base_pred, average='macro')
        print('task %12s test f-score: %.4f, %.4f, abstention: %.4f ' % (tasks[t], micro, macro, abstain / (true + false)))
        #print(confusion_matrix(base_true, base_pred))
        save_results(tasks[t], y_true, y_pred, y_prob, num_classes[t])

    print("Detailed abstention results")
    print_abs_stats(tasks, task_list, alpha, scales, trues, falses, abstains, max_abs, min_acc)


if __name__ == "__main__":
    main()
