from keras import backend as K
import numpy as np
from keras.utils import np_utils
import keras


def adjust_alpha(abs_params, X_test, truths_test, labels_val, model, alpha, add_index):

    task_names = abs_params['task_names']
    task_list = abs_params['task_list']
    # retrieve truth-pred pair
    avg_loss = 0.0
    ret = []
    ret_k = []

    # set abstaining classifier parameters
    max_abs = abs_params['max_abs']
    min_acc = abs_params['min_acc']
    alpha_scale_factor = abs_params['alpha_scale_factor']

    #print('labels_test', labels_test)
    #print('Add_index', add_index)

    feature_test = X_test
    #label_test = keras.utils.to_categorical(truths_test)

    loss = model.evaluate(feature_test, labels_val)
    avg_loss = avg_loss + loss[0]

    pred = model.predict(feature_test)
    #print('pred',pred.shape, pred)

    abs_gain = abs_params['abs_gain']
    acc_gain = abs_params['acc_gain']

    accs = []
    abst = []
    scales = []
    trues = []
    falses = []
    abstains = []

    for k in range((alpha.shape[0])):
        if k in task_list:
            truth_test = truths_test[:, k]
            alpha_k = K.eval(alpha[k])
            pred_classes = pred[k].argmax(axis=-1)
            #true_classes = labels_test[k].argmax(axis=-1)
            true_classes = truth_test

            #print('pred_classes',pred_classes.shape, pred_classes)
            #print('true_classes',true_classes.shape, true_classes)
            #print('labels',label_test.shape, label_test)

            true = K.eval(K.sum(K.cast(K.equal(pred_classes, true_classes), 'int64')))
            false = K.eval(K.sum(K.cast(K.not_equal(pred_classes, true_classes), 'int64')))
            abstain = K.eval(K.sum(K.cast(K.equal(pred_classes, add_index[k] - 1), 'int64')))

            #print(true, false, abstain)
            trues.append(true)
            falses.append(false)
            abstains.append(abstain)

            total = false + true
            tot_pred = total - abstain
            abs_acc = 1.0
            abs_frac = abstain / total

            if tot_pred > 0:
                abs_acc = true / tot_pred

            scale_k = alpha_scale_factor[k]
            min_scale = scale_k
            max_scale = 1. / scale_k

            acc_error = abs_acc - min_acc[k]
            acc_error = min(acc_error, 0.0)
            abs_error = abs_frac - max_abs[k]
            abs_error = max(abs_error, 0.0)
            new_scale = 1.0 + acc_gain * acc_error + abs_gain * abs_error

            # threshold to avoid huge swings
            new_scale = min(new_scale, max_scale)
            new_scale = max(new_scale, min_scale)

            scales.append(new_scale)
            # print('Scaling factor: ', new_scale)

            K.set_value(alpha[k], new_scale * alpha_k)

            ret_k.append(truth_test)
            ret_k.append(pred)
            ret_k.append(new_scale)

            ret.append(ret_k)

            accs.append(abs_acc)
            abst.append(abs_frac)
        else:
            accs.append(1.0)
            accs.append(0.0)
            trues.append(0)
            falses.append(0)
            abstains.append(0)
            scales.append(1.0)

    # print(trues, falses, abstains)

    print_abs_stats(task_names, task_list, alpha, scales, trues, falses, abstains, max_abs, min_acc)
    write_abs_stats('abs_stats.csv', alpha, accs, abst)

    return scales, alpha


def print_abs_stats(
        tasks,
        task_list,
        alphas,
        scale,
        trues,
        falses,
        abstains,
        max_abs,
        min_acc):

    print('        task,       alpha,     true,    false,  abstain,    total, tot_pred,   abs_frac,    max_abs,    abs_acc,    min_acc,    scaling')
    for k in range(len(tasks)):
        if k in task_list:
            # Compute interesting values
            total = trues[k] + falses[k]
            tot_pred = total - abstains[k]
            abs_frac = abstains[k] / total
            abs_acc = 1.0
            if tot_pred > 0:
                abs_acc = trues[k] / tot_pred

            print('{:>12s}, {:10.5e}, {:8d}, {:8d}, {:8d}, {:8d}, {:8d}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}'
                  .format(tasks[k], K.get_value(alphas[k]),
                          trues[k], falses[k] - abstains[k], abstains[k], total,
                          tot_pred, abs_frac, max_abs[k], abs_acc, min_acc[k], scale[k]))


def write_abs_stats(stats_file, alphas, accs, abst):

    # Open file for appending
    abs_file = open(stats_file, 'a')

    # we write all the results
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % K.get_value(alphas[k]))
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % accs[k])
    for k in range((alphas.shape[0])):
        abs_file.write("%10.5e," % abst[k])
    abs_file.write("\n")
    abs_file.close()


def modify_labels(numclasses_out, ytrain, ytest, yval=None):
    """ This function generates a categorical representation with a class added for indicating abstention.

    Parameters
    ----------
    numclasses_out : integer
        Original number of classes + 1 abstention class
    ytrain : ndarray
        Numpy array of the classes (labels) in the training set
    ytest : ndarray
        Numpy array of the classes (labels) in the testing set
    yval : ndarray
        Numpy array of the classes (labels) in the validation set
    """

    classestrain = np.max(ytrain) + 1
    classestest = np.max(ytest) + 1
    if yval is not None:
        classesval = np.max(yval) + 1

    print("Classes in training set:", classestrain)
    print("Classes in testing set:", classestest)

    #if (classestrain != classestest):
        #classestrain = max(classestrain, classestest)
        #classestest = max(classestrain, classestest)
    #if yval is not None:
        #assert(classesval == classestest)
    #assert((classestrain + 1) == numclasses_out)  # In this case only one other slot for abstention is created

    labels_train = np_utils.to_categorical(ytrain, numclasses_out)
    labels_test = np_utils.to_categorical(ytest, numclasses_out)
    if yval is not None:
        labels_val = np_utils.to_categorical(yval, numclasses_out)

    # For sanity check
    mask_vec = np.zeros(labels_train.shape)
    mask_vec[:, -1] = 1
    i = np.random.choice(range(labels_train.shape[0]))
    sanity_check = mask_vec[i, :] * labels_train[i, :]
    if ytrain.ndim > 1:
        ll = ytrain.shape[1]
    else:
        ll = 0

    for i in range(ll):
        for j in range(numclasses_out):
            if sanity_check[i, j] == 1:
                print('Problem at ', i, j)

    if yval is not None:
        return labels_train, labels_test, labels_val

    return labels_train, labels_test


def abstention_loss(alpha, mask):
    """ Function to compute abstention loss.
        It is composed by two terms:
        (i) original loss of the multiclass classification problem,
        (ii) cost associated to the abstaining samples.

    Parameters
    ----------
    alpha : Keras variable
        Weight of abstention term in cost function
    mask : ndarray
        Numpy array to use as mask for abstention:
        it is 1 on the output associated to the abstention class and 0 otherwise
    """

    def loss(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        base_pred = (1 - mask) * y_pred + K.epsilon()
        base_true = y_true
        base_cost = K.categorical_crossentropy(base_true, base_pred)
        abs_pred = y_pred[:, -1]
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = K.clip(abs_pred, 0, 1. - K.epsilon())

        return K.mean((1. - abs_pred) * base_cost - alpha * K.log(1. - abs_pred))

    loss.__name__ = 'abs_crossentropy'
    return loss


def abstention_acc_metric(nb_classes):
    """ Abstained accuracy:
        Function to estimate accuracy over the predicted samples
        after removing the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
        It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # matching in original classes
        true_pred = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # total abstention
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

        # total predicted in original classes
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        # guard against divide by zero
        condition = K.greater(total_pred, total_abs)
        abs_acc = K.switch(condition, true_pred / (total_pred - total_abs), total_pred / total_pred)
        return abs_acc

    metric.__name__ = 'abstention_acc'
    return metric


def abstention_metric(nb_classes):
    """ Function to estimate fraction of the samples where the model is abstaining.

    Parameters
    ----------
    nb_classes : int or ndarray
        Integer or numpy array defining indices of the abstention class
    """
    def metric(y_true, y_pred):
        """
        Parameters
        ----------
        y_true : keras tensor
            True values to predict
        y_pred : keras tensor
            Prediction made by the model.
            It is assumed that this keras tensor includes extra columns to store the abstaining classes.
        """
        # total abstention
        total_abs = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), nb_classes), 'int64'))

        # total predicted in original classes
        total_pred = K.sum(K.cast(K.equal(K.argmax(y_pred, axis=-1), K.argmax(y_pred, axis=-1)), 'int64'))

        return total_abs / total_pred

    metric.__name__ = 'abstention'
    return metric


def save_results(task, y_true, y_pred, y_prob, abs_class):
    # print(task, abs_class)
    comb = (y_true, y_pred, y_prob)
    all_dat = np.stack(comb, axis=-1)
    fmt = '%5d', '%5d', '%10.5f'
    np.savetxt('abs_' + task + '_results.txt', all_dat, fmt=fmt)
    #mismatches = np.where(y_true != y_pred)
    mismatches = np.where(y_true != y_pred)[0]
    #print(mismatches)
    #print(np.array(y_prob)[(mismatches)])
    # print(all_dat[(mismatches)])

    # generate the results on the base classes
    preds = np.array(y_pred)
    non_abs = np.where(preds < (abs_class - 1))[0]
    base_true = y_true[non_abs]
    base_pred = preds[non_abs]
    base_prob = np.array(y_prob)[non_abs]
    #base_pred = preds[preds < abs_class - 1]
    #base_true = y_true[preds < abs_class - 1]

    mismatches = np.where(base_true != base_pred)[0]
    # print(mismatches)
    all_mis = np.stack((base_true, base_pred, base_prob), axis = -1)
    # print(all_mis)
    np.savetxt('abs_' + task + '_mismatches.txt', all_mis[mismatches], fmt=fmt)
