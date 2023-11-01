"""
Performance evaluation of Spin-Wave Active Ring Neural Network (SWARNN) performing classification, regression and prediction tasks.
Author: Anirban Mukhopadhyay
Affiliation: Prof. Anil Prabhakar's Magnonics group
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.model_selection import StratifiedKFold, train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, KBinsDiscretizer
from numpy.linalg import pinv
import seaborn as sns
import pandas as pd

# a list of linear regressor
linregr = [LinearRegression, Ridge, Lasso, ElasticNet]


# visualization of the relations between feature variables
def vis_feat_rels(df_with_targetlabels: pd.core.frame.DataFrame, target_col_name: str,
                  save_paths: list[str] = None):
    """
    param df_with_targetlabels: Panda dataframe, where the column named
    have the target variable information
    param save_paths: path including the figure name for saving the pairplot and boxplot
    return:
    """
    # Feature pair plots
    pplt = sns.pairplot(df_with_targetlabels, hue=target_col_name, dropna=True, diag_kind='hist')

    # Feature box plots
    fig, ax = plt.subplots()
    df_with_targetlabels.boxplot(by=target_col_name, figsize=(10, 10), ax=ax,
                                 grid=False, patch_artist=True)
    fig.suptitle('')
    # for ax in axes.flatten():
    #     ax.set_xlabel([])
    # fig.supxlabel(target_col_name)

    if save_paths is not None:
        pplt.savefig(save_paths[0], dpi=300, bbox_inches='tight')
        fig.savefig(save_paths[1], dpi=300, bbox_inches='tight')


def find_nearest(arr, val, return_indx=False):
    """
    param arr: a numpy array or list
    param val: a value
    return: return the element of the array which is closest to the value
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    indx = np.argmin(np.abs(arr - val))
    if return_indx:
        return indx, arr[indx]
    else:
        return arr[indx]


def eval_classif(H, y, regr_modelID: int = 0,
                 testsize: float = 0.20, target_labels=None, plot_conf_mat=False, save_path=None):
    """
    param H: Hidden layer output matrix
    param y: Target variable
    param class_labels: discrete categories
    param save_path: path including the figure name for saving confusion matrix
    return: output layer weight matrix via ridge regression
    """

    oheenc = OneHotEncoder()

    H_train, H_test, y_train, y_test = train_test_split(H, y, test_size=testsize,
                                                        random_state=1, shuffle=True, stratify=y)

    # Source: https://stackoverflow.com/questions/55525195/
    # do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset
    y_train_enc = oheenc.fit_transform(y_train.reshape(-1, 1)).toarray()

    # Create ridge regression object
    print(f'You have chosen {linregr[regr_modelID]}.')
    regr = linregr[regr_modelID]()

    # Train the model using the train sets
    regr.fit(H_train, y_train_enc)

    # output weight matrix
    wout = regr.coef_

    # Make predictions using the testing set
    output = regr.predict(H_test)

    # Predicted label
    y_pred = oheenc.inverse_transform(output)

    if plot_conf_mat:
        im = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_labels)
        if save_path is not None:
            fig = im.figure_
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    print(classification_report(y_test, y_pred, labels=target_labels))

    return wout


def eval_timeseries_regr_cv(regr_modelID: int, H, y, numofFold: int = 5):
    """
    param H: Hidden layer output matrix
    param y: Target variable
    return: output layer weight matrix via ridge regression
    """
    # Create linear regression object
    print(f'You have chosen {linregr[regr_modelID]}.')
    regr = linregr[regr_modelID]()

    mse_cv = []
    y_pred = []
    y_test = []
    wout = []

    tss = TimeSeriesSplit(n_splits=numofFold)

    for i, (train_index, test_index) in enumerate(tss.split(H, y)):
        print(f"Fold {i}:" + f' Length of training dataset: {len(train_index)}' +
              f' Length of testing dataset: {len(test_index)}')

        H_train, H_test = H[train_index], H[test_index]
        y_train, ytest = y[train_index], y[test_index]
        y_test.append(ytest)

        # Train the model using the train sets
        regr.fit(H_train, y_train)

        # Output weight
        wout.append(regr.coef_)

        # Make predictions using the testing set
        output = regr.predict(H_test)
        y_pred.append(output)

        # MSE for each fold
        mse_cv.append(np.round(mean_squared_error(ytest, output), 2))

    mse_cv = np.array(mse_cv)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    wout = np.array(wout)

    print('MSE averaged over all the folds:', np.round(np.mean(mse_cv), 2))

    return y_test[-1, ...], y_pred[-1, ...], wout[-1, ...], mse_cv


def eval_classif_cv(regr_modelID: int, H, y, numofFold: int = 5,
                    target_labels=None, plot_conf_mat=False, save_path=None):
    """
    Training linear regression models on subsets of the available input data
    and evaluating them on the complementary subset of the data.
    param H: Hidden layer output matrix
    param y: Target variable
    return: Output weight matrix for fold with the highest accuracy.
            Accuracy score, precision and recall for all the folds.
    """
    # Create linear regression object
    print(f'You have chosen {linregr[regr_modelID]}.')
    regr = linregr[regr_modelID]()

    oheenc = OneHotEncoder()

    # Source: https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/
    accuracy_score_cv = []
    precision_cv = []
    recall_cv = []
    y_pred = []
    y_test = []
    wout = []

    skf = StratifiedKFold(n_splits=numofFold, random_state=1, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(H, y)):
        print(f"Fold {i}:" + f' Length of training dataset: {len(train_index)}' +
              f' Length of testing dataset: {len(test_index)}')

        H_train, H_test = H[train_index], H[test_index]
        y_train, ytest = y[train_index], y[test_index]
        y_test.append(ytest)

        # Source: https://stackoverflow.com/questions/55525195/
        # do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset
        y_train_enc = oheenc.fit_transform(y_train.reshape(-1, 1)).toarray()

        # Train the model using the train sets
        regr.fit(H_train, y_train_enc)

        # Output weight
        wout.append(regr.coef_)

        # Make predictions using the testing set
        output = oheenc.inverse_transform(regr.predict(H_test)).flatten()

        y_pred.append(output)
        accuracy_score_cv.append(np.round(balanced_accuracy_score(ytest, output), 2))
        precision_cv.append(np.round(precision_recall_fscore_support(ytest, output, average='weighted')[0], 2))
        recall_cv.append(np.round(precision_recall_fscore_support(ytest, output, average='weighted')[1], 2))

    accuracy_score_cv = np.array(accuracy_score_cv)
    precision_cv = np.array(precision_cv)
    recall_cv = np.array(recall_cv)
    y_pred = np.array(y_pred, dtype=object)
    y_test = np.array(y_test, dtype=object)
    wout = np.array(wout)

    print('Accuracy score averaged over all the folds:', np.round(np.mean(accuracy_score_cv), 2),
          'Precision averaged over all the folds:', np.round(np.mean(precision_cv), 2),
          'Recall averaged over all the folds:', np.round(np.mean(recall_cv), 2))

    # Plot confusion matrix for the fold which has maximum accuracy
    foldindx_maxacc = np.argmax(accuracy_score_cv)

    if plot_conf_mat:
        im = ConfusionMatrixDisplay.from_predictions(y_test[foldindx_maxacc, ...].astype(str),
                                                     y_pred[foldindx_maxacc, ...].astype(str),
                                                     display_labels=target_labels)
        if save_path is not None:
            fig = im.figure_
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return wout, accuracy_score_cv, precision_cv, recall_cv
                      

def feat_lin_scaleto_exp_param(dataset: np.ndarray, exp_param_min: np.ndarray,
                               exp_param_max: np.ndarray, decimals: int = 2):
    """
    param dataset: numpy array where each column represents a feature
    param exp_param_min: Minimum values of the experimental parameters

    param exp_param_max: Maximum values of the experimental parameters
    param decimals: Number of decimal places to round to
    return: transformed dataset where feature space is mapped on to experimental parameter space
    """
    slope = (exp_param_max - exp_param_min) / (np.amax(dataset, axis=0) - np.amin(dataset, axis=0))
    intercept = exp_param_max - slope * np.amax(dataset, axis=0)
    print("Slope:", slope)
    print('Intercept:', intercept)
    transformed_dataset = slope * dataset + intercept

    return np.round(transformed_dataset, decimals)


# Calculates the binomial coefficient nCr using the logarithmic formula
def nCr(n, r):
    # If r is greater than n, return 0
    if r > n:
        return 0

    # If r is 0 or equal to n, return 1
    if r == 0 or n == r:
        return 1
    # Initialize the logarithmic sum to 0
    res = 0

    # Calculate the logarithmic sum of the numerator and denominator using loop
    for i in range(r):
        # Add the logarithm of (n-i) and subtract the logarithm of (i+1)
        res += np.log(n - i) - np.log(i + 1)
    # Convert logarithmic sum back to a normal number
    return round(np.exp(res))


# Calculates number of states for each feature
def calc_numofFreqBins(numofInstances: int, numofFeats: int):
    val = 0
    numofFreqBins = numofFeats - 1

    while val <= numofInstances:
        numofFreqBins += 1
        val = nCr(numofFreqBins + numofFeats - 1, numofFeats)

    print(f'Number of combinations possible with {numofFreqBins} frequency bins is {val}.')

    return numofFreqBins


def feat_bininngto_exp_param(dataset: np.ndarray, exp_param_vals: np.ndarray):
    """
    param dataset: shape: (num of values, num of features)
    param exp_param_vals: discretized parameter values of shape: (num of values, num of features) or (num of values)
    """
    # calculating number of bins
    n_fvals = exp_param_vals.shape[0]

    est = KBinsDiscretizer(n_bins=n_fvals, encode='ordinal', strategy='uniform')

    indx = est.fit_transform(dataset).astype(int)

    ds_transformed = np.zeros_like(dataset).astype(float)

    if exp_param_vals.ndim == 1:
        for i in range(dataset.shape[1]):
            ds_transformed[:, i] = exp_param_vals[indx[:, i]]
    else:
        for i in range(dataset.shape[1]):
            ds_transformed[:, i] = exp_param_vals[indx[:, i], i]

    return ds_transformed


def find_unique_freqvals(freqvals: np.ndarray):
    """
    param freqvals: array of frequency values of shape: (num of freq values, num of features) or (num of freq values)
    """
    decimal_places = 0

    while True:
        # Round the array to the current decimal place
        rounded_arr = np.round(freqvals, decimals=decimal_places)

        # Check if all values are unique at this decimal place
        if (np.unique(rounded_arr, axis=0)).size < freqvals.size:
            decimal_places += 1
        else:
            break

    return np.round(freqvals, decimals=decimal_places)
