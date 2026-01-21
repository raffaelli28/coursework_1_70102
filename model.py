#!/usr/bin/env python3

import argparse
import csv
import random
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.metrics import fbeta_score


def run_rfc_training (x_train, y_train, x_test):
    """
    Function that defines and trains an XGBoost Model.
    Takes as input the x and y samples of the training dataset,
    and returns the trained model.
    """

    #x_train, y_train = make_classification(shuffle = True)
    
    model_rfc = RandomForestClassifier(max_depth = 23, random_state = 0)
    model_rfc.fit(x_train, y_train)

    y_pred = model_rfc.predict(x_test)

    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    r = csv.reader(open(flags.input))
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    next(r) # skip headers

    train_df = pd.read_csv('training.csv')
    test_df = pd.read_csv('test.csv')

    # One-hot encode the gender column
    train_df['sex'] = (train_df['sex'].astype(str).map({"m" : 0, "f" : 1}))
    test_df['sex'] = (test_df['sex'].astype(str).map({"m" : 0, "f" : 1}))

    # Compute minimum, maximum, and mean values of the available creatinine scores
    # Add these as columns to the training dataframe and avoid NaN values.
    result_cols = []
    for i in train_df.columns:
        if 'creatinine_result' in i:
            result_cols.append(i)
        else:
            continue

    date_time_cols = []
    for i in train_df.columns:
        if 'creatinine_date' in i:
            date_time_cols.append(i)
        else:
            continue

    result_cols_test = []
    for i in test_df.columns:
        if 'creatinine_result' in i:
            result_cols_test.append(i)
        else:
            continue

    date_time_cols_test = []
    for i in test_df.columns:
        if 'creatinine_date' in i:
            date_time_cols_test.append(i)
        else:
            continue
    
    train_df['creatinine_result_min'] =  train_df[result_cols].min(axis = 1, skipna = True)
    train_df['creatinine_result_max'] =  train_df[result_cols].max(axis = 1, skipna = True)
    train_df['creatinine_result_mean'] =  train_df[result_cols].mean(axis = 1, skipna = True)

    test_df['creatinine_result_min'] =  test_df[result_cols_test].min(axis = 1, skipna = True)
    test_df['creatinine_result_max'] =  test_df[result_cols_test].max(axis = 1, skipna = True)
    test_df['creatinine_result_mean'] =  test_df[result_cols_test].mean(axis = 1, skipna = True)

    # drop bulk of data
    train_df = train_df.drop(columns = date_time_cols)
    train_df = train_df.drop(columns = result_cols)

    test_df = test_df.drop(columns = date_time_cols_test)
    test_df = test_df.drop(columns = result_cols_test)

    y_train = np.array(train_df['aki'][:])
    x_train = train_df.drop(columns = ['aki'])

    y_test = np.array(test_df['aki'][:])
    x_test = test_df.drop(columns = ['aki'])

    # Setting up Scaler and Scaling the data
    #scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train, y_train)
    #x_test = scaler.transform(x_test)


    # MLP Model
    #model_mlp = MLPClassifier(hidden_layer_sizes = (256, 128, 64, 32), max_iter = 1000, random_state = 42, verbose = True)

    #print(x_train.columns)

    #model_mlp.fit(x_train, y_train)




    # Get Predictions for MLP Classifier
    y_pred = run_rfc_training (x_train, y_train.squeeze(), x_test)

    # Check f_3 score
    #print(f'Shape of prediction: {y_pred.shape}')
    #print(f'Type of data in prediction: {type(y_pred)}')
    #print(f'Shape of ground truth: {y_test.shape}')
    #print(f'Type of data in ground truth: {type(y_test)}')
    f3_score = fbeta_score(y_test, y_pred, beta = 3, pos_label = 'y')

    print(f"The F3_NHS score is: {f3_score}")


if __name__ == "__main__":
    main()