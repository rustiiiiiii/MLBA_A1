'''MLBA ASSIGNMENT 2
Group 18
'''

import numpy as np
import pandas as pd
import re
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from Pfeature.pfeature import *

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
               'R', 'S', 'T', 'V', 'W', 'Y']  # amino acid symbols to assist in feature generation

# initialise empty lists for collecting feature arrays during composition feature generation
A = []
C = []
D = []
E = []
F = []
G = []
H = []
I = []
K = []
L = []
M = []
N = []
P = []
Q = []
R = []
S = []
T = []
V = []
W = []
Y = []

# prompt user to input the paths of training and testing csv files


# open the train and test data csv files as specified by the user entered paths
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# extract the amino acid residue sequences from the training data
train_data_seq = train_data.iloc[:, 1]

# extract the class labels (0,1) corresponding to the amino acid residue sequences from the training data
train_data_labels = train_data.iloc[:, 0]

# extract the amino acid residue sequences from the testing data
test_data_seq = test_data.iloc[:, 1]


#-----------------------------------------------------------------------------------------------------------------#


def clear_arrays():
    ''' Clears the global arrays that collect the respective composition feature data for a dataset.
    Input: None
    Return: None
    '''
    global A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    A = []
    C = []
    D = []
    E = []
    F = []
    G = []
    H = []
    I = []
    K = []
    L = []
    M = []
    N = []
    P = []
    Q = []
    R = []
    S = []
    T = []
    V = []
    W = []
    Y = []


def feature_engineering(dataset):
    ''' Responsible for feature generation (engineering) of a dataset containing peptide sequences.
    Amino acid composition of a sequence is being used in generating features for that sequence.
    Input: A dataframe containing peptide sequences.
    Returns: A dataframe contaning the new features (Amino acid composition) corresponding the input dataset
    '''
    aac_wp('train.csv', 'a.csv')
    dataset_comp = pd.read_csv('a.csv')
    dataset_comp.drop(0, inplace=True)
    return dataset_comp


#-----------------------------------------------------------------------------------------------------------------#


def machine_learning_model():
    ''' Uses Stacking Classifier to stack the Random Forest Classifier with the Logistic Regression Classifier.
    Logistic Regression Classifier is used to combine the Random Forest base estimator.
    Return: the model variable.
    '''
    level0 = list()  # base estimators
    level0.append(('lr', LogisticRegression()))
    level0.append(('rf', RandomForestClassifier(n_estimators=700,
                                                oob_score="True", n_jobs=-1, max_features="sqrt")))
    # classifier which will be used to combine the base estimators.
    level1 = LogisticRegression()
    # default 5-fold cross validation
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


def evaluate_and_fit_model(train_features, train_labels):
    ''' Gets the machine learning model and preforms kfold cross validation on the model with accuracy as scoring criteria. 
    Displays the cross validation results. Then fits the model on the entire training data so that it can be used to make 
    predictions on test dataset.
    Input: the training amino acid composition features and training labels as separate dataframes.
    Return: the final refitted model
    '''
    model = machine_learning_model()  # gets the machine learning model variable
    # stratified 10-Fold cross validation repeated 5 times with different randomization in each repetition.
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores = cross_val_score(model, train_features, train_labels,
                             scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print("\nAccuracy scores for the model from k-fold cross validation: \n", scores)
    print('\nMean accuracy score: %.3f' % (mean(scores)))
    print('\nStandard deviation of accuracy score: %.3f' % (std(scores)))
    model.fit(train_features, train_labels)
    return model


#-----------------------------------------------------------------------------------------------------------------#


def main():
    global train_data_labels, train_data_seq, test_data_seq

    # Amino acid composition of the sequences in training dataset (feature generation)
    train_data_comp = feature_engineering(train_data_seq)

    # Amino acid composition of the sequences in testing dataset (feature generation)
    test_data_comp = feature_engineering(test_data_seq)

    model = evaluate_and_fit_model(train_data_comp, train_data_labels)
    predictions = model.predict_proba(test_data_comp)

    # create the final predictions dataframe and export it as a csv.
    finalpredictions = pd.DataFrame(
        {'ID': np.array(test_data.iloc[:, 0]), 'Label': predictions[:, 0]})
    finalpredictions.to_csv('Group_18_Predictions_Output.csv', index=False)


if __name__ == "__main__":
    main()
