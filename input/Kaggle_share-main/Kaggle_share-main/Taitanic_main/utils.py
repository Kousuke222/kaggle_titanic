import os
import numpy as np
import pandas as pd
import shap
from IPython.core.display import display
ver = 3

def fix_seed(seed):
    np.random.seed(seed)

def make_folder(path):
    p = ''
    for x in path.split('/'):
        p += x+'/'
        if not os.path.exists(p):
            os.mkdir(p)

def display_pd(data):
    # show details of data
    print(f'Train_df_shape : {data.shape}\n')
    print(f'{data.dtypes} \n')
    display(data.head())

def analysis_pd(data):
    # show statistics data and categorical data
    data = data.astype(
        {
        'PassengerId' : str,
        'Pclass' : str 
        }
    )
    print('--statistics--')
    display(data.describe())
    print('--categorical--')
    display(data.describe(exclude='number'))
    
    # return changed data <-- after astype
    return data

def make_all_pd(train_df,test_df):
    # make data adddinf train and test
    all_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
    all_df['Test_Flag'] = 0
    all_df.loc[train_df.shape[0]: , 'Test_Flag'] = 1
    
    return all_df


def complement_pd(data :pd.DataFrame,column :str,mode='median'):
    # complement data median or NAN
    if mode == 'median':
        data[column] = data[column].fillna(data[column].median())

    elif mode == 'NAN':
        data[column] = data[column].fillna('NAN')

    return data

def to_categorical(data,column,band):
    # convert statistics data to categorical data
    data[column] = pd.qcut(data[column], band)

    return data

def show_ratio(data,column1,colmun2):
    # show ratio of colmun2 against each colmun1
    print('=========')
    display(pd.crosstab(data[column1], data[colmun2], normalize='index'))


