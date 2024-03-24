import numpy as np
import pandas as pd
import argparse
from utils import fix_seed,make_folder

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

ver = 3

def main(args):
    ######## generative setting #########
    fix_seed(seed=args.seed)
    make_folder(args.output_path)
    ######## load data #########

    train_df = pd.read_csv(args.dataset_path + 'train.csv')
    test_df = pd.read_csv(args.dataset_path + 'test.csv')
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--n_fold', default=4,  type=int)
    parser.add_argument('--training_fold', default=0,  type=int)
    parser.add_argument('--is_training', default=1,  type=int)
    parser.add_argument('--is_evaluation', default=0,  type=int)
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--eval_step', default=1000, type=int)
    parser.add_argument('--patience', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--mu', default=7, type=int)
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--lambda-u', default=1, type=float)
    parser.add_argument('--dataset_path',
                        default='./dataset/', type=str)
    parser.add_argument('--output_path',
                        default='./output/debug/', type=str)
    args = parser.parse_args()

    #################
    main(args)