import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=str, default="train", choices=["train","test", 'submit'])
parser.add_argument('-train', type=str)
parser.add_argument('-test', type=str)
parser.add_argument('-verbose', action='store_true', default=False)
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-len_max_seq', type=int, default=512)
parser.add_argument('-bert_dir', type=str)
parser.add_argument('-do_lower_case', action='store_true', default=False)
parser.add_argument('-save', type=str, default='./save')
parser.add_argument('-predict', type=str, default='./predict')

parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.00001)
parser.add_argument('-l2', type=float, default=1e-8)

parser.add_argument('-iter', type=int, default=100)
parser.add_argument('-gpu', type=int, default=-1)
parser.add_argument('-patience', type=int, default=20)


opt = parser.parse_args()
