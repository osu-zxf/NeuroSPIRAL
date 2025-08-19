import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--model_folder', default='results/ckpts')
parser.add_argument('--result_folder', default='results/logs')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--best_model_path', default='results/ckpts/tox21/model_epoch_1.pth')
parser.add_argument('--dataset', type=str, default='toxcast', choices=['toxcast', 'tox21', 'toxric'])
args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
