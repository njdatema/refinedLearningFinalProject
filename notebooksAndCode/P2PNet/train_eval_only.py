import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crowd_datasets import build_dataset
from engine import evaluate_crowd_no_overlap
from models import build_model
import util.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet evaluation-only script', add_help=False)

    # keep most original args for compatibility
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='Unused for evaluation-only mode.')
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help='Name of the convolutional backbone to use')

    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--row', default=2, type=int, help='row number of anchor points')
    parser.add_argument('--line', default=2, type=int, help='line number of anchor points')

    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    parser.add_argument('--output_dir', default='./eval_log',
                        help='directory for evaluation logs')
    parser.add_argument('--resume', default='',
                        help='path to checkpoint (.pth) to evaluate')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    # new options
    parser.add_argument('--test_only', action='store_true',
                        help='evaluate only and exit')
    parser.add_argument('--vis_dir', default='',
                        help='optional directory to save visualized predictions')
    parser.add_argument('--split', default='test', choices=['val', 'test'],
                        help='which evaluation split to use; requires loader support')
    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_name = os.path.join(args.output_dir, 'eval_log.txt')

    with open(run_log_name, 'w') as log_file:
        log_file.write('Eval Log %s\n' % time.strftime('%c'))
        log_file.write(f'{args}\n')

    if not args.resume:
        raise ValueError('Please provide --resume /path/to/checkpoint.pth for evaluation.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model and criterion. criterion is not used during evaluation but retained for compatibility.
    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)
    model_without_ddp = model

    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    missing, unexpected = model_without_ddp.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint: {args.resume}')
    print('Missing keys:', missing)
    print('Unexpected keys:', unexpected)

    # Dataset loader
    loading_data = build_dataset(args=args)

    # Support both old loader signature loading_data(data_root)
    # and a revised one that can return (train, val, test).
    dataset_bundle = loading_data(args.data_root)

    if isinstance(dataset_bundle, tuple):
        if len(dataset_bundle) == 3:
            _, val_set, test_set = dataset_bundle
            eval_set = test_set if args.split == 'test' else val_set
        elif len(dataset_bundle) == 2:
            train_set, val_set = dataset_bundle
            # In the original repo, the second returned set is called val_set,
            # but for many custom setups it is actually the held-out test set.
            eval_set = val_set
            print("Loader returned only 2 splits. Using the second split for evaluation.")
        else:
            raise ValueError('Unexpected number of datasets returned by loading_data().')
    else:
        raise ValueError('loading_data() must return a tuple of datasets.')

    sampler_eval = torch.utils.data.SequentialSampler(eval_set)
    data_loader_eval = DataLoader(
        eval_set,
        1,
        sampler=sampler_eval,
        drop_last=False,
        collate_fn=utils.collate_fn_crowd,
        num_workers=args.num_workers,
    )

    vis_dir = args.vis_dir if args.vis_dir else None
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    mae, mse = evaluate_crowd_no_overlap(model, data_loader_eval, device, vis_dir=vis_dir)

    msg = f'Evaluation finished on split={args.split}. MAE={mae:.4f}, RMSE={mse:.4f}'
    print(msg)
    with open(run_log_name, 'a') as log_file:
        log_file.write(msg + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'P2PNet evaluation-only script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
