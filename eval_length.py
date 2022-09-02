import argparse
import logging
import os
import torch

from utils import model_load_state_dict, create_loss_fn
from text_main import create_model, evaluate, get_data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--data-path', default='/content/drive/MyDrive/Colab Notebooks/data', type=str, help='data path')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')

logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()

    args.device = torch.device('cuda', args.gpu)

    student_model = create_model(args)

    labeled_loader, unlabeled_loader, valid_loader, test_loader = get_data_loaders(args)

    criterion = create_loss_fn(args)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    evaluate(args, test_loader, student_model, criterion)
