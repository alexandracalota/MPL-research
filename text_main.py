import argparse
import logging
import math
import os
import random
import time
import json

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pytorch_pretrained_bert import BertAdam
import wandb
from tqdm import tqdm

from data.textLoader import get_data_loaders
from models import get_model
from models.models import ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn, save_checkpoint, reduce_tensor, model_load_state_dict, all_metrics)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data-path', default='/content/drive/MyDrive/Colab Notebooks/data', type=str, help='data path')
parser.add_argument('--train-file', default='train', type=str, help='train file name')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', ''], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
# parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num-classes', default=4, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=1e-4, type=float, help='train learning late')
parser.add_argument('--student_lr', default=1e-4, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true', help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument('--randaug', nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

# Parse Bert necessary args
parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
parser.add_argument("--hidden-sz", type=int, default=768)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
parser.add_argument('--k-img', default=65536, type=int, help='number of labeled examples')
parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument("--model", type=str, default="bert", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
parser.add_argument('--warmup', default=0.1, type=float, help='warmup epochs (unlabeled data based)')
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument('--text_soft_aug', type=str, default='none', choices=['none', 'back_translation', 'eda'])
parser.add_argument('--text_hard_aug', type=str, default='none', choices=['none', 'back_translation', 'eda'])
parser.add_argument('--text_prob_aug', type=float, default=1.0, help='probability of using augmented text')
parser.add_argument("--task", type=str, default="informative", choices=[
    "mmimdb", "vsnli", "food101", 'disaster_data', 'samples_4k_mmbt', 'informative', 'damage', 'humanitarian',
    'humanitarian1k', 'humanitarianO', 'informativeO', 'humanitarianM', 'humanitarianR', 'imdb'])
parser.add_argument("--max_seq_len", type=int, default=512)

parser.add_argument('--validation', type=bool, default=False)
parser.add_argument('--stats_dir', type=str, default='checkpoint')
parser.add_argument('--debug_p', action='store_true')
parser.set_defaults(debug_p=False)
parser.add_argument('--debug_f', action='store_true')
parser.set_defaults(debug_f=False)


TRAIN_PREDICTIONS = 'train_predictions'
TRAIN_ACTUAL_PREDICTIONS = 'train_actual_predictions'
UNLABELED_PREDICTIONS = 'unlabeled_predictions'
PSEUDO_LABELS = 'pseudo_labels'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def create_model(args):
    model = get_model(args)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    return model

def train_teacher(args, labeled_loader, unlabeled_loader, test_loader,
                  teacher_model, criterion, t_optimizer, t_scheduler, t_scaler):

    logger.info("***** Running Training *****")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        end = time.time()

        try:
            text_l, segment_l, mask_l, _, _, _, tgt_l = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            text_l, segment_l, mask_l, _, _, _, tgt_l = labeled_iter.next()

        data_time.update(time.time() - end)

        texts = torch.cat(text_l).to(args.device)
        segments = torch.cat(segment_l).to(args.device)
        masks = torch.cat(mask_l).to(args.device)

        with amp.autocast(enabled=args.amp):
            batch_size = text_l.shape[0]

            logits = teacher_model(texts, masks, segments)
            t_logits_l = logits[:batch_size]
            del logits

            targets = tgt_l.to(args.device)

            t_loss_l = criterion(t_logits_l, targets)
            t_loss_uda = t_loss_l

            if step % 100 == 0:
                print('\n')
                logger.info(targets)

        t_loss = t_loss_uda

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()

        if args.world_size > 1:
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                wandb.log({"train/1.s_loss": s_losses.avg,
                           "train/2.t_loss": t_losses.avg,
                           "train/3.t_labeled": t_losses_l.avg,
                           "train/4.t_unlabeled": t_losses_u.avg,
                           "train/5.t_mpl": t_losses_mpl.avg,
                           "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss, top1, top5, bin_test = evaluate(args, test_loader, test_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/acc@5", top5, args.num_eval)

                args.writer.add_scalar('precision/label0', bin_test['None/precision'][0], step)
                args.writer.add_scalar('precision/label1', bin_test['None/precision'][1], step)
                args.writer.add_scalar('precision/label2', bin_test['None/precision'][2], step)
                args.writer.add_scalar('precision/label3', bin_test['None/precision'][3], step)

                args.writer.add_scalar('recall/label0', bin_test['None/recall'][0], step)
                args.writer.add_scalar('recall/label1', bin_test['None/recall'][1], step)
                args.writer.add_scalar('recall/label2', bin_test['None/recall'][2], step)
                args.writer.add_scalar('recall/label3', bin_test['None/recall'][3], step)

                args.writer.add_scalar('f1/label0', bin_test['None/f1'][0], step)
                args.writer.add_scalar('f1/label1', bin_test['None/f1'][1], step)
                args.writer.add_scalar('f1/label2', bin_test['None/f1'][2], step)
                args.writer.add_scalar('f1/label3', bin_test['None/f1'][3], step)

                wandb.log({"test/loss": test_loss,
                           "test/acc@1": top1,
                           "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)
        wandb.log({"result/test_acc@1": args.best_top1})
    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    finetune(args, labeled_loader, test_loader, student_model, criterion)
    return

def train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Total steps = {args.total_steps}")

    predictions_file = os.path.dirname(args.stats_dir)
    predictions_file = os.path.join(predictions_file, args.name.replace(' ', '_') + '.json')
    all_train_predictions = []
    all_train_actual_predictions = []
    all_unlabeled_predictions = []
    all_pseudo_labels = []

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            text_l, segment_l, mask_l, _, _, _, tgt_l = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            text_l, segment_l, mask_l, _, _, _, tgt_l = labeled_iter.next()

        try:
            text_u_soft, segment_u_soft, mask_u_soft, text_u_hard, segment_u_hard, mask_u_hard, _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            text_u_soft, segment_u_soft, mask_u_soft, text_u_hard, segment_u_hard, mask_u_hard, _ = unlabeled_iter.next()

        data_time.update(time.time() - end)

        texts = torch.cat((text_l, text_u_soft, text_u_hard)).to(args.device)
        segments = torch.cat((segment_l, segment_u_soft, segment_u_hard)).to(args.device)
        masks = torch.cat((mask_l, mask_u_soft, mask_u_hard)).to(args.device)

        s_texts = torch.cat((text_l, text_u_soft)).to(args.device)
        s_segments = torch.cat((segment_l, segment_u_soft)).to(args.device)
        s_masks = torch.cat((mask_l, mask_u_soft)).to(args.device)


        with amp.autocast(enabled=args.amp):
            batch_size = text_l.shape[0]

            logits = teacher_model(texts, masks, segments)
            t_logits_l = logits[:batch_size]
            t_logits_u_soft, t_logits_u_hard = logits[batch_size:].chunk(2)
            del logits

            targets = tgt_l.to(args.device)

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_u_soft.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_u_hard, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_logits = student_model(s_texts, s_masks, s_segments)

            # if args.debug and step % 100 == 0:
            #     logger.info(f'\ns_texts shape: {s_texts.shape} s_masks shape: {s_masks.shape} s_segments shape: {s_segments.shape}')
            #     logger.info(f'logits shape: {s_logits.shape} targets shape: {targets.shape}')
            #     logger.info(s_logits)
            #     logger.info(targets)

            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            if all_train_predictions == []:
                all_train_predictions = [s_logits_l.argmax(axis=1).tolist()]
                all_train_actual_predictions = [targets.tolist()]
                all_unlabeled_predictions = [s_logits_us.argmax(axis=1).tolist()]
                all_pseudo_labels = [hard_pseudo_label.tolist()]
            else:
                all_train_predictions = all_train_predictions.append(s_logits_l.argmax(axis=1).tolist())
                all_train_actual_predictions = all_train_actual_predictions.append(targets.tolist())
                all_unlabeled_predictions = all_unlabeled_predictions.append(s_logits_us.argmax(axis=1).tolist())
                all_pseudo_labels = all_pseudo_labels.append(hard_pseudo_label.tolist())

            if args.debug_p:
                logger.info("\n")
                logger.info("S_LOGITS_L")
                logger.info(s_logits_l.argmax(axis=1).tolist())
                logger.info(s_logits_l)
                logger.info(all_train_predictions)
                logger.info("=========")
                logger.info("TARGETS")
                logger.info(targets.tolist())
                logger.info(targets)
                logger.info(all_train_actual_predictions)
                logger.info("S_LOGITS_US")
                logger.info("=========")
                logger.info(s_logits_us.argmax(axis=1).tolist())
                logger.info(s_logits_us)
                logger.info("HARD_PSEUDO_LABELS")
                logger.info("=========")
                logger.info(hard_pseudo_label.tolist())
                logger.info(hard_pseudo_label)

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(text_l.to(args.device), mask_l.to(args.device), segment_l.to(args.device))
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            # dot_product = s_loss_l_new - s_loss_l_old
            # test
            dot_product = s_loss_l_old - s_loss_l_new
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product
            _, hard_pseudo_label = torch.max(t_logits_u_soft.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_u_soft, hard_pseudo_label)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                # save predictions to json file
                logger.info("\n")
                logger.info(all_train_predictions)
                logger.info(all_train_actual_predictions)

                try:
                    with open(predictions_file, "r") as jsonFile:
                        data = json.load(jsonFile)
                except:
                    data = {}

                data = compute_data(args, data, all_train_predictions, all_train_actual_predictions, all_unlabeled_predictions, all_pseudo_labels)

                if args.debug_f:
                    logger.info("\nLogging predictions to file: \n")
                    logger.info(data)

                with open(predictions_file, "w") as jsonFile:
                    json.dump(data, jsonFile)

                all_train_predictions = []
                all_train_actual_predictions = []
                all_unlabeled_predictions = []
                all_pseudo_labels = []

                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                wandb.log({"train/1.s_loss": s_losses.avg,
                           "train/2.t_loss": t_losses.avg,
                           "train/3.t_labeled": t_losses_l.avg,
                           "train/4.t_unlabeled": t_losses_u.avg,
                           "train/5.t_mpl": t_losses_mpl.avg,
                           "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model

                train_loss, top1train, top5train, bin_train = evaluate(args, labeled_loader, test_model, criterion, "train")
                plot_metrics(args, train_loss, top1train, top5train, bin_train, step, "train")

                test_loss, top1, top5, bin_test = evaluate(args, test_loader, test_model, criterion, "test")
                plot_metrics(args, test_loss, top1, top5, bin_test, step, "test")

                wandb.log({"test/loss": test_loss,
                           "test/acc@1": top1,
                           "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)
        wandb.log({"result/test_acc@1": args.best_top1})
    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(args, labeled_loader, test_loader, student_model, criterion)
    return


def compute_data(args, data, all_train_predictions, all_train_actual_predictions, all_unlabeled_predictions, all_pseudo_labels):
    if args.debug_f:
        logger.info("data while adding train predictions")
    if data.get(TRAIN_PREDICTIONS, []) == []:
        if args.debug_f:
            logger.info(data.get(TRAIN_PREDICTIONS, []))
        data[TRAIN_PREDICTIONS] = all_train_predictions
    else:
        if args.debug_f:
            logger.info(data.get(TRAIN_PREDICTIONS, []))
        data[TRAIN_PREDICTIONS].extend(all_train_predictions)
    if args.debug_f:
        logger.info("data after adding train predictions")
        logger.info(data)
    if data.get(TRAIN_ACTUAL_PREDICTIONS, []) == []:
        data[TRAIN_ACTUAL_PREDICTIONS] = all_train_actual_predictions
    else:
        data[TRAIN_ACTUAL_PREDICTIONS].extend(all_train_actual_predictions)
    if data.get(UNLABELED_PREDICTIONS, []) == []:
        data[UNLABELED_PREDICTIONS] = all_unlabeled_predictions
    else:
        data[UNLABELED_PREDICTIONS].extend(all_unlabeled_predictions)
    if data.get(PSEUDO_LABELS, []) == []:
        data[PSEUDO_LABELS] = all_pseudo_labels
    else:
        data[PSEUDO_LABELS].extend(all_pseudo_labels)
    return data

def plot_metrics(args, test_loss, top1, top5, bin_test, step, evaluation_name):
    args.writer.add_scalar(evaluation_name + "/loss", test_loss, args.num_eval)
    args.writer.add_scalar(evaluation_name + "/acc@1", top1, args.num_eval)
    args.writer.add_scalar(evaluation_name + "/acc@5", top5, args.num_eval)

    args.writer.add_scalar(evaluation_name + '_precision/label0', bin_test['None/precision'][0], step)
    args.writer.add_scalar(evaluation_name + '_precision/label1', bin_test['None/precision'][1], step)
    args.writer.add_scalar(evaluation_name + '_precision/label2', bin_test['None/precision'][2], step)
    args.writer.add_scalar(evaluation_name + '_precision/label3', bin_test['None/precision'][3], step)

    args.writer.add_scalar(evaluation_name + '_recall/label0', bin_test['None/recall'][0], step)
    args.writer.add_scalar(evaluation_name + '_recall/label1', bin_test['None/recall'][1], step)
    args.writer.add_scalar(evaluation_name + '_recall/label2', bin_test['None/recall'][2], step)
    args.writer.add_scalar(evaluation_name + '_recall/label3', bin_test['None/recall'][3], step)

    args.writer.add_scalar(evaluation_name + '_f1/label0', bin_test['None/f1'][0], step)
    args.writer.add_scalar(evaluation_name + '_f1/label1', bin_test['None/f1'][1], step)
    args.writer.add_scalar(evaluation_name + '_f1/label2', bin_test['None/f1'][2], step)
    args.writer.add_scalar(evaluation_name + '_f1/label3', bin_test['None/f1'][3], step)

def evaluate(args, test_loader, model, criterion, evaluation_name):
    outputs = []
    targets = []

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, data in enumerate(test_iter):
            (text_x, segment_x, mask_x, text_y, segment_y, mask_y, tgt_x) = data
            data_time.update(time.time() - end)

            text_x, segment_x, mask_x, tgt_x = text_x.to(args.device), segment_x.to(args.device), mask_x.to(args.device), tgt_x.to(args.device)

            batch_size = text_x.shape[0]
            with amp.autocast(enabled=args.amp):
                logits_x = model(text_x, mask_x, segment_x)
                loss = criterion(logits_x, tgt_x)

                outputs.append(logits_x)
                targets.append(tgt_x)

            acc1, acc5 = accuracy(logits_x, tgt_x, (1, 4))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                evaluation_name + f" Evaluation: {step + 1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        b_metrics = all_metrics(outputs, targets)

        stats = {
            'outputs': outputs.tolist(),
            'preds': outputs.argmax(axis=1).tolist(),
            'targets': targets.tolist(),
            'scores': b_metrics,
        }
        stats_file = os.path.dirname(args.stats_dir)
        stats_file = os.path.join(stats_file, 'stats_file.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f)

        test_iter.close()
        return losses.avg, top1.avg, top5.avg, b_metrics


def finetune(args, train_loader, test_loader, model, criterion):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader) * args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch + 1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)
            wandb.log({"finetune/train_loss": losses.avg,
                       "finetune/test_loss": test_loss,
                       "finetune/acc@1": top1,
                       "finetune/acc@5": top5})

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("result/finetune_acc@1", args.best_top1)
            wandb.log({"result/fintune_acc@1": args.best_top1})
    return


def get_optimizer(model, args, lr):
    if args.optimizer == 'adam':
        return get_mmbt_optimizer(model, args, lr)
    if args.optimizer == 'sgd':
        my_momentum = 0.9  # 0.9
        print(f'Momentum set to {my_momentum}, nesterov set to {args.nesterov}')
        return optim.SGD(model.parameters(), lr=lr, momentum=my_momentum, nesterov=args.nesterov)
    raise ValueError('Invalid optimizer argument')


def get_mmbt_optimizer(model, args, lr):
    if args.model in ["bert"]:
        total_steps = (
                args.k_img
                / args.batch_size
                / args.gradient_accumulation_steps
                * args.epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0, },
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=lr,
            warmup=args.warmup,
            t_total=total_steps + 1,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        return get_mmbt_scheduler(optimizer, args)
    if args.scheduler == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
    raise ValueError('Invalid scheduler argument')


def get_mmbt_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, verbose=True, factor=args.lr_factor, min_lr=0.000001
    )


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)

    args.iteration = args.k_img // args.batch_size // args.world_size
    args.total_steps = args.epochs * args.iteration // args.gradient_accumulation_steps

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")
        wandb.init(name=args.name, project='MPL', config=args)

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # need to have args.<> ce accepta bertclf ca params, de luat din Doina cu valori default si lasat asa
    teacher_model = create_model(args)
    student_model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)
    avg_student_model = None
    if args.ema > 0:
        # args, model, args.ema_decay, device
        avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_text_optimizer = get_optimizer(teacher_model, args, args.teacher_lr)
    s_text_optimizer = get_optimizer(student_model, args, args.student_lr)

    t_text_scheduler = get_scheduler(t_text_optimizer, args)
    s_text_scheduler = get_scheduler(s_text_optimizer, args)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_text_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_text_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_text_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_text_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    labeled_loader, unlabeled_loader, valid_loader, test_loader = get_data_loaders(args)

    # add arg to check if I want to override/write test loader with validation loader
    if args.validation:
        test_loader = valid_loader

    if args.finetune:
        del t_scaler, t_text_scheduler, t_text_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_text_scheduler, s_text_optimizer
        finetune(args, labeled_loader, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_text_scheduler, t_text_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_text_scheduler, s_text_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return


    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_text_optimizer, s_text_optimizer, t_text_scheduler, s_text_scheduler, t_scaler, s_scaler)
    return


if __name__ == '__main__':
    main()
