import os
from collections import Counter

import pandas as pd

import torch
from pytorch_pretrained_bert import BertTokenizer

from data.vocab import Vocab

from torch.utils.data import DataLoader
from data.textDataset import TextDataset

TEXT_SIZE = 224


def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def get_labels_and_frequencies(path):
    df = pd.read_csv(path)
    label_freqs = Counter(df.Label)
    print(f'Labels frequencies: {label_freqs}')

    return list(label_freqs.keys()), label_freqs


def get_datasets(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, f"{args.train_file}.csv")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.num_classes = len(args.labels)

    labeled_dataset = TextDataset(
        os.path.join(args.data_path, f"{args.train_file}.csv"),
        tokenizer,
        vocab,
        args,
        text_aug1=args.text_soft_aug,
        text_aug2='none'
    )

    args.train_data_len = len(labeled_dataset)

    unlabeled_dataset = TextDataset(
        os.path.join(args.data_path, f"unlabeled.csv"),
        tokenizer,
        vocab,
        args,
        text_aug1=args.text_soft_aug,
        text_aug2=args.text_hard_aug
    )

    dev_dataset = TextDataset(
        os.path.join(args.data_path, f"dev.csv"),
        tokenizer,
        vocab,
        args,
        text_aug1='none',
        text_aug2='none'
    )

    test_dataset = TextDataset(
        os.path.join(args.data_path, f"test.csv"),
        tokenizer,
        vocab,
        args,
        text_aug1='none',
        text_aug2='none'
    )

    return labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset


def prepare_text_segment_mask(batch, index_text, index_seg):
    lens = [len(row[index_text]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens) if TEXT_SIZE is None else TEXT_SIZE

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    sentence_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        if length > max_seq_len:
            length = max_seq_len
        tokens, segment = input_row[index_text], input_row[index_seg]
        sentence_tensor[i_batch, :length] = tokens[:length]
        segment_tensor[i_batch, :length] = segment[:length]
        mask_tensor[i_batch, :length] = 1

    return sentence_tensor, segment_tensor, mask_tensor


def collate_function(batch):
    sentence_tensor1, segment_tensor1, mask_tensor1 = prepare_text_segment_mask(batch, 0, 1)
    sentence_tensor2, segment_tensor2, mask_tensor2 = prepare_text_segment_mask(batch, 2, 3)

    tgt_tensor = torch.cat([elem[4] for elem in batch]).long()

    return sentence_tensor1, segment_tensor1, mask_tensor1, sentence_tensor2, segment_tensor2, mask_tensor2, tgt_tensor


def get_data_loaders(args):
    labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset = get_datasets(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_function,
        drop_last=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        shuffle=True,
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        collate_fn=collate_function,
        drop_last=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * (args.mu + 1),
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_function,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * (args.mu + 1),
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_function,
    )

    return labeled_loader, unlabeled_loader, dev_loader, test_loader
