import random
import torch
from torch.utils.data import Dataset

import pandas as pd

class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, vocabulary, args, text_aug1='none', text_aug2='none'):
        self.data = pd.read_csv(csv_path)

        self.num_classes = self.data['Label'].unique().shape[0]
        print(f'Loaded {self.data.shape[0]} examples equally distributed from {self.num_classes} classes.')

        self.args = args
        self.tokenizer = tokenizer
        self.vocab = vocabulary
        self.max_seq_len = args.max_seq_len
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]
        self.text_aug1 = text_aug1
        self.text_aug2 = text_aug2

    def __len__(self):
        return self.data.shape[0]  # len(self.data)

    def get_sentence_and_segment(self, text):
        sentence = (
            self.text_start_token
            + self.tokenizer(text)[
                : (self.args.max_seq_len - 1)
            ]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        return sentence, segment

    def _get_augmented_sentence(self, row, text_aug):
        if text_aug == 'none':
            text = row['Text']
            return self.get_sentence_and_segment(text)

        elif text_aug == 'back_translate':
            text = random.choice(row['back_translate'])
            return self.get_sentence_and_segment(text)

        elif text_aug == 'eda':
            text = random.choice(row['eda'])
            return self.get_sentence_and_segment(text)

        else:
            raise NotImplementedError(f'{text_aug} is not a valid augmentation')

    def __getitem__(self, index):
        row = self.data.iloc[index]

        sentence1, segment1 = self._get_augmented_sentence(row, self.text_aug1)
        sentence2, segment2 = self._get_augmented_sentence(row, self.text_aug2)

        label_string = row['Label']
        label_int = self.args.labels.index(label_string)
        label = torch.LongTensor([label_int])

        return sentence1, segment1, sentence2, segment2, label

