"""
Modified from the original implementation: https://github.com/hannamw/gpt2-greater-than/blob/main/dataset.py
"""

import random
from typing import List
from pathlib import Path
import torch
from transformers import PreTrainedTokenizer

## Source: original paper https://github.com/hannamw/gpt2-greater-than/blob/main/cache/potential_nouns.txt
NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal", "assaults", "assessment", "attack", "attempts", 
    "campaign", "captivity", "case", "challenge", "chaos", "clash", "collaboration", "coma", "competition", 
    "confrontation", "consequence", "conspiracy", "construction", "consultation", "contact", "contract", 
    "convention", "cooperation", "custody", "deal", "decline", "decrease", "demonstrations", "development", 
    "disagreement", "disorder", "dispute", "domination", "dynasty", "effect", "effort", "employment", "endeavor", 
    "engagement", "epidemic", "evaluation", "exchange", "existence", "expansion", "expedition", "experiments", 
    "fall", "fame", "flights", "friendship", "growth", "hardship", "hostility", "illness", "impact", "imprisonment", 
    "improvement", "incarceration", "increase", "insurgency", "invasion", "investigation", "journey", "kingdom", 
    "marriage", "modernization", "negotiation", "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague", "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", "reign", "relationship", "retaliation", "riot", "rise", 
    "rivalry", "romance", "rule", "sanctions", "shift", "siege", "slump", "stature", "stint", "strikes", "study", 
    "test", "testing", "tests", "therapy", "tour", "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work"
]

def generate_real_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {year} to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def real_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 YY to the year XX2".split()
    if eos:
        sentence = ["<|endoftext|>"] + sentence
    return sentence


def generate_bad_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {century}01 to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def bad_sentence_prompt(eos: bool = False) -> List[str]:
    sentence = f"The NOUN lasted from the year XX1 01 to the year XX2".split()
    if eos:
        sentence = ["<|endoftext|>"] + sentence
    return sentence


def is_valid_year(year: str, tokenizer) -> bool:
    _year = " " + year
    token = tokenizer(_year, add_special_tokens=False)["input_ids"]
    detok = tokenizer.convert_ids_to_tokens(token)
    return len(detok) == 2 and len(detok[1]) == 2


class YearDataset:
    years_to_sample_from: torch.Tensor
    N: int
    ordered: bool
    eos: bool

    nouns: List[str]
    years: torch.Tensor
    years_YY: torch.Tensor
    good_sentences: List[str]
    bad_sentences: List[str]
    good_toks: torch.Tensor
    bad_toks: torch.Tensor
    good_prompt: List[str]
    bad_prompt: List[str]
    good_mask: torch.Tensor
    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        years_to_sample_from,
        N: int,
        tokenizer: PreTrainedTokenizer,
        balanced: bool = True,
        eos: bool = False,
        device: str = "cpu",
        random_seed: int = 0,
    ):
        # Fixing seed for reproducibility
        random.seed(random_seed)

        self.years_to_sample_from = years_to_sample_from
        self.N = N
        self.eos=eos

        nouns = []

        # Filtering nouns: only single token ones
        for noun in NOUNS:
            if len(tokenizer.encode(f" {noun}", add_special_tokens=False)) == 1:
                nouns.append(noun)

        if isinstance(nouns, str):
            noun_list = [nouns]
        elif isinstance(nouns, list):
            noun_list = nouns
        elif isinstance(nouns, Path):
            with open(nouns, "r") as f:
                noun_list = [line.strip() for line in f]
        else:
            raise ValueError(f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}")

        self.nouns = random.choices(noun_list, k=N)

        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(N):
                sample_pool = self.years_to_sample_from[years_to_sample_from_YY == current_year]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            self.years = torch.tensor(years)
        else:
            self.years = torch.tensor(self.years_to_sample_from[torch.randint(0, len(self.years_to_sample_from), (N,))])

        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        self.good_sentences = [
            generate_real_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]
        self.bad_sentences = [
            generate_bad_sentence(noun, int(year.item()), eos=eos) for noun, year in zip(self.nouns, self.years)
        ]

        self.good_prompt = real_sentence_prompt(eos=eos)
        self.bad_prompt = bad_sentence_prompt(eos=eos)

        good_tokenized = tokenizer(self.good_sentences, return_tensors="pt", add_special_tokens=False)
        self.good_toks, good_attn = good_tokenized["input_ids"], good_tokenized["attention_mask"]
        assert torch.all(good_attn == 1)
        bad_tokenized = tokenizer(self.bad_sentences, return_tensors="pt", add_special_tokens=False)
        self.bad_toks, bad_attn = bad_tokenized["input_ids"], bad_tokenized["attention_mask"]
        assert torch.all(bad_attn == 1)

        # there's a better way to do this
        _good_logits_masks = []
        for year in self.years_YY:
            logits_mask = torch.arange(100)
            _good_logits_masks.append(logits_mask > year)
        self.good_mask = torch.stack(_good_logits_masks)

        self.good_toks = self.good_toks.to(device)
        self.bad_toks = self.bad_toks.to(device)
        self.good_mask = self.good_mask.to(device)

        # Creating a word_idx to make it easier.
        # All sentences have the same length for this dataset
        # Also, they always have the same structure
        #'The {NOUN} lasted from the year {XX}{YY} to the year {XX}'

        self.word_idx = {}
        END_POS = self.good_toks.size(-1) - 1
        for i in range(len(self.good_prompt)):
            gram_role = self.good_prompt[i]
            if i == 0:
                self.word_idx["starts"] = torch.zeros(self.N, dtype=int).to(device)
            elif i == END_POS:
                self.word_idx["end"] = torch.full((self.N, ), END_POS, dtype=int).to(device) # same as XX
            else:
                if gram_role in self.word_idx:
                    self.word_idx[f"{gram_role} (1)"] = torch.full((self.N, ), i, dtype=int).to(device)
                else:
                    self.word_idx[gram_role] = torch.full((self.N, ), i, dtype=int).to(device)

        self.possible_targets = [f"{i}{j}" for i in range(10) for j in range(10)]
        self.possible_targets_toks = tokenizer.convert_tokens_to_ids(self.possible_targets)
        self.YY_toks = [self.good_toks[i, x].item() for i, x in enumerate(self.word_idx["YY"])]

        self.toks = self.good_toks # Just to make it easy

    def __len__(self):
        return self.N

def get_valid_years(
    tokenizer,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years, add_special_tokens=False)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor([(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized])
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]