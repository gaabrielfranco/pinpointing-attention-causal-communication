"""
Modified from the original implementation (Automatic_Circuits_Script.ipynb): 
https://cmathw.itch.io/identifying-a-preliminary-circuit-for-predicting-gendered-pronouns-in-gpt-2-smal
"""
import torch

class GenderedPronoun:
    def __init__(self, model, model_family, device="cpu", prepend_bos=False):
        if model_family == "gemma" and not prepend_bos:
            raise Exception(f"model_family={model_family} requires prepend_bos=True")
                
        self.templates = [
            "So {name} is a really great friend, isn't",
            "So {name} is such a good cook, isn't",
            "So {name} is a very good athlete, isn't",
            "So {name} is a really nice person, isn't",
            "So {name} is such a funny person, isn't"
            ]

        self.male_names = [
            "John",
            "David",
            "Mark",
            "Paul",
            "Ryan",
            "Gary",
            "Jack",
            "Sean",
            "Carl",
            "Joe",    
        ]

        self.female_names = [
            "Mary",
            "Lisa",
            "Anna",
            "Sarah",
            "Amy",
            "Carol",
            "Karen",
            "Susan",
            "Julie",
            "Judy"
        ]

        self.sentences = []
        self.answers = []
        self.wrongs = []

        self.responses = [' he', ' she']

        count = 0

        for name in self.male_names + self.female_names:
            for template in self.templates:
                cur_sentence = template.format(name = name)
                self.sentences.append(cur_sentence)

        batch_size = len(self.sentences)

        count = 0

        for _ in range(batch_size):
            if count < (0.5 * len(self.sentences)):
                self.answers.append(self.responses[0])
                self.wrongs.append(self.responses[1])
                count += 1
            else:
                self.answers.append(self.responses[1])
                self.wrongs.append(self.responses[0])

        self.tokens = model.to_tokens(self.sentences, prepend_bos = prepend_bos).to(device)
        self.toks = self.tokens # To also follow the IOIDataset   
        self.answers = torch.tensor(model.tokenizer(self.answers, add_special_tokens=False)["input_ids"]).squeeze().to(device)
        self.wrongs = torch.tensor(model.tokenizer(self.wrongs, add_special_tokens=False)["input_ids"]).squeeze().to(device)

        if not prepend_bos and model_family in ["gpt2", "pythia"]:
            self.word_idx = {
                "end": torch.full((batch_size, ), self.tokens.shape[1]-1, dtype=int), # sentences have all the same size
                "end-1": torch.full((batch_size, ), self.tokens.shape[1]-2, dtype=int), # sentences have all the same size
                "punct": torch.full((batch_size, ), self.tokens.shape[1]-3, dtype=int), # sentences have all the same size
                "noun": torch.full((batch_size, ), self.tokens.shape[1]-4, dtype=int), # sentences have all the same size
                "adj": torch.full((batch_size, ), self.tokens.shape[1]-5, dtype=int), # sentences have all the same size
                "I2": torch.full((batch_size, ), self.tokens.shape[1]-6, dtype=int), # sentences have all the same size
                "I1": torch.full((batch_size, ), self.tokens.shape[1]-7, dtype=int), # sentences have all the same size
                "is": torch.full((batch_size, ), self.tokens.shape[1]-8, dtype=int), # sentences have all the same size
                "name": torch.full((batch_size, ), self.tokens.shape[1]-9, dtype=int), # sentences have all the same size
                # Case for only we prepend_bos, then "So" is not the start
                #"So": torch.full((batch_size, ), self.tokens.shape[1]-10, dtype=int), # sentences have all the same size
                "starts": torch.zeros(batch_size, dtype=int),
            }
        elif prepend_bos and model_family in ["gpt2", "pythia"]:
            self.word_idx = {
                "end": torch.full((batch_size, ), self.tokens.shape[1]-1, dtype=int), # sentences have all the same size
                "end-1": torch.full((batch_size, ), self.tokens.shape[1]-2, dtype=int), # sentences have all the same size
                "punct": torch.full((batch_size, ), self.tokens.shape[1]-3, dtype=int), # sentences have all the same size
                "noun": torch.full((batch_size, ), self.tokens.shape[1]-4, dtype=int), # sentences have all the same size
                "adj": torch.full((batch_size, ), self.tokens.shape[1]-5, dtype=int), # sentences have all the same size
                "I2": torch.full((batch_size, ), self.tokens.shape[1]-6, dtype=int), # sentences have all the same size
                "I1": torch.full((batch_size, ), self.tokens.shape[1]-7, dtype=int), # sentences have all the same size
                "is": torch.full((batch_size, ), self.tokens.shape[1]-8, dtype=int), # sentences have all the same size
                "name": torch.full((batch_size, ), self.tokens.shape[1]-9, dtype=int), # sentences have all the same size
                "So": torch.full((batch_size, ), self.tokens.shape[1]-10, dtype=int), # sentences have all the same size
                "starts": torch.zeros(batch_size, dtype=int),
            }
        elif model_family in ["gemma"]: # Gemma assumes BOS always
            self.word_idx = {
                "end": torch.full((batch_size, ), self.tokens.shape[1]-1, dtype=int), # sentences have all the same size
                "end-1": torch.full((batch_size, ), self.tokens.shape[1]-2, dtype=int), # sentences have all the same size
                "end-2": torch.full((batch_size, ), self.tokens.shape[1]-3, dtype=int), # sentences have all the same size
                "punct": torch.full((batch_size, ), self.tokens.shape[1]-4, dtype=int), # sentences have all the same size
                "noun": torch.full((batch_size, ), self.tokens.shape[1]-5, dtype=int), # sentences have all the same size
                "adj": torch.full((batch_size, ), self.tokens.shape[1]-6, dtype=int), # sentences have all the same size
                "I2": torch.full((batch_size, ), self.tokens.shape[1]-7, dtype=int), # sentences have all the same size
                "I1": torch.full((batch_size, ), self.tokens.shape[1]-8, dtype=int), # sentences have all the same size
                "is": torch.full((batch_size, ), self.tokens.shape[1]-9, dtype=int), # sentences have all the same size
                "name": torch.full((batch_size, ), self.tokens.shape[1]-10, dtype=int), # sentences have all the same size
                "So": torch.full((batch_size, ), self.tokens.shape[1]-11, dtype=int), # sentences have all the same size
                "starts": torch.zeros(batch_size, dtype=int),
            }

        for key in self.word_idx:
            self.word_idx[key] = self.word_idx[key].to(device)

    def __len__(self):
        return len(self.sentences)