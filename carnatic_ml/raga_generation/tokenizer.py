import json
import os

class SwaraTokenizer:
    """Simple tokenizer that maps swara tokens (like 'S','R2') to integers and back."""
    def __init__(self, vocab=None):
        if vocab is None:
            # default swara vocabulary covering common variants
            vocab = ['S','R1','R2','R3','G1','G2','G3','M1','M2','P','D1','D2','D3','N1','N2','N3','REST']
        self.vocab = vocab
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, sequence):
        return [self.token_to_id.get(tok, self.token_to_id['REST']) for tok in sequence]

    def decode(self, ids):
        return [self.id_to_token.get(i, 'REST') for i in ids]

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
