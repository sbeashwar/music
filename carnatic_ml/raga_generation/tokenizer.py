import json
import os

# Enharmonic equivalents: these pairs share the same pitch
# G1=R2 (semitone 2), G2=R3 (semitone 3), N1=D2 (semitone 9), N2=D3 (semitone 10)
ENHARMONIC_MAP = {
    'G1': 'R2',   # Shuddha Gandhara = Chatushruti Rishabha
    'G2': 'R3',   # Sadharana Gandhara = Shatshruti Rishabha
    'N1': 'D2',   # Shuddha Nishada = Chatushruti Dhaivata
    'N2': 'D3',   # Kaisiki Nishada = Shatshruti Dhaivata
}


def normalize_swara(swara: str) -> str:
    """Normalize a swara token to its canonical form.
    
    Maps enharmonic equivalents to a single canonical name:
      G1 -> R2, G2 -> R3, N1 -> D2, N2 -> D3
    This ensures the same pitch is always represented by the same token.
    """
    return ENHARMONIC_MAP.get(swara, swara)


class SwaraTokenizer:
    """Simple tokenizer that maps swara tokens (like 'S','R2') to integers and back.
    
    When normalize=True, enharmonic equivalents are collapsed to canonical forms
    (G1->R2, G2->R3, N1->D2, N2->D3) before encoding. This reduces the effective
    vocabulary from 17 to 13 pitch classes and helps the model learn pitch
    relationships without duplicates.
    """
    def __init__(self, vocab=None, normalize=False):
        if vocab is None:
            # default swara vocabulary covering common variants
            vocab = ['S','R1','R2','R3','G1','G2','G3','M1','M2','P','D1','D2','D3','N1','N2','N3','REST']
        self.vocab = vocab
        self.normalize = normalize
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, sequence):
        """Encode a sequence of swara tokens to integer IDs.
        
        If normalize=True, enharmonic equivalents are mapped to canonical forms
        before encoding (e.g., G1 -> R2's ID, N1 -> D2's ID).
        """
        result = []
        for tok in sequence:
            if self.normalize:
                tok = normalize_swara(tok)
            result.append(self.token_to_id.get(tok, self.token_to_id['REST']))
        return result

    def decode(self, ids):
        return [self.id_to_token.get(i, 'REST') for i in ids]

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'normalize': self.normalize}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Support both old format (plain list) and new format (dict with normalize)
        if isinstance(data, list):
            return cls(vocab=data, normalize=False)
        return cls(vocab=data.get('vocab', data), normalize=data.get('normalize', False))
