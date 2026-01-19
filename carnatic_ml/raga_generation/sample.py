import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from raga_generation.tokenizer import SwaraTokenizer
from raga_generation.utils import RagaGenerator

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def sample(seed_sequence, length=32, temperature=1.0, allowed_swaras=None):
    model_path = os.path.join(MODEL_DIR, 'small_raga_gen.h5')
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model not found. Train first.')
    model = load_model(model_path)
    tokenizer = SwaraTokenizer.load(tokenizer_path)

    seq = seed_sequence[:]
    # prepare allowed id mask if provided
    allowed_ids = None
    if allowed_swaras:
        allowed_tokens = set([s.upper() for s in allowed_swaras])
        allowed_ids = set()
        for tok, idx in tokenizer.token_to_id.items():
            if tok.upper() in allowed_tokens:
                allowed_ids.add(idx)

    for _ in range(length):
        x = np.array([seq[-6:]])
        preds = model.predict(x, verbose=0)[0]
        # apply temperature
        preds = np.log(preds + 1e-9) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # if allowed_ids provided, mask out disallowed tokens
        if allowed_ids is not None:
            mask = np.zeros_like(preds)
            for i in allowed_ids:
                if 0 <= i < len(mask):
                    mask[i] = preds[i]
            total = mask.sum()
            if total > 0:
                preds = mask / total
            # else fall back to unmasked preds

        next_id = np.random.choice(len(preds), p=preds)
        seq.append(int(next_id))

    return tokenizer.decode(seq)


if __name__ == '__main__':
    rg = RagaGenerator()
    # use a phrase from the raga as seed if available
    rd = rg.get_raga_definition('mohanam')
    seed = rd['phrases'][0] if rd and rd.get('phrases') else ['S','R2','G3']
    from raga_generation.tokenizer import SwaraTokenizer
    tok = SwaraTokenizer()
    seed_ids = tok.encode(seed)
    allowed = rg.get_raga_scale('mohanam') or seed
    out = sample(seed_ids, length=32, temperature=0.8, allowed_swaras=allowed)
    print('Generated sequence:', out)
