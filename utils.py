import torch
import logging

def log(msg):
    print(msg)
    logging.info(msg)

def detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_tokenizer_for_embedding(args):
    # Placeholder for your tokenizer (customize if needed)
    return lambda text: len(text.split())
