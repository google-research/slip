from typing import Sequence, List

class _Vocab:
    # cf: https://github.com/nickbhat/mogwai/blob/main/mogwai/vocab.py

    def __init__(self):
        self.ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
        self.A2N = {a: n for n, a in enumerate(self.ALPHABET)}
        self.A2N["X"] = 20

    def convert_indices_to_tokens(self, indices: Sequence[int]) -> List[str]:
        return [self.ALPHABET[i] for i in indices]

    def convert_tokens_to_indices(self, tokens: Sequence[str], skip_unknown: bool = False) -> List[int]:
        if skip_unknown:
            return [self.A2N[token] for token in tokens if token in self.A2N]
        else:
            return [self.A2N.get(token, 20) for token in tokens]

    def tokenize(self, sequence: str) -> List[int]:
        return self.convert_tokens_to_indices(list(sequence))

    def __len__(self) -> int:
        return 20

Vocab = _Vocab()
