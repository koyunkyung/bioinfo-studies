import json
from pathlib import Path
from typing import Dict, Optional

class SimpleGeneVocab:
    def __init__(self, token2idx: Dict[str, int], pad_token: str = "<pad>"):
        self.token2idx = token2idx
        self.idx2token = {v: k for k, v in token2idx.items()}
        self.pad_token = pad_token
        
    @classmethod
    def from_file(cls, file_path: str) -> 'SimpleGeneVocab':
        with open(file_path, 'r') as f:
            token2idx = json.load(f)
        return cls(token2idx)
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def __getitem__(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx.get(self.pad_token, 0))
    
    def get(self, token: str, default: Optional[int] = None) -> int:
        return self.token2idx.get(token, default if default is not None else self.token2idx[self.pad_token])