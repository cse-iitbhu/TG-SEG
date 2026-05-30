import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=256, freeze=True):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.resizer = nn.Linear(768, hidden_dim)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, captions, device):
        
        tokens = self.tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        out = self.model(**tokens)
        word_feats = self.resizer(out.last_hidden_state)
        sent_feats = self.resizer(out.pooler_output)
        attn_mask = tokens.attention_mask == 0

        return word_feats, sent_feats, attn_mask
