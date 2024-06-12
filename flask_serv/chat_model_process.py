import torch
from torch import nn
from torch.nn.functional import pad
import math

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

EMBED_DIM = 512
NUM_HEAD = 8
NUM_LAYERS = 6

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(r"LangLern\PythonSimpleServer\tokenizer_music.json")
vocab_size = len(tokenizer.get_vocab())

#  MODEL INITIALIZING
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(TransformerModel, self).__init__()
        self.dropout = nn.Dropout()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, dropout=0.2)
        self.max_len = max_len

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=NUM_HEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dropout=0.2,
            batch_first=True,
            norm_first=True
            )

        self.linear = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def create_mask(self, src, tgt, pad_idx):
            src_seq_len = src.shape[1]
            tgt_seq_len = tgt.shape[1]

            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

            src_padding_mask = (src == pad_idx)
            tgt_padding_mask = (tgt == pad_idx)
            return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, input, target):
        src_embed = self.pos_encode(self.token_embedding(input))
        tgt_embed = self.pos_encode(self.token_embedding(target))

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(input, target, 0)

        # output = self.transformer(src_embed, tgt_embed, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, None)
        output = self.transformer(src_embed, tgt_embed, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, None)
        output = self.linear(output)

        return output
    
model = torch.load(r"LangLern\PythonSimpleServer\model_music.pt")

def greedy_decode(model, src):
  src = src.to(device)
  tgt = torch.ones(1, 1).fill_(1).type(torch.long).to(device)

  for i in range(model.max_len):
        out = model(src, tgt)

        _, next_word = torch.max(out, dim=2, keepdim=True)
        next_word = next_word.squeeze()
        next_word = next_word.item() if not next_word.size() else next_word[-1].item()

        tgt = torch.cat([tgt,
                        torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=1)

        if next_word == 2:
            break

  return tgt

sent_preprocess = lambda x: [1] + tokenizer.encode(x).ids + [2]

def eval(model, text):
    text = sent_preprocess(text)
    text = pad(torch.tensor(text), (0, model.max_len-len(text))).unsqueeze(0).to(device)
    # print(text)
    model.eval()
    # output = model(text, text)
    output = greedy_decode(model, text)

    return tokenizer.decode(output.squeeze().tolist())