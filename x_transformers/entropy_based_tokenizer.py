from itertools import zip_longest

import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from x_transformers.x_transformers import Decoder, TransformerWrapper

import einx
from einops import repeat, rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# entropy based tokenizer applied in byte-latent transformer paper
# they use a simple entropy threshold for segmenting a string into variable sized tokens

# https://arxiv.org/abs/2412.09871

class EntropyBasedTokenizer(Module):
    def __init__(
        self,
        decoder: TransformerWrapper,
        entropy_threshold: float
    ):
        super().__init__()
        assert isinstance(decoder.attn_layers, Decoder)

        self.decoder = decoder
        self.entropy_threshold = entropy_threshold

    @torch.no_grad()
    def forward(
        self,
        seq,
        lens = None, # Int['b']
        return_segmented_seq = False
    ):
        self.decoder.eval()

        is_var_length = exists(lens)
        batch, seq_len, device = *seq.shape, seq.device

        arange = torch.arange(seq_len, device = device)

        # forward through a small trained decoder and get the entropies of the logits

        _, intermediates = self.decoder(seq, return_logit_entropies = True)

        entropies = intermediates.logit_entropies

        # get length mask for boundaries

        mask = tensor(True, device = device)

        if is_var_length:
            mask = einx.less('n, b -> b n', arange, lens)

        # the mask for tokens that were of a sufficient surprise level

        over_thres_mask = (entropies >= self.entropy_threshold) & mask

        # needed for selecting out indices at entropy threshold mask

        arange_plus_one = arange + 1
        arange_plus_one = repeat(arange_plus_one, 'n -> b n', b = batch)

        # get a tensor of Int['b num_tokens'] with the token lengths, zero padded

        boundaries = over_thres_mask.clone()

        # set the boundary of the last token

        # if `lens` not given, assume always last token
        # but if `lens` were given, then properly set the index

        if not is_var_length:
            boundaries[..., -1] = True
        else:
            scatter_indices = rearrange(lens - 1, 'b -> b 1')
            boundaries.scatter_(-1, scatter_indices, True)

        num_tokens = boundaries.sum(dim = -1) # number of tokens

        indices = arange_plus_one[boundaries].split(num_tokens.tolist())

        # get the token lengths

        token_lengths = []

        for one_indices in indices:
            padded_indices = F.pad(one_indices, (1, 0), value = 0.)
            one_token_lengths = padded_indices[1:] - padded_indices[:-1]

            token_lengths.append(one_token_lengths)

        token_lengths = pad_sequence(token_lengths, batch_first = True)

        # early return

        if not return_segmented_seq:
            return token_lengths

        # segment the sequence based on the token lengths

        lens = default(lens, (None,))
        segmented_seq = []

        for one_seq, one_len, one_token_length in zip_longest(seq, lens, token_lengths):

            if exists(one_len):
                one_seq = one_seq[:one_len]

            one_token_length = one_token_length[one_token_length > 0]

            splitted_seq = one_seq.split(one_token_length.tolist())
            segmented_seq.append(splitted_seq)

        return segmented_seq
