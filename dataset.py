from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    A custom Dataset class for handling bilingual translation data, used to prepare input-output pairs
    for training a sequence-to-sequence model (e.g., Transformer) in a supervised learning setup.

    Parameters:
        ds (list): A list of translation pairs, where each pair contains source and target language texts.
        tokenizer_src (object): Tokenizer for the source language.
        tokenizer_tgt (object): Tokenizer for the target language.
        src_lang (str): Source language identifier (e.g., "en" for English).
        tgt_lang (str): Target language identifier (e.g., "de" for German).
        seq_len (int): Maximum sequence length for inputs and outputs.

    Returns:
        Dictionary containing:
            encoder_input (Tensor): Source sequence tensor with [SOS] and [EOS] tokens. Shape: (seq_len,)
            decoder_input (Tensor): Target sequence tensor with [SOS] token. Shape: (seq_len,)
            encoder_mask (Tensor): Mask for encoder input. Shape: (1, 1, seq_len)
            decoder_mask (Tensor): Mask for decoder input with causal masking. Shape: (1, seq_len, seq_len)
            label (Tensor): Target sequence tensor with [EOS] token. Shape: (seq_len,)
            src_text (str): Original source text.
            tgt_text (str): Original target text.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        # Store the provided parameters
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len  # Maximum sequence length for padding and truncation

        # Define special tokens using the source tokenizer
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)  # Start of Sentence
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)  # End of Sentence
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)  # Padding token

    def __len__(self):
        """Returns the total number of translation pairs in the dataset."""
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        """
        Fetches a single data point from the dataset, processes it, and returns tensors for model input.

        Parameters:
            index (int): Index of the data pair to fetch.

        Returns:
            dict: Contains encoder/decoder inputs, masks, and labels.
        """
        # Step 1: Retrieve the source and target texts
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Step 2: Tokenize source and target texts to obtain token IDs
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Step 3: Calculate padding requirements for both source and target
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS

        # Raise an error if the sentence length exceeds maximum allowed sequence length
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Step 4: Create encoder input by concatenating [SOS], tokens, [EOS], and padding
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # Step 5: Create decoder input by concatenating [SOS], tokens, and padding
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Step 6: Create label tensor by appending [EOS] to target tokens and padding
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Step 7: Verify tensor sizes match the expected sequence length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Step 8: Return input-output tensors and masks
        return {
            "encoder_input": encoder_input,  # Encoder input tensor
            "decoder_input": decoder_input,  # Decoder input tensor
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # Encoder mask
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # Decoder mask
            "label": label,  # Ground truth labels for decoder output
            "src_text": src_text,  # Source text for reference
            "tgt_text": tgt_text   # Target text for reference
        }


def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for decoder inputs to prevent a position from attending to future tokens.

    Parameters:
        size (int): The length of the sequence (seq_len).

    Returns:
        Tensor: A mask tensor of shape (1, seq_len, seq_len), where True indicates allowed attention.
    """
    # Create an upper triangular matrix with 1s above the diagonal
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)

    # Return the mask where values are 0 (allowed) and 1 (blocked)
    return mask == 0
