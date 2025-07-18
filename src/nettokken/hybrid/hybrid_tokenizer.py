import os
import tempfile

import torch
from transformers import ByT5Tokenizer

from src.nettokken.hybrid.pcap_tokenizer import PCAPTokenizer


class HybridByT5PCAPTokenizer(ByT5Tokenizer):
    """
    A tokenizer that combines ByT5's byte-level tokenization with PCAP tokenization,
    supporting mixed inputs of text and PCAP data.
    """

    def __init__(
            self,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=0,
            additional_special_tokens=None,
            pcap_vocab_size=280,  # Increased for new special tokens
            **kwargs,
    ):
        # Initialize with special tokens for content boundaries
        special_tokens = [
            "<pcap_start>",
            "<pcap_end>",
            "<text_start>",
            "<text_end>",
            "<pcap_attachment>",
            "<packet_start>",
            "<packet_end>",
            "<flow_end>",
            "<field_sep>"
        ]

        # Add any user-provided special tokens
        if additional_special_tokens:
            additional_special_tokens.extend(special_tokens)
        else:
            additional_special_tokens = special_tokens

        # Add special tokens for link types
        known_link_types = [0, 1, 8, 9, 10, 101, 105, 113, 127]
        for link_type in known_link_types:
            additional_special_tokens.append(f"<link_type_{link_type}>")

        # Initialize the ByT5Tokenizer first
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # Initialize the PCAP tokenizer
        self.pcap_tokenizer = PCAPTokenizer(vocab_size=pcap_vocab_size)

        # Store the token IDs for easy access
        self.pcap_start_token_id = self.convert_tokens_to_ids("<pcap_start>")
        self.pcap_end_token_id = self.convert_tokens_to_ids("<pcap_end>")
        self.text_start_token_id = self.convert_tokens_to_ids("<text_start>")
        self.text_end_token_id = self.convert_tokens_to_ids("<text_end>")
        self.pcap_attachment_token_id = self.convert_tokens_to_ids("<pcap_attachment>")

        # --- THIS IS THE FIX ---
        # Add the attributes needed by get_2d_position_indices back
        self.packet_start_token_id = self.convert_tokens_to_ids("<packet_start>")
        self.packet_end_token_id = self.convert_tokens_to_ids("<packet_end>")
        self.flow_end_token_id = self.convert_tokens_to_ids("<flow_end>")
        self.field_sep_token_id = self.convert_tokens_to_ids("<field_sep>")

        # --- Create a unified mapping for ALL PCAP tokens ---
        self.pcap_to_byt5_token_map = {}

        # 1. Map special PCAP tokens to the ByT5 special tokens
        self.pcap_to_byt5_token_map[self.pcap_tokenizer.special_tokens['packet_start']] = self.packet_start_token_id
        self.pcap_to_byt5_token_map[self.pcap_tokenizer.special_tokens['packet_end']] = self.packet_end_token_id
        self.pcap_to_byt5_token_map[self.pcap_tokenizer.special_tokens['flow_end']] = self.flow_end_token_id
        self.pcap_to_byt5_token_map[self.pcap_tokenizer.special_tokens['field_sep']] = self.field_sep_token_id

        # 2. Add and map link type tokens
        for link_type in known_link_types:
            token_str = f"<link_type_{link_type}>"
            self.pcap_to_byt5_token_map[link_type] = self.convert_tokens_to_ids(token_str)

    def encode_mixed_input(self, text=None, pcap_bytes=None, pcap_file_path=None):
        """
        Encode mixed input of text and PCAP data.

        Args:
            text: Text content (optional)
            pcap_bytes: Raw PCAP bytes (optional)
            pcap_file_path: Path to a PCAP file (optional)

        Returns:
            Token IDs for the mixed input
        """
        token_ids = []

        # Handle text if provided
        if text is not None:
            token_ids.append(self.text_start_token_id)
            text_tokens = super().encode(text, add_special_tokens=False)
            token_ids.extend(text_tokens)
            token_ids.append(self.text_end_token_id)

        # Handle PCAP bytes if provided
        if pcap_bytes is not None:
            token_ids.append(self.pcap_start_token_id)
            pcap_tokens = self._tokenize_pcap_bytes(pcap_bytes)
            token_ids.extend(pcap_tokens)
            token_ids.append(self.pcap_end_token_id)

        # Handle PCAP file if provided
        if pcap_file_path:
            token_ids.append(self.pcap_attachment_token_id)
            pcap_tokens = self._tokenize_pcap_file(pcap_file_path)
            token_ids.extend(pcap_tokens)
            token_ids.append(self.pcap_end_token_id)

        return token_ids

    def _tokenize_pcap_bytes(self, pcap_bytes):
        """
        Tokenize raw PCAP bytes.

        Args:
            pcap_bytes: Raw PCAP data as bytes

        Returns:
            Tokenized representation of the PCAP data
        """
        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(pcap_bytes)

        # Tokenize the temporary file
        try:
            tokens = self._tokenize_pcap_file(temp_file_path)
        finally:
            os.unlink(temp_file_path)

        return tokens

    def _tokenize_pcap_file(self, pcap_file_path):
        """
        Tokenize a PCAP file using the simplified output from PCAPTokenizer.
        """
        try:
            tokenized_flows = self.pcap_tokenizer.tokenize_pcap(pcap_file_path)
            converted_tokens = []

            # The ByT5 byte offset is 3 (<pad>, </s>, <unk>)
            BYT5_BYTE_OFFSET = 3

            for flow_id, tokens in tokenized_flows.items():
                for token in tokens:
                    if token > 255:  # This is a special token (e.g., packet_start, field_sep)
                        # Find the corresponding ByT5 special token ID
                        # This requires building a reverse map from PCAP token ID to ByT5 ID in __init__
                        byt5_token_id = self.pcap_to_byt5_token_map.get(token, self.unk_token_id)
                        converted_tokens.append(byt5_token_id)
                    else:  # This is a raw byte value (0-255)
                        # Convert the byte to its corresponding ByT5 token ID
                        converted_tokens.append(token + BYT5_BYTE_OFFSET)

            return converted_tokens
        except Exception as e:
            import logging
            logging.error(f"Error tokenizing PCAP file {pcap_file_path}: {e}")
            return []

    def decode_mixed_input(self, token_ids):
        """
        Decode token IDs back to text and PCAP data using simplified logic.
        """
        result = {}
        i = 0

        while i < len(token_ids):
            if token_ids[i] == self.text_start_token_id:
                text_tokens = []
                i += 1
                while i < len(token_ids) and token_ids[i] != self.text_end_token_id:
                    text_tokens.append(token_ids[i])
                    i += 1
                if i < len(token_ids):
                    i += 1
                result['text'] = super().decode(text_tokens, skip_special_tokens=True)

            elif token_ids[i] == self.pcap_start_token_id or token_ids[i] == self.pcap_attachment_token_id:
                pcap_tokens = []
                i += 1
                while i < len(token_ids) and token_ids[i] != self.pcap_end_token_id:
                    pcap_tokens.append(token_ids[i])
                    i += 1
                if i < len(token_ids):
                    i += 1

                # Reverse mapping: from ByT5 ID back to PCAP's native tokens
                rev_map = {v: k for k, v in self.pcap_to_byt5_token_map.items()}
                pcap_native_tokens = []
                BYT5_BYTE_OFFSET = 3

                for token in pcap_tokens:
                    if token in rev_map:
                        pcap_native_tokens.append(rev_map[token])  # Convert special token
                    elif BYT5_BYTE_OFFSET <= token < BYT5_BYTE_OFFSET + 256:
                        pcap_native_tokens.append(token - BYT5_BYTE_OFFSET)  # Convert byte

                pcap_data = self.pcap_tokenizer.decode_flow(pcap_native_tokens)
                result['pcap_data'] = pcap_data
            else:
                i += 1

        return result

    # Add methods to handle the different input scenarios
    def tokenize_text_with_pcap(self, text, pcap_file_path):
        """Text with a PCAP file attachment"""
        if not os.path.exists(pcap_file_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file_path}")

        token_ids = []

        # Only add text tokens if there is actual text content
        if text:
            token_ids.append(self.text_start_token_id)
            text_tokens = super().encode(text, add_special_tokens=False)
            token_ids.extend(text_tokens)
            token_ids.append(self.text_end_token_id)

        # Add PCAP tokens
        token_ids.append(self.pcap_attachment_token_id)
        pcap_tokens = self._tokenize_pcap_file(pcap_file_path)
        token_ids.extend(pcap_tokens)
        token_ids.append(self.pcap_end_token_id)

        return token_ids

    def tokenize_text_followed_by_pcap(self, text, pcap_bytes):
        """Text followed by PCAP bytes"""
        return self.encode_mixed_input(text=text, pcap_bytes=pcap_bytes)

    def tokenize_pcap_followed_by_text(self, pcap_bytes, text):
        """PCAP bytes followed by text"""
        # Note: The order is preserved in the token sequence
        return self.encode_mixed_input(pcap_bytes=pcap_bytes, text=text)

    def get_2d_position_indices(self, input_ids, pcap_only=False):
        """
        Generate position indices for both row-wise and column-wise traversal.

        Args:
            input_ids: token IDs
            pcap_only: if True, only assign 2D positions to PCAP tokens

        Returns:
            position_indices: tensor of shape [batch_size, seq_len, 2]
                             where [:,:,0] contains row-wise positions
                             and [:,:,1] contains column-wise positions
        """
        if isinstance(input_ids, torch.Tensor):
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size = len(input_ids)
            seq_length = len(input_ids[0]) if batch_size > 0 else 0
            device = "cpu"

        # Initialize position indices
        position_indices = torch.zeros((batch_size, seq_length, 2), dtype=torch.long, device=device)

        for b in range(batch_size):
            tokens = input_ids[b] if isinstance(input_ids, list) else input_ids[b].tolist()

            # Initialize row-wise positions (standard sequential positions)
            row_positions = torch.arange(seq_length, device=device)
            position_indices[b, :, 0] = row_positions

            # For column-wise positions, we need to identify packet structure
            in_pcap = False
            packet_start_positions = []
            field_positions = {}  # Maps field index within packet to list of positions
            current_field_idx = 0

            for i, token_id in enumerate(tokens):
                token_id = token_id if isinstance(token_id, int) else token_id.item()

                # Check if we're entering PCAP section
                if token_id == self.pcap_attachment_token_id or token_id == self.pcap_start_token_id:
                    in_pcap = True
                    continue

                # Check if we're leaving PCAP section
                if token_id == self.pcap_end_token_id:
                    in_pcap = False
                    continue

                # Skip non-PCAP tokens if pcap_only is True
                if pcap_only and not in_pcap:
                    continue

                # Track PCAP structure
                if in_pcap:
                    if token_id == self.packet_start_token_id:
                        packet_start_positions.append(i)
                        current_field_idx = 0
                    elif token_id == self.field_sep_token_id:
                        current_field_idx += 1
                    else:
                        # Regular token within a field
                        if current_field_idx not in field_positions:
                            field_positions[current_field_idx] = []
                        field_positions[current_field_idx].append(i)

            # Assign column-wise positions based on field positions
            col_position = 0
            for field_idx in sorted(field_positions.keys()):
                for pos in field_positions[field_idx]:
                    position_indices[b, pos, 1] = col_position
                    col_position += 1

            # For non-PCAP tokens, use row-wise position as column-wise too
            if not pcap_only:
                for i in range(seq_length):
                    if position_indices[b, i, 1] == 0 and not in_pcap:
                        position_indices[b, i, 1] = position_indices[b, i, 0]

        return position_indices

    def tokenize_with_2d_positions(self, text=None, pcap_file_path=None):
        """
        Tokenize inputs and return token IDs with 2D position indices.

        Args:
            text: Text content (optional)
            pcap_file_path: Path to a PCAP file (optional)

        Returns:
            tuple: (token_ids, position_indices)
        """
        token_ids = self.tokenize_text_with_pcap(text, pcap_file_path)

        # Convert to tensor if needed
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Add batch dimension if needed
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # Generate 2D position indices
        position_indices = self.get_2d_position_indices(token_ids)

        return token_ids, position_indices