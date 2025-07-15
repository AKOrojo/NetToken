import os

from src.nettokken.hybrid.hybrid_tokenizer import HybridByT5PCAPTokenizer


# Helper class for terminal colors
class Colors:
    SPECIAL = '\033[93m'
    PACKET = '\033[96m'
    FIELD = '\033[92m'
    RESET = '\033[0m'


def main():
    """
    Main function to demonstrate the HybridByT5PCAPTokenizer.
    """
    print("--- Initializing HybridByT5PCAPTokenizer ---")

    tokenizer = HybridByT5PCAPTokenizer()
    print("Tokenizer initialized successfully.\n")

    pcap_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test.pcap')
    example_text = "Analyzing network traffic for potential threats."

    print(f"Text for tokenization: '{example_text}'")
    print(f"PCAP file for tokenization: '{pcap_file_path}'\n")

    # --- Tokenization ---
    print("--- Step 1: Tokenizing Text and PCAP Attachment ---")
    token_ids = tokenizer.tokenize_text_with_pcap(
        text=example_text,
        pcap_file_path=pcap_file_path
    )
    print(f"Generated {len(token_ids)} tokens in total.\n")

    # --- Step 2: Decoding Tokens to Human-Readable Format ---
    print("--- Step 2: Decoding Tokens to Human-Readable Strings (with Colors) ---")

    # Create a reverse map from token ID to string for easy lookup
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    # Define color mappings for special tokens
    color_map = {
        tokenizer.text_start_token_id: Colors.SPECIAL,
        tokenizer.text_end_token_id: Colors.SPECIAL,
        tokenizer.pcap_attachment_token_id: Colors.SPECIAL,
        tokenizer.pcap_end_token_id: Colors.SPECIAL,
        tokenizer.packet_start_token_id: Colors.PACKET,
        tokenizer.packet_end_token_id: Colors.PACKET,
        tokenizer.flow_end_token_id: Colors.PACKET,
        tokenizer.field_sep_token_id: Colors.FIELD,
    }

    max_display = 150
    formatted_tokens = []

    for i, token_id in enumerate(token_ids):
        if max_display <= i < len(token_ids) - max_display:
            if formatted_tokens[-1] != f"{Colors.RESET}...":
                formatted_tokens.append(f"{Colors.RESET}...")
            continue

        token_str = id_to_token.get(token_id, f"<id_{token_id}>")
        color = Colors.RESET

        if token_id in color_map:
            color = color_map[token_id]
        elif "link_type" in token_str:
            color = Colors.SPECIAL

        # For byte data the character representation
        if color == Colors.RESET:
            formatted_tokens.append(tokenizer.decode([token_id]))
        else:
            formatted_tokens.append(f"{color}{token_str}{Colors.RESET}")

    print(" ".join(formatted_tokens))
    print("\n")

    # --- Step 3: Decoding Back to Structured Data ---
    print("--- Step 3: Decoding Back to Structured Data ---")
    structured_output = tokenizer.decode_mixed_input(token_ids)

    print("Decoded structured output:")
    if 'text' in structured_output:
        print(f"  - Recovered Text: '{structured_output['text']}'")
    if 'pcap_data' in structured_output:
        num_packets = len(structured_output['pcap_data'])
        print(f"  - Recovered PCAP Data: Detected {num_packets} packets.")
        for i, (link_type, time_interval, pkt_bytes) in enumerate(structured_output['pcap_data'][:3]):
            print(
                f"    - Packet {i + 1}: LinkType={link_type}, TimeDelta={time_interval:.6f}s, Size={len(pkt_bytes)} bytes")
        if num_packets > 3:
            print("    - ... (and so on)")

    # --- Step 4: Generating 2D Positional Indices ---
    print("\n--- Step 4: Generating 2D Positional Indices ---")
    token_ids_tensor, position_indices = tokenizer.tokenize_with_2d_positions(
        text=example_text,
        pcap_file_path=pcap_file_path
    )

    print(f"Shape of token_ids_tensor: {token_ids_tensor.shape}")
    print(f"Shape of position_indices: {position_indices.shape}")
    print("Position indices tensor contains [row, column] positions for each token.")


if __name__ == "__main__":
    main()