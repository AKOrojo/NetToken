# Test the corrected tokenizer
from src.nettokken.bpe.byte_level_field_aware_tokenizer import ByteLevelFieldAwareTokenizer

tokenizer = ByteLevelFieldAwareTokenizer()

# Train on some PCAP files
training_files = ['../data/test.pcap']
tokenizer.train(training_files, num_merges=100)

# Test encoding/decoding
test_bytes = b'\x45\x00\x00\x1c'
encoded = tokenizer.encode_bytes(test_bytes)
decoded = tokenizer.decode_tokens(encoded)
print(f"Round-trip test: {test_bytes == decoded}")

# Save and reload
tokenizer.save_trained_data('vocab.json', 'merges.json')
new_tokenizer = ByteLevelFieldAwareTokenizer('vocab.json', 'merges.json')

# Tokenize a PCAP
tokens = new_tokenizer.tokenize_pcap('../data/test.pcap')
print(f"Tokenized {len(tokens)} tokens")