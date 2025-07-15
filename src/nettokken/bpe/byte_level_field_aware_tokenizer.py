#!/usr/bin/env python3
"""
Optimized BPE training script for large-scale PCAP data processing.
Fixed multiprocessing temp file handling issues.
"""

import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

import psutil
from scapy.all import rdpcap, Packet
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pcap_file_worker(pcap_path: str, max_sequence_length: int = 1024) -> List[List[int]]:
    """Process a single PCAP file and return field sequences - standalone function for multiprocessing"""
    sequences = []
    try:
        packets = rdpcap(pcap_path)
        for packet in packets:
            for field_name, field_bytes in _get_field_byte_slices(packet):
                token_sequence = list(field_bytes)
                if 1 < len(token_sequence) <= max_sequence_length:
                    sequences.append(token_sequence)
    except Exception as e:
        logger.warning(f"Error processing {pcap_path}: {e}")
    return sequences


def _get_field_byte_slices(packet: Packet):
    """Extract field byte slices from packet (standalone function for multiprocessing)"""
    raw_packet_bytes = bytes(packet)
    current_pos = 0
    layer = packet

    while layer:
        layer_name = layer.name

        if not hasattr(layer, 'fields_desc') or not layer.fields_desc:
            if current_pos < len(raw_packet_bytes):
                yield f"{layer_name}_payload", raw_packet_bytes[current_pos:]
            break

        next_layer_start = len(raw_packet_bytes)
        if layer.payload:
            try:
                next_layer_start = raw_packet_bytes.find(bytes(layer.payload), current_pos)
            except:
                pass

        for field in layer.fields_desc:
            if hasattr(layer, field.name):
                field_value = getattr(layer, field.name)
                try:
                    field_bytes = field.addfield(packet, b'', field_value)
                    field_len = len(field_bytes)

                    field_start = current_pos
                    field_end = field_start + field_len

                    if field_end <= next_layer_start:
                        field_name = f"{layer_name}_{field.name}"
                        yield field_name, raw_packet_bytes[field_start:field_end]
                        current_pos = field_end
                    else:
                        break
                except:
                    continue

        current_pos = next_layer_start
        layer = layer.payload


def process_pcap_chunk_worker(pcap_files_chunk: List[str], temp_dir: str, max_sequence_length: int) -> str:
    """Process a chunk of PCAP files and save to temporary file - standalone function for multiprocessing"""
    # Create unique temp file name
    chunk_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    temp_file = os.path.join(temp_dir, f"sequences_{chunk_id}.pkl")

    all_sequences = []
    for pcap_file in pcap_files_chunk:
        sequences = process_pcap_file_worker(pcap_file, max_sequence_length)
        all_sequences.extend(sequences)

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Save sequences to temporary file
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(all_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Verify file was created
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Failed to create temp file: {temp_file}")

        logger.info(f"Worker {os.getpid()} created temp file: {temp_file} with {len(all_sequences)} sequences")
        return temp_file

    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to create temp file {temp_file}: {e}")
        raise


class OptimizedBPETrainer:
    """Optimized BPE trainer for large-scale PCAP data"""

    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 vocab_sizes: List[int],
                 num_workers: int = None,
                 chunk_size: int = 100,
                 max_sequence_length: int = 1024,
                 checkpoint_interval: int = 100):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vocab_sizes = sorted(vocab_sizes)
        self.num_workers = num_workers or max(1, mp.cpu_count() - 2)
        self.chunk_size = chunk_size  # Number of PCAP files per chunk
        self.max_sequence_length = max_sequence_length
        self.checkpoint_interval = checkpoint_interval

        # Create temp directory on local SSD for intermediate data
        self.temp_dir = Path(tempfile.mkdtemp(dir='/tmp', prefix='bpe_training_'))
        logger.info(f"Using temp directory: {self.temp_dir}")

        # Initialize base vocabulary
        self.base_vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {
            "<pad>": 256,
            "<unk>": 257,
            "<field_sep>": 258,
            "<eos>": 259,
        }

    def __del__(self):
        # Cleanup temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def collect_initial_sequences(self) -> List[str]:
        """Collect all sequences from PCAP files using parallel processing"""
        pcap_files = list(self.data_dir.glob("*.pcap")) + list(self.data_dir.glob("*.pcapng"))
        logger.info(f"Found {len(pcap_files)} PCAP files")

        if not pcap_files:
            raise ValueError("No PCAP files found in data directory")

        # Convert paths to strings for serialization
        pcap_file_paths = [str(p) for p in pcap_files]

        # Split files into chunks
        chunks = [pcap_file_paths[i:i + self.chunk_size]
                  for i in range(0, len(pcap_file_paths), self.chunk_size)]

        temp_files = []

        # Use smaller number of workers to avoid resource contention
        actual_workers = min(self.num_workers, len(chunks))
        logger.info(f"Using {actual_workers} workers for {len(chunks)} chunks")

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(
                    process_pcap_chunk_worker,
                    chunk,
                    str(self.temp_dir),
                    self.max_sequence_length
                )
                future_to_chunk[future] = chunk

            # Collect results
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks),
                               desc="Processing PCAP files"):
                try:
                    temp_file = future.result(timeout=300)  # 5 minute timeout per chunk

                    # Verify the file exists before adding to list
                    if os.path.exists(temp_file):
                        temp_files.append(temp_file)
                        logger.info(f"Successfully processed chunk, temp file: {temp_file}")
                    else:
                        logger.error(f"Temp file {temp_file} does not exist after processing")

                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Failed to process chunk {chunk}: {e}")
                    continue

                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent}%")

        logger.info(f"Successfully created {len(temp_files)} temporary files")
        return temp_files

    def count_pairs_from_files(self, temp_files: List[str],
                               current_merges: Dict[Tuple[int, int], int]) -> Counter:
        """Count pairs from temporary files efficiently"""
        pair_counts = Counter()

        for temp_file in tqdm(temp_files, desc="Counting pairs"):
            try:
                if not os.path.exists(temp_file):
                    logger.warning(f"Temp file {temp_file} not found, skipping")
                    continue

                with open(temp_file, 'rb') as f:
                    sequences = pickle.load(f)

                for sequence in sequences:
                    # Apply current merges to sequence
                    merged_sequence = self.apply_merges_to_sequence(sequence, current_merges)

                    # Count pairs
                    for i in range(len(merged_sequence) - 1):
                        pair = (merged_sequence[i], merged_sequence[i + 1])
                        pair_counts[pair] += 1

                # Periodic garbage collection
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing temp file {temp_file}: {e}")
                continue

        return pair_counts

    @staticmethod
    def apply_merges_to_sequence(self, sequence: List[int],
                                 merges: Dict[Tuple[int, int], int]) -> List[int]:
        """Apply merges to a single sequence"""
        if not merges:
            return sequence

        result = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1:
                pair = (sequence[i], sequence[i + 1])
                if pair in merges:
                    result.append(merges[pair])
                    i += 2
                    continue
            result.append(sequence[i])
            i += 1

        return result

    def update_sequences_batch(self, temp_file: str, most_frequent_pair: Tuple[int, int],
                               new_token_id: int, current_merges: Dict[Tuple[int, int], int]) -> str:
        """Update sequences in a temporary file with new merge"""
        try:
            if not os.path.exists(temp_file):
                logger.warning(f"Temp file {temp_file} not found for update")
                return temp_file

            with open(temp_file, 'rb') as f:
                sequences = pickle.load(f)

            # Create new temporary file
            new_temp_file = self.temp_dir / f"sequences_updated_{os.getpid()}_{int(time.time() * 1000000)}.pkl"

            # Apply all merges including the new one
            updated_merges = current_merges.copy()
            updated_merges[most_frequent_pair] = new_token_id

            updated_sequences = []
            for sequence in sequences:
                merged_sequence = self.apply_merges_to_sequence(sequence, updated_merges)
                updated_sequences.append(merged_sequence)

            with open(new_temp_file, 'wb') as f:
                pickle.dump(updated_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Remove old file
            os.remove(temp_file)

            return str(new_temp_file)

        except Exception as e:
            logger.error(f"Error updating sequences in {temp_file}: {e}")
            return temp_file

    def train_bpe(self, vocab_size: int):
        """Train BPE with specified vocabulary size"""
        logger.info(f"Training BPE with vocabulary size {vocab_size}")

        # Calculate number of merges needed
        num_merges = vocab_size - len(self.base_vocab) - len(self.special_tokens)

        # Initialize vocabulary and merges
        vocab = self.base_vocab.copy()
        for token_str, token_id in self.special_tokens.items():
            vocab[token_id] = token_str.encode('utf-8')

        merges = {}
        merge_lookup = {}
        next_token_id = max(self.special_tokens.values()) + 1

        # Collect initial sequences
        logger.info("Collecting initial sequences...")
        temp_files = self.collect_initial_sequences()

        if not temp_files:
            raise ValueError("No sequences collected from PCAP files")

        # Training loop
        for merge_step in range(num_merges):
            logger.info(f"Merge step {merge_step + 1}/{num_merges}")

            # Count pairs
            pair_counts = self.count_pairs_from_files(temp_files, merge_lookup)

            if not pair_counts:
                logger.info("No more pairs to merge")
                break

            # Find most frequent pair
            most_frequent_pair, count = pair_counts.most_common(1)[0]
            logger.info(f"Most frequent pair: {most_frequent_pair} (count: {count})")

            # Record merge
            merges[most_frequent_pair] = {
                'rank': merge_step,
                'token_id': next_token_id
            }
            merge_lookup[most_frequent_pair] = next_token_id

            # Update vocabulary
            left_bytes = vocab[most_frequent_pair[0]]
            right_bytes = vocab[most_frequent_pair[1]]
            vocab[next_token_id] = left_bytes + right_bytes

            # Update sequences (single-threaded to avoid complications)
            new_temp_files = []
            for temp_file in tqdm(temp_files, desc="Updating sequences"):
                new_temp_file = self.update_sequences_batch(
                    temp_file, most_frequent_pair, next_token_id, merge_lookup
                )
                new_temp_files.append(new_temp_file)

            temp_files = new_temp_files
            next_token_id += 1

            # Checkpoint
            if (merge_step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(vocab_size, vocab, merges, merge_step + 1)

            # Memory management
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            logger.info(f"Memory usage: {memory_percent}%")

        # Save final model
        self.save_model(vocab_size, vocab, merges)

        # Cleanup temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        logger.info(f"Training complete for vocab size {vocab_size}")

    def save_checkpoint(self, vocab_size: int, vocab: Dict, merges: Dict, step: int):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoints_vocab{vocab_size}"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step{step}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'vocab': vocab,
                'merges': merges,
                'step': step
            }, f)
        logger.info(f"Saved checkpoint at step {step}")

    def save_model(self, vocab_size: int, vocab: Dict, merges: Dict):
        """Save final model"""
        model_dir = self.output_dir / f"bpe_vocab{vocab_size}"
        model_dir.mkdir(exist_ok=True)

        # Save vocabulary
        serializable_vocab = {}
        for k, v in vocab.items():
            if isinstance(v, bytes):
                serializable_vocab[k] = list(v)
            else:
                serializable_vocab[k] = v

        with open(model_dir / "vocab.json", 'w') as f:
            json.dump(serializable_vocab, f, indent=2)

        # Save merges
        serializable_merges = {
            f"{p[0]},{p[1]}": v for p, v in merges.items()
        }
        with open(model_dir / "merges.json", 'w') as f:
            json.dump(serializable_merges, f, indent=2)

        logger.info(f"Saved model with vocab size {vocab_size}")

    def train_all(self):
        """Train models for all specified vocabulary sizes"""
        for vocab_size in self.vocab_sizes:
            start_time = time.time()

            try:
                self.train_bpe(vocab_size)

                elapsed_time = time.time() - start_time
                logger.info(f"Completed vocab size {vocab_size} in {elapsed_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error training vocab size {vocab_size}: {e}")
                continue

            # Cleanup between runs
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Train BPE on large PCAP dataset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing PCAP files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save trained models")
    parser.add_argument("--vocab-sizes", type=int, nargs="+",
                        default=[512, 1024, 4096, 8192, 16384, 32768],
                        help="Vocabulary sizes to train")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of PCAP files per chunk")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N merge steps")

    args = parser.parse_args()

    # Log system info
    logger.info(f"CPU cores: {mp.cpu_count()}")
    logger.info(f"Total memory: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")

    # Create trainer and run
    trainer = OptimizedBPETrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_sizes=args.vocab_sizes,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval
    )

    trainer.train_all()


if __name__ == "__main__":
    main()