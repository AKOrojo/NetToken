import logging
import os
import struct
from collections import defaultdict

from scapy.all import rdpcap, Raw


class PCAPTokenizer:
    def __init__(self, vocab_size=280, offset=3):
        self.vocab_size = vocab_size
        self.offset = offset
        self.special_tokens = {
            'packet_start': 0x100 + offset,
            'packet_end': 0x101 + offset,
            'flow_end': 0x102 + offset,
            'field_sep': 0x103 + offset,
        }
        if vocab_size < 280:
            raise ValueError(f"Vocab size {vocab_size} is too small. Minimum is 280.")
        self.hex_to_token = {i: i + offset for i in range(256)}
        self.allocated_tokens = set(range(offset, 256 + offset))
        self.allocated_tokens.update(self.special_tokens.values())
        self.link_types = {
            0: self._allocate_token(), 1: self._allocate_token(),
            8: self._allocate_token(), 9: self._allocate_token(),
            10: self._allocate_token(), 101: self._allocate_token(),
            105: self._allocate_token(), 113: self._allocate_token(),
            127: self._allocate_token(),
        }
        self.flows = defaultdict(list)
        self.logger = logging.getLogger('PCAPTokenizer')

    def _allocate_token(self):
        for token_id in range(256 + self.offset, self.vocab_size + self.offset):
            if token_id not in self.allocated_tokens:
                self.allocated_tokens.add(token_id)
                return token_id
        raise ValueError(f"Token vocabulary limit of {self.vocab_size} exceeded")

    def tokenize_pcap(self, pcap_file):
        """
        Tokenize a PCAP file.
        Args:
            pcap_file: Path to the PCAP file.
        Returns:
            Dictionary mapping flow identifiers to token lists
        """
        try:
            packets_from_file = rdpcap(pcap_file)
        except Exception as e:
            self.logger.error(f"Error reading PCAP file '{pcap_file}': {e}")
            return {}

        if not packets_from_file:
            self.logger.warning(f"No packets found in PCAP file '{pcap_file}'.")
            return {}

        self.flows = defaultdict(list)
        base_name = os.path.basename(pcap_file)
        flow_id = f"{base_name}"

        # Scapy's rdpcap reads packets in order, but explicit sorting is safer
        sorted_packets = sorted(packets_from_file, key=lambda p: float(p.time))
        self.flows[flow_id] = sorted_packets

        tokenized_flows_output = {}
        for flow_id, flow_packets_list in self.flows.items():
            if not flow_packets_list:
                continue
            tokenized_flows_output[flow_id] = self._tokenize_flow(flow_packets_list)

        return tokenized_flows_output

    def _tokenize_flow(self, packets_in_flow):
        tokens = []
        prev_time = None
        for packet in packets_in_flow:
            tokens.append(self.special_tokens['packet_start'])

            # Add link type with field separator
            link_type_token = self._get_link_type_token(packet)
            tokens.append(link_type_token)
            tokens.append(self.special_tokens['field_sep'])

            # Add timing information with field separator
            curr_time = float(packet.time)
            time_interval = curr_time - prev_time if prev_time is not None else 0.0
            if time_interval < 0:
                self.logger.warning(f"Negative time interval ({time_interval}s) detected.")
                time_interval = 0.0
            time_tokens = self._encode_time_interval(time_interval)
            tokens.extend(time_tokens)
            tokens.append(self.special_tokens['field_sep'])

            prev_time = curr_time

            # Add packet data using the new, robust method
            packet_tokens = self._encode_packet_data(packet)
            tokens.extend(packet_tokens)

            # Remove trailing separator if one was added
            if tokens and tokens[-1] == self.special_tokens['field_sep']:
                tokens.pop()

            tokens.append(self.special_tokens['packet_end'])

        tokens.append(self.special_tokens['flow_end'])
        return tokens

    def _get_link_type_token(self, packet):
        link_type = packet.linktype if hasattr(packet, 'linktype') else 1  # Default to Ethernet
        if link_type in self.link_types:
            return self.link_types[link_type]
        try:
            self.link_types[link_type] = self._allocate_token()
            return self.link_types[link_type]
        except ValueError:
            self.logger.warning(f"Vocab limit for link type {link_type}, using default.")
            return self.link_types.get(1)

    def _encode_time_interval(self, time_interval):
        # This function now returns raw bytes (0-255)
        packed = struct.pack('!d', time_interval)
        return list(packed)

    def _encode_packet_data(self, packet):
        """
        Encode packet data by iterating through Scapy's parsed fields.
        This is more robust than manual byte parsing.
        """
        tokens = []
        current_layer = packet

        while current_layer:
            if isinstance(current_layer, (Raw)):
                break

            layer_name = current_layer.name

            for field in current_layer.fields_desc:
                if field.name in ['options', 'load', 'padding']: continue
                if not hasattr(current_layer, field.name): continue

                field_value = current_layer.getfieldval(field.name)

                try:
                    field_bytes = field.i2m(packet, field_value)
                    if field_bytes:
                        # Append raw bytes (0-255), not offset tokens
                        tokens.extend(list(field_bytes))
                        tokens.append(self.special_tokens['field_sep'])
                except Exception as e:
                    self.logger.debug(f"Could not convert field '{field.name}' to bytes: {e}")

            if hasattr(current_layer, 'options') and current_layer.options:
                try:
                    options_bytes = b''.join(opt.pack() for opt in current_layer.options)
                    if options_bytes:
                        # Append raw bytes (0-255), not offset tokens
                        tokens.extend(list(options_bytes))
                        tokens.append(self.special_tokens['field_sep'])
                except Exception as e:
                    self.logger.debug(f"Could not pack options for layer {layer_name}: {e}")

            current_layer = current_layer.payload

        if isinstance(current_layer, Raw):
            payload_bytes = bytes(current_layer.load)
            if payload_bytes:
                # Append raw bytes (0-255), not offset tokens
                tokens.extend(list(payload_bytes))
                tokens.append(self.special_tokens['field_sep'])

        return tokens

    def decode_flow(self, tokens):
        """
        Decodes a list of tokens back into a list of Scapy packets with metadata.

        Args:
            tokens: A list of integer tokens generated by _tokenize_flow.

        Returns:
            A list of tuples, where each tuple contains (link_type, time_interval, raw_packet_bytes).
        """
        if not tokens:
            return []

        # Create a reverse mapping from token ID to link type number
        token_to_link_type = {v: k for k, v in self.link_types.items()}

        decoded_packets = []
        i = 0
        while i < len(tokens):
            if tokens[i] != self.special_tokens['packet_start']:
                # Skip any unexpected tokens until a packet starts
                i += 1
                continue

            i += 1  # Move past 'packet_start'

            # --- 1. Decode Link Type ---
            link_type_token = tokens[i]
            link_type = token_to_link_type.get(link_type_token, 1)  # Default to Ethernet
            i += 2  # Move past link_type_token and its field_sep

            # --- 2. Decode Time Interval ---
            time_bytes = bytes(tokens[i:i + 8])
            if len(time_bytes) == 8:
                time_interval = struct.unpack('!d', time_bytes)[0]
            else:
                time_interval = 0.0  # Default on error

            i += 8  # Move past time tokens
            if i < len(tokens) and tokens[i] == self.special_tokens['field_sep']:
                i += 1  # Move past time's field_sep

            # --- 3. Decode Packet Bytes ---
            packet_data_bytes = []
            while i < len(tokens) and tokens[i] != self.special_tokens['packet_end']:
                if tokens[i] != self.special_tokens['field_sep']:
                    byte_val = tokens[i]
                    packet_data_bytes.append(byte_val)
                i += 1

            raw_packet = bytes(packet_data_bytes)

            # THIS IS THE FIX: Append the 3-item tuple the main script expects.
            decoded_packets.append((link_type, time_interval, raw_packet))

            if i < len(tokens) and tokens[i] == self.special_tokens['packet_end']:
                i += 1  # Move past 'packet_end'

            if i < len(tokens) and tokens[i] == self.special_tokens['flow_end']:
                break  # End of flow reached

        return decoded_packets