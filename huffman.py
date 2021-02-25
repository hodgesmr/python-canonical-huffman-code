# 
# Compression software using a Canonical Huffman code
# https://en.wikipedia.org/wiki/Canonical_Huffman_code
# 
# Usage:
# Compress: python huffman.py -i input_file -o output_file
# Decompress: python huffman.py -d -i input_file -o output_file
# 
# The softare uses an alphabet of 257 symbols - 256 symbols for the byte values
# and 1 symbol for the EOF marker. The compressed file format starts with a header of 257
# code lengths, treated as a canonical code, and then followed by the Huffman-coded data.
# 

from __future__ import annotations

import argparse
import heapq
import os
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

EOF = 256  # 1 greater than the possible values for input bytes
DEPTH_LIMIT = 255  # 1 byte
SYMBOL_LIMIT = 257  # Bytes 00000000 -> 11111111, plus our EOF symbol

class Node(ABC):
    pass


class InternalNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right


class Leaf(Node):
    def __init__(self, symbol: int):
        if symbol < 0:
            raise ValueError("Symbol must be non-negative")
        self.symbol = symbol


class HeapObject():
    def __init__(self, frequency: int, symbol: int, node:Node):
        self.frequency = frequency
        self.symbol = symbol
        self.node = node

    def __lt__(self, other: HeapObject):
        if self.frequency == other.frequency:
            return self.symbol < other.symbol
        return self.frequency < other.frequency

    def __gt__(self, other: HeapObject):
        if self.frequency == other.frequency:
            return self.symbol > other.symbol
        return self.frequency > other.frequency

    def __eq__(self, other: HeapObject):
        return self.frequency == other.frequency and self.symbol == other.symbol

    def __ne__(self, other: HeapObject):
        return not(self == other)


class Code():
    def __init__(self, symbol: int, value: int, depth: int):
        self.symbol = symbol
        self.value = value
        self.depth = depth        

    def __eq__(self, other: Code):
        return self.value == other.value and self.depth == other.depth

    def __ne__(self, other: Code):
        return not(self == other)

    def __hash__(self):
        return hash((self.value, self.depth))


class BitWriter():
    def __init__(self, out: BinaryIO):
        self.out = out
        self.current_byte = 0
        self.bits_written = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def write_bit(self, b):
        if b not in (0, 1):
            raise ValueError(f"Got unexpected bit: {b}")

        # Push the bit onto the byte buffer, writing a full byte at a time
        self.current_byte = (self.current_byte << 1) | b
        self.bits_written += 1
        if self.bits_written > 0 and self.bits_written % 8 == 0:
            self.out.write(bytes([self.current_byte]))
            self.current_byte = 0

    def write_bits_for_code(self, code: Code):
        if code.depth > DEPTH_LIMIT:
            raise ValueError(f"Unsupported output depth: {code.depth} for input symbol: {code.symbol}")
        # Walk the code value for its entire depth, writing each bit
        for i in reversed(range(code.depth)):
            place = code.value & (1 << i)
            if place > 0:
                self.write_bit(1)
            else:
                self.write_bit(0)
    
    def close(self):
        # If we have unwritten bits, pad with zeros, then write
        while self.bits_written % 8 != 0:
            self.write_bit(0)


class BitReader():
    def __init__(self, input: BinaryIO):
        self.input = input
        self.current_byte = 0
        self.bits_read = 0

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def read_bit(self):
        while self.current_byte != -1:
            if self.bits_read % 8 == 0:
                input_bytes = self.input.read(1)
                if len(input_bytes) < 1:
                    self.current_byte = -1
                    return
                self.current_byte = input_bytes[0]
            
            place = self.current_byte & (1 << (8-(self.bits_read % 8))-1)
            self.bits_read += 1

            if place > 0:
                yield 1
            else:
                yield 0

    def close(self):
        self.current_byte = -1


def compress(input_file_path: str, output_file_path: str):
    # Indexes 0-255 are for out input file byte frequencies
    # Index 256 will have a frequency of 1, to indicate our EOF marker
    # Because our word length is 1 byte (8 bits), 255 is the max value, leaving symbol 256 available
    frequencies = [0]*SYMBOL_LIMIT
    frequencies[EOF] = 1

    with open(input_file_path, "rb") as input_file, open(output_file_path, "wb") as output_file:
        while (input_bytes := input_file.read(1)):  # Returns a bytes object of length 1
            frequencies[input_bytes[0]] += 1

        heap = []
        # Build the heap of symbols with non-zero frequencies
        for i, freq in enumerate(frequencies):
            if freq > 0:
                heapq.heappush(heap, HeapObject(freq, i, Leaf(i)))

        # Pad until there are at least two items
        for i, freq in enumerate(frequencies):
            if len(heap) >=2:
                break
            if freq == 0:
                heapq.heappush(heap, HeapObject(freq, i, Leaf(i)))

        # At this point, if we don't have at least two items, we can't continue
        if len(heap) < 2:
            raise ValueError("Input file has insufficient data to apply encoding.")

        # Loop over the HeapObjects and build out the code tree of InternalNodes and Leafs
        while len(heap) > 1:
            # Pop off two HeapObjects
            x = heapq.heappop(heap)
            y = heapq.heappop(heap)
            # Create a new HeapObject made of internal nodes based on x and y
            # The Heap was created with only Leaf objects, but we're building the code tree with InternalNodes
            # Once thsi loop gets the heap down to len == 1, we'll have our code tree like:
            #       .
            #      / \
            #     A   .
            #        / \
            #       B   .
            #          / \
            #         C   D
            # This process took me a while to understand!
            # It helped to draw a simple min-heap on paper (4+ nodes) and walk through it.
            # Also, the examples and comments in this project were extremely helpful:
            # https://github.com/nayuki/Reference-Huffman-coding/tree/master/python
            z = HeapObject(x.frequency + y.frequency, min(x.symbol, y.symbol), InternalNode(x.node, y.node))
            heapq.heappush(heap, z)

        # Now, we have a single HeapObject, with our code tree inside
        code_tree = heap[0].node

        codes = set()
        def build_code_list(node, code_value, depth):
            if isinstance(node, InternalNode):
                build_code_list(node.left, (code_value<<1), depth+1)
                build_code_list(node.right, (code_value<<1)+1, depth+1)
            elif isinstance(node, Leaf):
                codes.add(Code(node.symbol, code_value, depth))

        build_code_list(code_tree, 0, 0)

        # https://en.wikipedia.org/wiki/Canonical_Huffman_code#Algorithm
        # Sort first by code depts and second by alphabetical value
        canonical_codes = sorted(codes, key=lambda x: (x.depth, x.symbol))
        # The first symbol gets assigned a codeword which is the same length as the symbol's original
        # codeword but all zeros
        current_depth = canonical_codes[0].depth
        # Each subsequent symbol is assigned the next binary number in sequence
        # When you reach a longer codeword, then after incrementing, append zeros until
        # the length of the new codeword is equal to the length of the old codeword
        codebook_depths = [0]*SYMBOL_LIMIT
        next_code_value = 0
        for code in canonical_codes:
            code.value = next_code_value
            if code.depth > current_depth:
                code.value = code.value << (code.depth - current_depth)
                current_depth = code.depth
            next_code_value = code.value + 1

            codebook_depths[code.symbol] = current_depth

        # Now fill out the code book array for easy index access symbol (0-256) -> code
        canonical_codebook = [None]*SYMBOL_LIMIT
        for code in canonical_codes:
            canonical_codebook[code.symbol] = code

        # The first 257 bytes of the file are the codebook depths
        for depth in codebook_depths:
            output_file.write(bytes([depth]))

        with BitWriter(output_file) as bit_writer:  # Use a context manager to automatically call .close() at the end
            # We've already read the file once to build the codebook
            # Seek to the begginging to read again and encode
            input_file.seek(0)
            while (input_bytes := input_file.read(1)):  # read 1 byte at a time
                code = canonical_codebook[input_bytes[0]]  # translate to our huffman code
                bit_writer.write_bits_for_code(code)  # write the bits for that code
        
            # Now we're at the end of the input file, so write our EOF symbol
            code = canonical_codebook[EOF]
            bit_writer.write_bits_for_code(code)


def decompress(input_file_path: str, output_file_path: str):
    # The depths of each symbol, Depth -> [Sym_1, Sym_2...]
    codebook_depths = defaultdict(list)

    # A dictionary of Code -> Symbol
    # Codes are hashed by Value and Depth
    codebook = {}

    with open(input_file_path, "rb") as input_file, open(output_file_path, "wb") as output_file:
        # Read the codebook_depth header
        for symbol in range(SYMBOL_LIMIT):  
            input_bytes = input_file.read(1)
            depth = input_bytes[0]
            codebook_depths[depth].append(symbol)
        
        # Rebuild the symbol table
        # Again: https://en.wikipedia.org/wiki/Canonical_Huffman_code#Algorithm
        current_depth = None
        next_code_value = 0
        for depth in range(DEPTH_LIMIT):
            symbols_at_depth = codebook_depths[depth]
            if depth and symbols_at_depth:
                if not current_depth:
                    current_depth = depth
                if depth > current_depth:
                    next_code_value = next_code_value << (depth - current_depth)
                    current_depth = depth
                for symbol in symbols_at_depth:
                    code = Code(symbol, next_code_value, depth)
                    codebook[code] = symbol
                    next_code_value = next_code_value + 1
        
        # We now have a codebook to translate a Code's Value and Depth back to it's uncompressed Symbol
        # Read in the compressed file, yielding one bit at a time
        with BitReader(input_file) as bit_reader:
            code_candidate = Code(0, 0, 0)  # A dummy code that will attempt to lookup on value and depth
            for bit in bit_reader.read_bit():
                code_candidate.depth += 1
                if bit == 0:
                    code_candidate.value = (code_candidate.value<<1)
                else:
                    code_candidate.value = (code_candidate.value<<1)+1
                # When we find a translation
                if code_candidate in codebook:  # This works because Code hashing and equality ignores symbol
                    if codebook[code_candidate] == EOF:  # Done writing
                        break
                    # We have the original Symbol from the stream, write it to the output
                    output_file.write(bytes([codebook[code_candidate]]))
                    # And prepare a new candidate for the next Symbol
                    code_candidate = Code(0, 0, 0)    
        

def run(input_file_path: str, output_file_path: str, decompress_mode: bool=False):
    if decompress_mode:
        decompress(input_file_path, output_file_path)
    else:
        compress(input_file_path, output_file_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(prog="huffman")
    arg_parser.add_argument("-i", dest="input_file_path", help="The path of the file to compress.", required=True)
    arg_parser.add_argument("-o", dest="output_file_path", help="The output file path for the compresed file", required=True)
    arg_parser.add_argument("-d", dest="decompress", action="store_true", help="Flag to run in decompress mode; otherwise run in compress mode")
    args = arg_parser.parse_args()

    run(args.input_file_path, args.output_file_path, args.decompress)
