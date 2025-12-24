"""
Comprehensive JPEG Encoder Implementation

This module implements a complete baseline JPEG encoder from scratch, including:
- RGB to YCbCr color space conversion
- Chroma subsampling (4:4:4, 4:2:2, 4:2:0)
- 8x8 block-based Discrete Cosine Transform (DCT)
- Quantization with standard/custom tables
- Zigzag scanning
- Run-length encoding (RLE) for AC coefficients
- Differential pulse-code modulation (DPCM) for DC coefficients
- Huffman encoding with standard JPEG tables
- Complete JPEG file structure with all required markers

Author: JPEG Encoder Implementation
"""

import numpy as np
from typing import Tuple, List, Optional, BinaryIO
import struct
from enum import Enum
from dataclasses import dataclass
import math


# =============================================================================
# JPEG Markers
# =============================================================================

class Marker:
    """JPEG marker definitions"""
    SOI = 0xFFD8   # Start of Image
    EOI = 0xFFD9   # End of Image
    APP0 = 0xFFE0  # Application segment (JFIF)
    DQT = 0xFFDB   # Define Quantization Table
    SOF0 = 0xFFC0  # Start of Frame (Baseline DCT)
    DHT = 0xFFC4   # Define Huffman Table
    SOS = 0xFFDA   # Start of Scan
    COM = 0xFFFE   # Comment


# =============================================================================
# Subsampling Modes
# =============================================================================

class SubsamplingMode(Enum):
    """Chroma subsampling modes"""
    MODE_444 = "4:4:4"  # No subsampling
    MODE_422 = "4:2:2"  # Horizontal subsampling
    MODE_420 = "4:2:0"  # Horizontal and vertical subsampling


# =============================================================================
# Standard JPEG Quantization Tables (ITU-T T.81 Annex K)
# =============================================================================

LUMINANCE_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

CHROMINANCE_QUANTIZATION_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float64)


# =============================================================================
# Zigzag Scanning Order
# =============================================================================

ZIGZAG_ORDER = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])


# =============================================================================
# Standard Huffman Tables (ITU-T T.81 Annex K)
# =============================================================================

# DC Luminance Huffman Table
DC_LUMINANCE_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DC_LUMINANCE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# DC Chrominance Huffman Table
DC_CHROMINANCE_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
DC_CHROMINANCE_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# AC Luminance Huffman Table
AC_LUMINANCE_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
AC_LUMINANCE_VALUES = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
]

# AC Chrominance Huffman Table
AC_CHROMINANCE_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
AC_CHROMINANCE_VALUES = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
]


# =============================================================================
# Huffman Table Class
# =============================================================================

@dataclass
class HuffmanCode:
    """Represents a single Huffman code"""
    code: int
    length: int


class HuffmanTable:
    """Huffman encoding/decoding table"""

    def __init__(self, bits: List[int], values: List[int]):
        """
        Build Huffman table from JPEG-format specification.

        Args:
            bits: Number of codes of each length (1-16 bits)
            values: Symbol values in order of increasing code length
        """
        self.bits = bits
        self.values = values
        self.codes = {}
        self._build_table()

    def _build_table(self):
        """Build the Huffman code table"""
        code = 0
        value_idx = 0

        for bit_length in range(1, 17):
            for _ in range(self.bits[bit_length - 1]):
                if value_idx < len(self.values):
                    self.codes[self.values[value_idx]] = HuffmanCode(code, bit_length)
                    value_idx += 1
                code += 1
            code <<= 1

    def encode(self, symbol: int) -> HuffmanCode:
        """Get Huffman code for a symbol"""
        if symbol not in self.codes:
            raise ValueError(f"Symbol {symbol} not in Huffman table")
        return self.codes[symbol]


# =============================================================================
# Bit Writer for JPEG Stream
# =============================================================================

class BitWriter:
    """Writes bits to a byte stream with JPEG byte stuffing"""

    def __init__(self):
        self.buffer = bytearray()
        self.bit_buffer = 0
        self.bit_count = 0

    def write_bits(self, value: int, num_bits: int):
        """Write bits to the buffer"""
        if num_bits == 0:
            return

        # Add bits to buffer
        self.bit_buffer = (self.bit_buffer << num_bits) | (value & ((1 << num_bits) - 1))
        self.bit_count += num_bits

        # Flush complete bytes
        while self.bit_count >= 8:
            self.bit_count -= 8
            byte = (self.bit_buffer >> self.bit_count) & 0xFF
            self.buffer.append(byte)

            # Clear the flushed bits to prevent overflow
            self.bit_buffer &= (1 << self.bit_count) - 1

            # JPEG byte stuffing: add 0x00 after 0xFF
            if byte == 0xFF:
                self.buffer.append(0x00)

    def flush(self):
        """Flush remaining bits, padding with 1s"""
        if self.bit_count > 0:
            # Pad with 1s to byte boundary
            padding = 8 - self.bit_count
            self.write_bits((1 << padding) - 1, padding)

    def get_data(self) -> bytes:
        """Get the written data"""
        return bytes(self.buffer)


# =============================================================================
# Color Space Conversion
# =============================================================================

def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCbCr color space.

    Uses ITU-R BT.601 conversion matrix:
    Y  =  0.299 R + 0.587 G + 0.114 B
    Cb = -0.169 R - 0.331 G + 0.500 B + 128
    Cr =  0.500 R - 0.419 G - 0.081 B + 128

    Args:
        rgb: Input image as (H, W, 3) numpy array with values 0-255

    Returns:
        YCbCr image as (H, W, 3) numpy array with values 0-255
    """
    rgb = rgb.astype(np.float64)

    # Conversion matrix
    transform = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])

    # Apply transformation
    ycbcr = np.zeros_like(rgb)
    ycbcr[:, :, 0] = (transform[0, 0] * rgb[:, :, 0] +
                      transform[0, 1] * rgb[:, :, 1] +
                      transform[0, 2] * rgb[:, :, 2])
    ycbcr[:, :, 1] = (transform[1, 0] * rgb[:, :, 0] +
                      transform[1, 1] * rgb[:, :, 1] +
                      transform[1, 2] * rgb[:, :, 2] + 128)
    ycbcr[:, :, 2] = (transform[2, 0] * rgb[:, :, 0] +
                      transform[2, 1] * rgb[:, :, 1] +
                      transform[2, 2] * rgb[:, :, 2] + 128)

    return np.clip(ycbcr, 0, 255)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr to RGB color space (inverse transformation).

    Args:
        ycbcr: Input image as (H, W, 3) numpy array with values 0-255

    Returns:
        RGB image as (H, W, 3) numpy array with values 0-255
    """
    ycbcr = ycbcr.astype(np.float64)

    rgb = np.zeros_like(ycbcr)
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1] - 128
    cr = ycbcr[:, :, 2] - 128

    rgb[:, :, 0] = y + 1.402 * cr
    rgb[:, :, 1] = y - 0.344136 * cb - 0.714136 * cr
    rgb[:, :, 2] = y + 1.772 * cb

    return np.clip(rgb, 0, 255).astype(np.uint8)


# =============================================================================
# Chroma Subsampling
# =============================================================================

def subsample_channel(channel: np.ndarray, mode: SubsamplingMode,
                      is_chroma: bool) -> np.ndarray:
    """
    Apply chroma subsampling to a single channel.

    Args:
        channel: Single channel as (H, W) numpy array
        mode: Subsampling mode
        is_chroma: True if this is a chroma channel (Cb or Cr)

    Returns:
        Subsampled channel
    """
    if not is_chroma or mode == SubsamplingMode.MODE_444:
        return channel

    h, w = channel.shape

    if mode == SubsamplingMode.MODE_422:
        # Horizontal subsampling only (average pairs horizontally)
        new_w = (w + 1) // 2
        subsampled = np.zeros((h, new_w), dtype=channel.dtype)
        for y in range(h):
            for x in range(new_w):
                x1 = x * 2
                x2 = min(x1 + 1, w - 1)
                subsampled[y, x] = (channel[y, x1].astype(np.float64) +
                                   channel[y, x2].astype(np.float64)) / 2
        return subsampled

    elif mode == SubsamplingMode.MODE_420:
        # Both horizontal and vertical subsampling
        new_h = (h + 1) // 2
        new_w = (w + 1) // 2
        subsampled = np.zeros((new_h, new_w), dtype=channel.dtype)
        for y in range(new_h):
            for x in range(new_w):
                y1 = y * 2
                y2 = min(y1 + 1, h - 1)
                x1 = x * 2
                x2 = min(x1 + 1, w - 1)
                subsampled[y, x] = (channel[y1, x1].astype(np.float64) +
                                   channel[y1, x2].astype(np.float64) +
                                   channel[y2, x1].astype(np.float64) +
                                   channel[y2, x2].astype(np.float64)) / 4
        return subsampled

    return channel


# =============================================================================
# Block Operations
# =============================================================================

def pad_to_multiple(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Pad image to be a multiple of block_size in both dimensions.
    Uses edge replication for padding.

    Args:
        image: Input image as (H, W) or (H, W, C) numpy array
        block_size: Block size (default 8 for JPEG)

    Returns:
        Padded image
    """
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        h, w, channels = image.shape

    # Calculate padding needed
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h == 0 and pad_w == 0:
        return image.squeeze() if channels == 1 else image

    # Pad using edge values
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

    return padded.squeeze() if channels == 1 else padded


def split_into_blocks(channel: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Split a 2D array into block_size x block_size blocks.

    Args:
        channel: Input channel as (H, W) numpy array
        block_size: Block size (default 8)

    Returns:
        Array of blocks with shape (num_blocks_h, num_blocks_w, block_size, block_size)
    """
    h, w = channel.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    blocks = np.zeros((num_blocks_h, num_blocks_w, block_size, block_size),
                      dtype=channel.dtype)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            blocks[i, j] = channel[i*block_size:(i+1)*block_size,
                                   j*block_size:(j+1)*block_size]

    return blocks


# =============================================================================
# Discrete Cosine Transform (DCT)
# =============================================================================

def create_dct_matrix(n: int = 8) -> np.ndarray:
    """
    Create the DCT-II transformation matrix.

    Args:
        n: Size of the matrix (default 8 for JPEG)

    Returns:
        DCT matrix of shape (n, n)
    """
    dct_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == 0:
                dct_matrix[i, j] = 1 / np.sqrt(n)
            else:
                dct_matrix[i, j] = np.sqrt(2 / n) * np.cos((2 * j + 1) * i * np.pi / (2 * n))

    return dct_matrix


# Pre-compute DCT matrix for 8x8 blocks
DCT_MATRIX = create_dct_matrix(8)
DCT_MATRIX_T = DCT_MATRIX.T


def dct_2d(block: np.ndarray) -> np.ndarray:
    """
    Apply 2D DCT to an 8x8 block.

    The 2D DCT is computed as: D * block * D^T
    where D is the DCT matrix.

    Args:
        block: 8x8 numpy array (values should be level-shifted by -128)

    Returns:
        DCT coefficients as 8x8 numpy array
    """
    return DCT_MATRIX @ block @ DCT_MATRIX_T


def idct_2d(block: np.ndarray) -> np.ndarray:
    """
    Apply 2D inverse DCT to an 8x8 block.

    The 2D IDCT is computed as: D^T * block * D

    Args:
        block: 8x8 numpy array of DCT coefficients

    Returns:
        Spatial domain values as 8x8 numpy array
    """
    return DCT_MATRIX_T @ block @ DCT_MATRIX


def dct_2d_slow(block: np.ndarray) -> np.ndarray:
    """
    Direct 2D DCT implementation (slower but educational).

    Args:
        block: 8x8 numpy array

    Returns:
        DCT coefficients
    """
    n = 8
    result = np.zeros((n, n), dtype=np.float64)

    def alpha(k):
        return 1 / np.sqrt(n) if k == 0 else np.sqrt(2 / n)

    for u in range(n):
        for v in range(n):
            sum_val = 0
            for x in range(n):
                for y in range(n):
                    sum_val += (block[x, y] *
                               np.cos((2 * x + 1) * u * np.pi / (2 * n)) *
                               np.cos((2 * y + 1) * v * np.pi / (2 * n)))
            result[u, v] = alpha(u) * alpha(v) * sum_val

    return result


# =============================================================================
# Quantization
# =============================================================================

def scale_quantization_table(table: np.ndarray, quality: int) -> np.ndarray:
    """
    Scale quantization table based on quality factor.

    Quality factor works similarly to libjpeg:
    - 50 = no scaling
    - 1-50: scale up (more compression, lower quality)
    - 51-100: scale down (less compression, higher quality)

    Args:
        table: Base quantization table (8x8)
        quality: Quality factor (1-100)

    Returns:
        Scaled quantization table
    """
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    scaled = np.floor((table * scale + 50) / 100)
    scaled = np.clip(scaled, 1, 255)

    return scaled.astype(np.uint8)


def quantize(dct_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """
    Quantize DCT coefficients.

    Args:
        dct_block: 8x8 DCT coefficient block
        quant_table: 8x8 quantization table

    Returns:
        Quantized coefficients as integers
    """
    return np.round(dct_block / quant_table).astype(np.int32)


def dequantize(quant_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
    """
    Dequantize coefficients (for decoder).

    Args:
        quant_block: 8x8 quantized coefficient block
        quant_table: 8x8 quantization table

    Returns:
        Dequantized DCT coefficients
    """
    return (quant_block * quant_table).astype(np.float64)


# =============================================================================
# Zigzag Scanning
# =============================================================================

def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    Convert 8x8 block to 1D array using zigzag scan order.

    Args:
        block: 8x8 numpy array

    Returns:
        1D array of 64 elements in zigzag order
    """
    flat = block.flatten()
    return flat[ZIGZAG_ORDER]


def inverse_zigzag(array: np.ndarray) -> np.ndarray:
    """
    Convert 1D zigzag array back to 8x8 block.

    Args:
        array: 1D array of 64 elements in zigzag order

    Returns:
        8x8 numpy array
    """
    block = np.zeros(64, dtype=array.dtype)
    block[ZIGZAG_ORDER] = array
    return block.reshape(8, 8)


# =============================================================================
# Run-Length Encoding for AC Coefficients
# =============================================================================

@dataclass
class RLESymbol:
    """Represents a run-length encoded AC coefficient"""
    run_length: int  # Number of preceding zeros (0-15)
    size: int        # Number of bits needed for the value
    value: int       # The actual coefficient value


def encode_ac_coefficients(zigzag_coeffs: np.ndarray) -> List[RLESymbol]:
    """
    Run-length encode AC coefficients.

    Args:
        zigzag_coeffs: 64 coefficients in zigzag order (first is DC, rest are AC)

    Returns:
        List of RLE symbols
    """
    ac_coeffs = zigzag_coeffs[1:]  # Skip DC coefficient
    symbols = []
    zero_count = 0

    for coeff in ac_coeffs:
        if coeff == 0:
            zero_count += 1
        else:
            # Handle runs of zeros > 15 with ZRL (zero run length) symbols
            while zero_count > 15:
                symbols.append(RLESymbol(15, 0, 0))  # ZRL symbol (0xF0)
                zero_count -= 16

            # Calculate size (number of bits needed)
            size = bit_length(coeff)
            symbols.append(RLESymbol(zero_count, size, coeff))
            zero_count = 0

    # End of block marker if there are trailing zeros
    if zero_count > 0:
        symbols.append(RLESymbol(0, 0, 0))  # EOB symbol (0x00)

    return symbols


def bit_length(value: int) -> int:
    """
    Calculate the number of bits needed to represent a value.

    Args:
        value: Integer value (can be negative)

    Returns:
        Number of bits needed (0-11 for JPEG)
    """
    if value == 0:
        return 0
    return int(np.floor(np.log2(abs(value)))) + 1


def encode_coefficient(value: int, size: int) -> Tuple[int, int]:
    """
    Encode a coefficient value for Huffman coding.

    Positive values: straight binary representation
    Negative values: one's complement (flip all bits)

    Args:
        value: Coefficient value
        size: Number of bits

    Returns:
        Tuple of (encoded_value, num_bits)
    """
    if size == 0:
        return 0, 0

    if value >= 0:
        return value, size
    else:
        # One's complement for negative values
        return value + (1 << size) - 1, size


# =============================================================================
# JPEG Encoder Class
# =============================================================================

class JPEGEncoder:
    """
    Complete JPEG encoder implementation.

    Supports:
    - Baseline DCT encoding (SOF0)
    - Adjustable quality (1-100)
    - Multiple chroma subsampling modes (4:4:4, 4:2:2, 4:2:0)
    - Standard Huffman tables
    """

    def __init__(self, quality: int = 75,
                 subsampling: SubsamplingMode = SubsamplingMode.MODE_420):
        """
        Initialize the JPEG encoder.

        Args:
            quality: Quality factor (1-100, default 75)
            subsampling: Chroma subsampling mode (default 4:2:0)
        """
        self.quality = quality
        self.subsampling = subsampling

        # Scale quantization tables
        self.lum_quant = scale_quantization_table(LUMINANCE_QUANTIZATION_TABLE, quality)
        self.chr_quant = scale_quantization_table(CHROMINANCE_QUANTIZATION_TABLE, quality)

        # Build Huffman tables
        self.dc_lum_huffman = HuffmanTable(DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES)
        self.dc_chr_huffman = HuffmanTable(DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES)
        self.ac_lum_huffman = HuffmanTable(AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES)
        self.ac_chr_huffman = HuffmanTable(AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES)

    def encode(self, image: np.ndarray) -> bytes:
        """
        Encode an RGB image to JPEG format.

        Args:
            image: RGB image as (H, W, 3) numpy array with values 0-255

        Returns:
            JPEG file as bytes
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be RGB image with shape (H, W, 3)")

        original_height, original_width = image.shape[:2]

        # Convert to YCbCr
        ycbcr = rgb_to_ycbcr(image)

        # Separate channels
        y_channel = ycbcr[:, :, 0]
        cb_channel = ycbcr[:, :, 1]
        cr_channel = ycbcr[:, :, 2]

        # Apply chroma subsampling
        cb_subsampled = subsample_channel(cb_channel, self.subsampling, is_chroma=True)
        cr_subsampled = subsample_channel(cr_channel, self.subsampling, is_chroma=True)

        # Pad channels to multiples of 8 (or 16 for subsampled chroma)
        block_size = 8
        if self.subsampling == SubsamplingMode.MODE_420:
            y_channel = pad_to_multiple(y_channel, 16)
            cb_subsampled = pad_to_multiple(cb_subsampled, 8)
            cr_subsampled = pad_to_multiple(cr_subsampled, 8)
        elif self.subsampling == SubsamplingMode.MODE_422:
            y_channel = pad_to_multiple(y_channel, 16)
            cb_subsampled = pad_to_multiple(cb_subsampled, 8)
            cr_subsampled = pad_to_multiple(cr_subsampled, 8)
        else:
            y_channel = pad_to_multiple(y_channel, 8)
            cb_subsampled = pad_to_multiple(cb_subsampled, 8)
            cr_subsampled = pad_to_multiple(cr_subsampled, 8)

        # Split into 8x8 blocks
        y_blocks = split_into_blocks(y_channel, block_size)
        cb_blocks = split_into_blocks(cb_subsampled, block_size)
        cr_blocks = split_into_blocks(cr_subsampled, block_size)

        # Process blocks
        y_encoded = self._process_channel_blocks(y_blocks, is_luminance=True)
        cb_encoded = self._process_channel_blocks(cb_blocks, is_luminance=False)
        cr_encoded = self._process_channel_blocks(cr_blocks, is_luminance=False)

        # Encode scan data
        scan_data = self._encode_scan_data(y_encoded, cb_encoded, cr_encoded,
                                           y_blocks.shape[:2], cb_blocks.shape[:2])

        # Build JPEG file
        return self._build_jpeg_file(original_width, original_height, scan_data)

    def _process_channel_blocks(self, blocks: np.ndarray,
                                is_luminance: bool) -> List[np.ndarray]:
        """
        Process all blocks of a channel through DCT and quantization.

        Args:
            blocks: Array of 8x8 blocks with shape (H, W, 8, 8)
            is_luminance: True for Y channel, False for Cb/Cr

        Returns:
            List of quantized coefficient arrays in zigzag order
        """
        quant_table = self.lum_quant if is_luminance else self.chr_quant
        num_blocks_h, num_blocks_w = blocks.shape[:2]
        encoded_blocks = []

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = blocks[i, j].astype(np.float64)

                # Level shift (subtract 128)
                block = block - 128

                # Apply DCT
                dct_block = dct_2d(block)

                # Quantize
                quant_block = quantize(dct_block, quant_table)

                # Zigzag scan
                zigzag = zigzag_scan(quant_block)

                encoded_blocks.append(zigzag)

        return encoded_blocks

    def _encode_scan_data(self, y_blocks: List[np.ndarray],
                          cb_blocks: List[np.ndarray],
                          cr_blocks: List[np.ndarray],
                          y_shape: Tuple[int, int],
                          c_shape: Tuple[int, int]) -> bytes:
        """
        Encode all blocks into compressed scan data.

        Args:
            y_blocks: List of Y channel quantized blocks
            cb_blocks: List of Cb channel quantized blocks
            cr_blocks: List of Cr channel quantized blocks
            y_shape: (num_blocks_h, num_blocks_w) for Y channel
            c_shape: (num_blocks_h, num_blocks_w) for chroma channels

        Returns:
            Compressed scan data as bytes
        """
        bit_writer = BitWriter()

        # Track DC predictors for differential encoding
        dc_pred = [0, 0, 0]  # Y, Cb, Cr

        y_h, y_w = y_shape
        c_h, c_w = c_shape

        # Determine MCU structure based on subsampling
        if self.subsampling == SubsamplingMode.MODE_420:
            # MCU = 4 Y blocks + 1 Cb block + 1 Cr block
            mcu_y_h, mcu_y_w = 2, 2
        elif self.subsampling == SubsamplingMode.MODE_422:
            # MCU = 2 Y blocks + 1 Cb block + 1 Cr block
            mcu_y_h, mcu_y_w = 1, 2
        else:
            # MCU = 1 Y block + 1 Cb block + 1 Cr block
            mcu_y_h, mcu_y_w = 1, 1

        num_mcu_h = c_h
        num_mcu_w = c_w

        y_idx = 0
        c_idx = 0

        for mcu_y in range(num_mcu_h):
            for mcu_x in range(num_mcu_w):
                # Encode Y blocks in MCU
                for dy in range(mcu_y_h):
                    for dx in range(mcu_y_w):
                        block_y = mcu_y * mcu_y_h + dy
                        block_x = mcu_x * mcu_y_w + dx
                        if block_y < y_h and block_x < y_w:
                            idx = block_y * y_w + block_x
                            dc_pred[0] = self._encode_block(
                                bit_writer, y_blocks[idx], dc_pred[0],
                                self.dc_lum_huffman, self.ac_lum_huffman
                            )

                # Encode Cb block
                if c_idx < len(cb_blocks):
                    dc_pred[1] = self._encode_block(
                        bit_writer, cb_blocks[c_idx], dc_pred[1],
                        self.dc_chr_huffman, self.ac_chr_huffman
                    )

                # Encode Cr block
                if c_idx < len(cr_blocks):
                    dc_pred[2] = self._encode_block(
                        bit_writer, cr_blocks[c_idx], dc_pred[2],
                        self.dc_chr_huffman, self.ac_chr_huffman
                    )

                c_idx += 1

        bit_writer.flush()
        return bit_writer.get_data()

    def _encode_block(self, writer: BitWriter, block: np.ndarray,
                      dc_pred: int, dc_huffman: HuffmanTable,
                      ac_huffman: HuffmanTable) -> int:
        """
        Encode a single block using Huffman coding.

        Args:
            writer: BitWriter instance
            block: Quantized coefficients in zigzag order
            dc_pred: Previous DC value for differential encoding
            dc_huffman: DC Huffman table
            ac_huffman: AC Huffman table

        Returns:
            Current DC value (for next block's prediction)
        """
        # Encode DC coefficient (differential)
        dc_value = int(block[0])
        dc_diff = dc_value - dc_pred

        dc_size = bit_length(dc_diff)
        dc_code = dc_huffman.encode(dc_size)
        writer.write_bits(dc_code.code, dc_code.length)

        if dc_size > 0:
            encoded_val, num_bits = encode_coefficient(dc_diff, dc_size)
            writer.write_bits(encoded_val, num_bits)

        # Encode AC coefficients with RLE
        ac_symbols = encode_ac_coefficients(block)

        for symbol in ac_symbols:
            # Huffman symbol = (run_length << 4) | size
            huffman_symbol = (symbol.run_length << 4) | symbol.size
            ac_code = ac_huffman.encode(huffman_symbol)
            writer.write_bits(ac_code.code, ac_code.length)

            if symbol.size > 0:
                encoded_val, num_bits = encode_coefficient(symbol.value, symbol.size)
                writer.write_bits(encoded_val, num_bits)

        return dc_value

    def _build_jpeg_file(self, width: int, height: int, scan_data: bytes) -> bytes:
        """
        Build complete JPEG file with all markers and headers.

        Args:
            width: Image width
            height: Image height
            scan_data: Compressed scan data

        Returns:
            Complete JPEG file as bytes
        """
        output = bytearray()

        # SOI (Start of Image)
        output.extend(struct.pack('>H', Marker.SOI))

        # APP0 (JFIF marker)
        output.extend(self._build_app0_segment())

        # DQT (Quantization tables)
        output.extend(self._build_dqt_segment(0, self.lum_quant))
        output.extend(self._build_dqt_segment(1, self.chr_quant))

        # SOF0 (Start of Frame - Baseline DCT)
        output.extend(self._build_sof0_segment(width, height))

        # DHT (Huffman tables)
        output.extend(self._build_dht_segment(0, 0, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES))
        output.extend(self._build_dht_segment(0, 1, DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES))
        output.extend(self._build_dht_segment(1, 0, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES))
        output.extend(self._build_dht_segment(1, 1, AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES))

        # SOS (Start of Scan)
        output.extend(self._build_sos_segment())

        # Compressed image data
        output.extend(scan_data)

        # EOI (End of Image)
        output.extend(struct.pack('>H', Marker.EOI))

        return bytes(output)

    def _build_app0_segment(self) -> bytes:
        """Build JFIF APP0 segment"""
        segment = bytearray()
        segment.extend(struct.pack('>H', Marker.APP0))

        # Segment content
        content = bytearray()
        content.extend(b'JFIF\x00')  # Identifier
        content.extend(struct.pack('>BB', 1, 1))  # Version 1.1
        content.append(0)  # Units (0 = no units)
        content.extend(struct.pack('>HH', 1, 1))  # X and Y density
        content.extend(struct.pack('>BB', 0, 0))  # No thumbnail

        # Length includes itself (2 bytes)
        segment.extend(struct.pack('>H', len(content) + 2))
        segment.extend(content)

        return bytes(segment)

    def _build_dqt_segment(self, table_id: int, quant_table: np.ndarray) -> bytes:
        """Build DQT segment for a quantization table"""
        segment = bytearray()
        segment.extend(struct.pack('>H', Marker.DQT))

        # Precision (0 = 8-bit) and table ID
        precision_id = (0 << 4) | table_id

        # Table data in zigzag order
        table_data = bytearray([precision_id])
        flat_table = quant_table.flatten()
        for idx in ZIGZAG_ORDER:
            table_data.append(int(flat_table[idx]))

        # Length
        segment.extend(struct.pack('>H', len(table_data) + 2))
        segment.extend(table_data)

        return bytes(segment)

    def _build_sof0_segment(self, width: int, height: int) -> bytes:
        """Build SOF0 (Start of Frame) segment"""
        segment = bytearray()
        segment.extend(struct.pack('>H', Marker.SOF0))

        content = bytearray()
        content.append(8)  # Sample precision (8 bits)
        content.extend(struct.pack('>HH', height, width))
        content.append(3)  # Number of components

        # Component specifications
        if self.subsampling == SubsamplingMode.MODE_420:
            # Y: 2x2 sampling, Cb/Cr: 1x1 sampling
            content.extend([1, 0x22, 0])  # Y: component 1, 2x2, quant table 0
            content.extend([2, 0x11, 1])  # Cb: component 2, 1x1, quant table 1
            content.extend([3, 0x11, 1])  # Cr: component 3, 1x1, quant table 1
        elif self.subsampling == SubsamplingMode.MODE_422:
            # Y: 2x1 sampling, Cb/Cr: 1x1 sampling
            content.extend([1, 0x21, 0])  # Y: component 1, 2x1, quant table 0
            content.extend([2, 0x11, 1])  # Cb: component 2, 1x1, quant table 1
            content.extend([3, 0x11, 1])  # Cr: component 3, 1x1, quant table 1
        else:
            # 4:4:4: No subsampling
            content.extend([1, 0x11, 0])  # Y: component 1, 1x1, quant table 0
            content.extend([2, 0x11, 1])  # Cb: component 2, 1x1, quant table 1
            content.extend([3, 0x11, 1])  # Cr: component 3, 1x1, quant table 1

        segment.extend(struct.pack('>H', len(content) + 2))
        segment.extend(content)

        return bytes(segment)

    def _build_dht_segment(self, table_class: int, table_id: int,
                           bits: List[int], values: List[int]) -> bytes:
        """Build DHT (Huffman table) segment"""
        segment = bytearray()
        segment.extend(struct.pack('>H', Marker.DHT))

        content = bytearray()
        # Table class (0=DC, 1=AC) and ID
        content.append((table_class << 4) | table_id)

        # Number of codes of each length (1-16)
        content.extend(bits)

        # Symbol values
        content.extend(values)

        segment.extend(struct.pack('>H', len(content) + 2))
        segment.extend(content)

        return bytes(segment)

    def _build_sos_segment(self) -> bytes:
        """Build SOS (Start of Scan) segment"""
        segment = bytearray()
        segment.extend(struct.pack('>H', Marker.SOS))

        content = bytearray()
        content.append(3)  # Number of components

        # Component selectors (component ID, DC/AC table IDs)
        content.extend([1, 0x00])  # Y: DC table 0, AC table 0
        content.extend([2, 0x11])  # Cb: DC table 1, AC table 1
        content.extend([3, 0x11])  # Cr: DC table 1, AC table 1

        # Spectral selection and successive approximation
        content.append(0)   # Start of spectral selection (0 for baseline)
        content.append(63)  # End of spectral selection (63 for baseline)
        content.append(0)   # Successive approximation (0 for baseline)

        segment.extend(struct.pack('>H', len(content) + 2))
        segment.extend(content)

        return bytes(segment)


# =============================================================================
# JPEG Decoder Class (Basic Implementation)
# =============================================================================

class JPEGDecoder:
    """
    Basic JPEG decoder for verification purposes.
    Note: This is a simplified decoder that only handles baseline JPEG files
    encoded by this encoder.
    """

    def __init__(self):
        self.quant_tables = {}
        self.huffman_tables = {}
        self.width = 0
        self.height = 0
        self.components = []

    def decode(self, data: bytes) -> np.ndarray:
        """
        Decode JPEG data to RGB image.

        Args:
            data: JPEG file as bytes

        Returns:
            RGB image as numpy array
        """
        # This is a placeholder for a full decoder implementation
        # For now, we rely on external libraries for verification
        raise NotImplementedError("Full decoder not implemented. Use PIL/Pillow for decoding.")


# =============================================================================
# Utility Functions
# =============================================================================

def encode_image(image: np.ndarray, quality: int = 75,
                 subsampling: SubsamplingMode = SubsamplingMode.MODE_420) -> bytes:
    """
    Convenience function to encode an RGB image to JPEG.

    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
        quality: Quality factor (1-100)
        subsampling: Chroma subsampling mode

    Returns:
        JPEG file as bytes
    """
    encoder = JPEGEncoder(quality=quality, subsampling=subsampling)
    return encoder.encode(image)


def save_jpeg(image: np.ndarray, filepath: str, quality: int = 75,
              subsampling: SubsamplingMode = SubsamplingMode.MODE_420):
    """
    Encode and save an RGB image as JPEG file.

    Args:
        image: RGB image as numpy array (H, W, 3) with values 0-255
        filepath: Output file path
        quality: Quality factor (1-100)
        subsampling: Chroma subsampling mode
    """
    jpeg_data = encode_image(image, quality, subsampling)
    with open(filepath, 'wb') as f:
        f.write(jpeg_data)


def create_test_image(width: int = 256, height: int = 256) -> np.ndarray:
    """
    Create a test image with gradients and patterns.

    Args:
        width: Image width
        height: Image height

    Returns:
        RGB test image as numpy array
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create color gradients
    for y in range(height):
        for x in range(width):
            # Red gradient (left to right)
            image[y, x, 0] = int(255 * x / width)
            # Green gradient (top to bottom)
            image[y, x, 1] = int(255 * y / height)
            # Blue checkerboard pattern
            if ((x // 32) + (y // 32)) % 2 == 0:
                image[y, x, 2] = 200
            else:
                image[y, x, 2] = 55

    return image


def analyze_jpeg_quality(original: np.ndarray, compressed: np.ndarray) -> dict:
    """
    Analyze quality metrics between original and compressed images.

    Args:
        original: Original RGB image
        compressed: Compressed/decompressed RGB image

    Returns:
        Dictionary with quality metrics (MSE, PSNR, etc.)
    """
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)

    # Mean Squared Error
    mse = np.mean((original - compressed) ** 2)

    # Peak Signal-to-Noise Ratio
    if mse > 0:
        psnr = 10 * np.log10((255 ** 2) / mse)
    else:
        psnr = float('inf')

    # Mean Absolute Error
    mae = np.mean(np.abs(original - compressed))

    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main function for command-line usage"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='JPEG Encoder - Comprehensive implementation from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s input.png output.jpg
  %(prog)s input.png output.jpg -q 90
  %(prog)s input.png output.jpg -q 50 -s 422
  %(prog)s --test  # Run with test image
        '''
    )

    parser.add_argument('input', nargs='?', help='Input image file (PNG, BMP, etc.)')
    parser.add_argument('output', nargs='?', help='Output JPEG file')
    parser.add_argument('-q', '--quality', type=int, default=75,
                        help='Quality factor (1-100, default: 75)')
    parser.add_argument('-s', '--subsampling', choices=['444', '422', '420'],
                        default='420', help='Chroma subsampling (default: 420)')
    parser.add_argument('--test', action='store_true',
                        help='Run with a test image')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Map subsampling argument
    subsampling_map = {
        '444': SubsamplingMode.MODE_444,
        '422': SubsamplingMode.MODE_422,
        '420': SubsamplingMode.MODE_420
    }
    subsampling = subsampling_map[args.subsampling]

    if args.test:
        # Generate and encode test image
        print("Generating test image...")
        test_image = create_test_image(256, 256)

        output_path = args.output or 'test_output.jpg'
        print(f"Encoding with quality={args.quality}, subsampling={args.subsampling}...")

        save_jpeg(test_image, output_path, args.quality, subsampling)
        print(f"Saved to: {output_path}")

        # Try to verify with PIL if available
        try:
            from PIL import Image

            # Re-open and check
            decoded = np.array(Image.open(output_path).convert('RGB'))
            metrics = analyze_jpeg_quality(test_image, decoded)

            print(f"\nQuality metrics:")
            print(f"  MSE:  {metrics['mse']:.2f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  MAE:  {metrics['mae']:.2f}")
        except ImportError:
            print("\nInstall PIL/Pillow to verify output: pip install Pillow")

        return 0

    if not args.input or not args.output:
        parser.print_help()
        return 1

    # Load input image
    try:
        from PIL import Image

        print(f"Loading: {args.input}")
        pil_image = Image.open(args.input).convert('RGB')
        image = np.array(pil_image)

        if args.verbose:
            print(f"  Size: {image.shape[1]}x{image.shape[0]}")

        print(f"Encoding with quality={args.quality}, subsampling={args.subsampling}...")
        save_jpeg(image, args.output, args.quality, subsampling)

        import os
        file_size = os.path.getsize(args.output)
        print(f"Saved to: {args.output} ({file_size:,} bytes)")

        if args.verbose:
            # Verify and show metrics
            decoded = np.array(Image.open(args.output).convert('RGB'))
            metrics = analyze_jpeg_quality(image, decoded)
            print(f"\nQuality metrics:")
            print(f"  MSE:  {metrics['mse']:.2f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  MAE:  {metrics['mae']:.2f}")

        return 0

    except ImportError:
        print("Error: PIL/Pillow required for loading images.")
        print("Install with: pip install Pillow")
        return 1
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
