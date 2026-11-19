import math
import struct
import zlib
from pathlib import Path
from PIL import Image

MAGIC = b"PNGLMIMG"

def encode_png(binary_path, png_path):

    # read bin
    data = Path(binary_path).read_bytes()

    # store original length + checksum
    header = (MAGIC + struct.pack("<Q", len(data)) + struct.pack("<I", zlib.crc32(data)))
    payload = header + data

   
    pixels_needed = math.ceil(len(payload) / 3)

    # square image
    width = math.ceil(math.sqrt(pixels_needed))
    height = math.ceil(pixels_needed / width)

    # unused = 0
    capacity = width * height * 3
    payload += bytes(capacity - len(payload))

    # rgb
    image = Image.frombytes("RGB", (width, height), payload)
    image.save(png_path, compress_level=1)