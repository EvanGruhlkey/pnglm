import json
import struct
import numpy as np
import torch

MAGIC = b"PNGLM001"

def save_model(model, path):

    # get tensors
    state = model.state_dict()

    metadata = {
        "architecture" : "gpt2",
        "config" : model.config.to_dict(),
        "tensors" : {}
    }
    offset = 0

    # describe tensors
    for name, tensor in state.items():

        # bytes :)
        size = tensor.numel() * 4

        metadata["tensors"][name] = {
            "shape" : list(tensor.shape),
            "dtype" : "float32",
            "offset" : offset,
            "nbytes" : size
        }

        offset += size

    # metadata to utf-8 bytes
    header = json.dumps(metadata).encode("utf-8")

    with open(path, "wb") as f:

        # iden
        f.write(MAGIC)

        # header size
        f.write(struct.pack("<Q", len(header)))

        # metadata
        f.write(header)

        # weights
        for t in state.values():
            array = t.detach().cpu().contiguous().numpy()
            array = array.astype("<f4", copy=False)
            f.write(array.tobytes())

    print(f"file saved to: {path}")

    def load_model(path):
        with open(path, "rb") as f:
            if f.read(8) != MAGIC:
                raise ValueError("invalid file")

            # metadata length
            header_length = struct.uncpack("<Q", f.read(8))[0]

            # metadata
            metadata = json.loads(f.read(header_length).decode("utf-8"))

            data_start = f.tell()
            tensors = {}

            # reconstruct tensors
            for name, info in metadata["tensors"].items():
                # prob need error handling here

                f.seek(data_start + info["offset"])
                raw = f.read(info["nbytes"])

                # bytes to tensors
                array = np.frombuffer(raw, dtype="<f4").copy()
                array = array.reshape(info["shape"])

                tensors[name] = torch.from_numpy(array)
            return tensors, metadata
            