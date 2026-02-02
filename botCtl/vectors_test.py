# --------------------------------------------------------------
# demo_python.py
# --------------------------------------------------------------
import vectors_pb2

import numpy as np

import json

# ------------------------------------------------------------------
# Helper utilities (optional but handy)
# ------------------------------------------------------------------
MAX_TEXT_LEN = 24          # our “fixed‑length” constraint for the string
VECTOR_LENGTH = 1024          # the required size N for every int16 vector


def _check_text(txt: str) -> str:
    """Enforce the maximum length for the string field."""
    if len(txt.encode('utf-8')) > MAX_TEXT_LEN:
        raise ValueError(f"text exceeds {MAX_TEXT_LEN} bytes")
    return txt


def _check_int16(val: int) -> int:
    """Validate that a Python int fits into signed int16."""
    if not (-2**15 <= val < 2**15):
        raise ValueError(f"{val} is out of int16 range")
    return val


# ------------------------------------------------------------------
# Build a Container instance
# ------------------------------------------------------------------
def build_container(intents, embeddings) -> vectors_pb2.Container:
    container = vectors_pb2.Container()

    # --------------------
    # 1) Fill list1 (pairs)
    # --------------------
    current_id = None
    current_cnt = 0
    total_cnt = 0
    for intent in intents:
        if intent != current_id:
            if current_id is not None:
                pair = container.intents.add()
                pair.id = _check_text(current_id)          # enforce fixed‑length rule
                pair.occurance = current_cnt
            current_id = intent
            current_cnt = 1
        else:
            current_cnt += 1
        total_cnt += 1  
    # final
    pair = container.intents.add()
    pair.id = _check_text(current_id)          # enforce fixed‑length rule
    pair.occurance = current_cnt

    print(f"Built intents: {len(container.intents)}, total occurrences: {total_cnt}")
    # --------------------

    # 2) Fill list2 (int16 vectors)
    for vec in embeddings:
        if len(vec) != VECTOR_LENGTH:
            raise ValueError(f"Each vector must have exactly {VECTOR_LENGTH} elements")
        vec = vec.astype(np.int32)  # use int32 for intermediate to avoid overflow
        iv = container.vectors.add()
        # Store as int32 but validate int16 range
        iv.vector.extend([_check_int16(v) for v in vec])
    print(f"Built vectors: {len(container.vectors)}")
    # --------------------


    # Store the explicit count (optional)
    container.intent_count = len(container.intents)
    container.vector_count = len(container.vectors)

    return container


# ------------------------------------------------------------------
# Serialisation / Deserialisation
# ------------------------------------------------------------------
def serialise_deserialise(intents, embeddings):
    # Build the message

    msg = build_container(intents, embeddings)

    # ---- Serialise to bytes (ready to send over a socket, write to file, etc.) ----
    payload = msg.SerializeToString()
    with open("vectors.bin", "wb") as f:
        f.write(payload)

    print("\n--- Serialized payload (hex) ---")
    print("Size:",len(payload))
    # print(payload.hex())

    # ---- Deserialise back into a fresh object ----
    recovered = vectors_pb2.Container()
    recovered.ParseFromString(payload)

    # Pretty‑print the recovered structure
    print("\n--- Recovered Container ---")
    print(f"intent_count: {recovered.intent_count}")
    for i, p in enumerate(recovered.intents, start=1):
        print(f"  Pair {i}: id='{p.id}', occurance={p.occurance}")

    print(f"\nvector_count: {recovered.vector_count}")
    for i, vec in enumerate(recovered.vectors, start=1):
        # Cast back to Python int16 for display (just for illustration)
        int16_vals = [int(v) for v in vec.vector]  # they are already ints
        #print(f"  Vector {i}: {int16_vals}")

    return [v for v in recovered.vectors]
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Sample data
    import os
    filename = "../rawData/intent_vectors.json"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    with open(filename, 'r') as f:
        raw = json.load(f)

    if raw.get("vectors") is None:
        raise ValueError("Invalid file format: 'vectors' key not found.")
    vectors = np.array(raw["vectors"], dtype=np.float32)
    if raw.get("intents") is None:
        raise ValueError("Invalid file format: 'intents' key not found.")
    intents = raw["intents"]

    print(f"Number of vectors: {len(vectors)}, shape: {vectors.shape}")

    # Normalize each vector to unit length.
    # Axis=1 means we take the norm across each row (each vector).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    vectors_normalized = vectors / norms
    vectors_normalized_int = (vectors_normalized * 32767).astype(np.int32)
    vecs = serialise_deserialise(intents, vectors_normalized_int)
    print(f"\nTotal recovered vectors: {len(vecs)}")
    vecs_float = []
    for v in vecs:
        intVals = np.array([int(vi) for vi in v.vector], dtype=np.int32)
        float_vals = intVals.astype(np.float32) / 32767.0    
        vecs_float.append(float_vals)

    vecs_float = np.array(vecs_float, dtype=np.float32)

    # Verify that the recovered vectors match the original normalized vectors

    differences = np.abs(vectors_normalized - vecs_float)
    max_difference = np.max(differences)
    print(f"\nMax difference between original and recovered vectors: {max_difference}")

