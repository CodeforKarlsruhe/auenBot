from sqlalchemy import create_engine, Column, Integer, String, func, text, desc, inspect
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
import numpy as np
from pgvector.sqlalchemy import Vector

import json
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

# Normalize each vector to unit length.
# Axis=1 means we take the norm across each row (each vector).
norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
vectors_normalized = vectors / norms

print(f"Number of vectors: {len(vectors_normalized)}, shape: {vectors_normalized.shape}")


# Database connection
import private as pr
engine = create_engine(f"postgresql+psycopg2://{pr.PG_USER}:{pr.PG_PWD}@{pr.PG_HOST}/{pr.PG_NAME}")
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define the table
class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    intent = Column(String)
    value = Column(Vector(1024))  # 1024-dimensional vector


# Drop the table if it exists
inspector = inspect(engine)
if inspector.has_table("embeddings"):
    Embedding.__table__.drop(engine)


# Create the table
Base.metadata.create_all(engine)



embs  = [Embedding(intent=intent, value=vec) for intent, vec in zip(intents, vectors_normalized)]

# Insert sample data
session = Session()
session.add_all(embs)

session.commit()

session.close()



