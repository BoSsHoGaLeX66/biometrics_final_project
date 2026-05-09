from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import serial
import time

from src.app.database import ModelType, StoredUserEmbedding, connect_database, get_all_user_embeddings

ser = serial.Serial("/dev/cu.usbmodem1101", 115200, timeout=1)
time.sleep(2)

DEFAULT_DATABASE_PATH = Path("data/speaker_embeddings.db")
IDENTIFICATION_THRESHOLD = 0.7
_database_path = DEFAULT_DATABASE_PATH
_model_type: ModelType = "homegrown"


@dataclass(frozen=True)
class IdentifiedUser:
    """
    User identity returned after a successful voice embedding match.

    Attributes:
        user_id: Primary key of the matched registered user.
        name: Registered user name.
        target: Whether the matched user is marked as a target.
        similarity: Cosine similarity between the input embedding and the stored embedding.
        user_embedding_id: Stored embedding row that produced the match.
    """

    user_id: int
    name: str
    target: bool
    similarity: float
    user_embedding_id: int


def set_database_path(database_path: str | Path) -> None:
    """
    Configure the database path used by identify.

    Args:
        database_path: SQLite database path containing enrolled speaker embeddings.
    """
    global _database_path
    _database_path = Path(database_path)


def set_model_type(model_type: ModelType) -> None:
    """
    Configure the selected model type used to filter candidate embeddings.

    Args:
        model_type: Active speaker recognition model family.

    Raises:
        ValueError: If the model type is invalid.
    """
    if model_type not in ("homegrown", "finetuned"):
        raise ValueError("model_type must be either 'homegrown' or 'finetuned'")

    global _model_type
    _model_type = model_type


def identify(features: torch.Tensor) -> IdentifiedUser | None:
    PORT = "/dev/cu.usbmodem1101"
    BAUD = 115200

    """
    Identify a registered user by comparing a voice embedding to stored embeddings.

    Args:
        features: Speaker embedding tensor produced by the inference model.

    Returns:
        Matched user when cosine similarity is above 0.9, otherwise None.
    """
    if not _database_path.exists():
        print(f"No speaker database found at {_database_path}. Identity: unknown. Target: no.")
        return None

    query_embedding = _flatten_embedding(features)
    connection = connect_database(_database_path)
    try:
        stored_embeddings = get_all_user_embeddings(connection, _model_type)
    finally:
        connection.close()

    best_match = _find_best_match(query_embedding, stored_embeddings)
    if best_match is None:
        print("Identity: unknown. Target: no.")
        return None
    elif best_match.similarity < IDENTIFICATION_THRESHOLD:
        print("Did not meet threshold for identity.")
        print(f"Identity: {best_match.name}. Target: no. Similarity: {best_match.similarity:.3f}.")
        return None

    target_status = "yes" if best_match.target else "no"

    ser.write(b"HIGH\n")
    print("GPIO HIGH")
    time.sleep(3)
    ser.write(b"LOW\n")


    print(f"Identity: {best_match.name}. Target: {target_status}. Similarity: {best_match.similarity:.3f}.")
    return best_match


def _flatten_embedding(features: torch.Tensor) -> torch.Tensor:
    """
    Convert model output features into a normalized one-dimensional embedding tensor.

    Args:
        features: Speaker embedding tensor produced by a model.

    Returns:
        Normalized one-dimensional embedding tensor on CPU.
    """
    embedding = features.detach().float().cpu().reshape(-1)
    return F.normalize(embedding, dim=0)


def _find_best_match(
    query_embedding: torch.Tensor,
    stored_embeddings: list[StoredUserEmbedding],
) -> IdentifiedUser | None:
    """
    Find the stored speaker embedding with the highest cosine similarity.

    Args:
        query_embedding: Normalized speaker embedding from the current audio sample.
        stored_embeddings: Enrolled speaker embeddings from the database.

    Returns:
        Best matching user, or None when there are no compatible embeddings.
    """
    best_match: IdentifiedUser | None = None
    for stored_embedding in stored_embeddings:
        stored_tensor = torch.tensor(stored_embedding.embedding, dtype=torch.float32)
        if stored_tensor.numel() != query_embedding.numel():
            continue

        similarity = float(F.cosine_similarity(query_embedding, stored_tensor, dim=0).item())
        if best_match is None or similarity > best_match.similarity:
            best_match = IdentifiedUser(
                user_id=stored_embedding.user_id,
                name=stored_embedding.name,
                target=stored_embedding.target,
                similarity=similarity,
                user_embedding_id=stored_embedding.user_embedding_id,
            )
    return best_match
