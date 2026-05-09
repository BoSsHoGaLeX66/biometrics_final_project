import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ModelType = Literal["homegrown", "finetuned"]

SCHEMA_SQL = """
-- Users in your biometric database
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    target BOOLEAN NOT NULL DEFAULT 0
);

-- Stores the actual embedding vectors using sqlite-vec
CREATE VIRTUAL TABLE embeddings USING vec0(
    embedding FLOAT[256]
);
CREATE TABLE user_embeddings (
    user_embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    embedding_id INTEGER NOT NULL,
    model_type TEXT NOT NULL CHECK (
        model_type IN ('homegrown', 'finetuned')
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(user_id)
        ON DELETE CASCADE
);
"""

EMBEDDING_DIM = 256


@dataclass(frozen=True)
class StoredUserEmbedding:
    """
    Stored speaker embedding with the user metadata needed for identification.

    Attributes:
        user_id: Primary key of the registered user.
        name: Registered user name.
        target: Whether the registered user is marked as a target.
        user_embedding_id: Primary key of the user_embeddings row.
        embedding: Speaker embedding values read from the vector database.
    """

    user_id: int
    name: str
    target: bool
    user_embedding_id: int
    embedding: tuple[float, ...]


def _get_sqlite_vec():
    """
    Import sqlite-vec when vector database functionality is used.

    Returns:
        Imported sqlite_vec module.

    Raises:
        RuntimeError: If sqlite-vec is not installed in the active environment.
    """
    try:
        import sqlite_vec
    except ModuleNotFoundError as exc:
        raise RuntimeError("sqlite-vec is required to use the biometric embedding database") from exc
    return sqlite_vec


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.enable_load_extension(True)
    _get_sqlite_vec().load(connection)
    connection.enable_load_extension(False)

    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """
    Create the biometric database tables using the project SQL schema.

    Args:
        connection: SQLite connection with sqlite-vec already loaded.
    """
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def get_user_id(connection: sqlite3.Connection, name: str) -> int | None:
    """
    Look up a registered user by name.

    Args:
        connection: SQLite database connection.
        name: User name to search for.

    Returns:
        Matching user ID, or None when the user does not exist.
    """
    row = connection.execute(
        "SELECT user_id FROM users WHERE name = ? ORDER BY user_id LIMIT 1",
        (name,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


def ensure_user(
    connection: sqlite3.Connection,
    name: str,
    target: bool = False,
) -> int:
    """
    Return an existing user ID or insert the user when no match exists.

    Args:
        connection: SQLite database connection.
        name: User name to find or create.
        target: Whether this user is marked as a target in the users table.

    Returns:
        Existing or newly created user ID.
    """
    user_id = get_user_id(connection, name)
    if user_id is not None:
        return user_id

    cursor = connection.execute(
        "INSERT INTO users (name, target) VALUES (?, ?)",
        (name, int(target)),
    )
    return int(cursor.lastrowid)


def insert_user_embedding(
    connection: sqlite3.Connection,
    name: str,
    embedding: Iterable[float],
    model_type: ModelType,
    target: bool = False,
) -> int:
    """
    Insert a model embedding and link it to a user record.

    The user is looked up by name first. If the user does not exist, a new row is
    inserted into users before the vector is stored and linked in user_embeddings.

    Args:
        connection: SQLite database connection.
        name: User name associated with the embedding.
        embedding: Speaker embedding containing exactly 256 float values.
        model_type: Model family that produced the embedding.
        target: Whether a newly inserted user should be marked as a target.

    Returns:
        New user_embeddings row ID linking the user to the stored vector.

    Raises:
        ValueError: If the model type is invalid or the embedding dimension is not 256.
    """
    if model_type not in ("homegrown", "finetuned"):
        raise ValueError("model_type must be either 'homegrown' or 'finetuned'")

    serialized_embedding = serialize_embedding(embedding)

    with connection:
        user_id = ensure_user(connection, name=name, target=target)
        embedding_cursor = connection.execute(
            "INSERT INTO embeddings (embedding) VALUES (?)",
            (serialized_embedding,),
        )
        embedding_id = int(embedding_cursor.lastrowid)
        link_cursor = connection.execute(
            """
            INSERT INTO user_embeddings (user_id, embedding_id, model_type)
            VALUES (?, ?, ?)
            """,
            (user_id, embedding_id, model_type),
        )
        return int(link_cursor.lastrowid)


def get_all_user_embeddings(connection: sqlite3.Connection, model_type: ModelType) -> list[StoredUserEmbedding]:
    """
    Retrieve stored user embeddings for one model and their associated user metadata.

    Args:
        connection: SQLite database connection with sqlite-vec loaded.
        model_type: Model family whose stored embeddings should be returned.

    Returns:
        List of matching model embeddings joined with user identity and target status.

    Raises:
        ValueError: If the model type is invalid.
    """
    if model_type not in ("homegrown", "finetuned"):
        raise ValueError("model_type must be either 'homegrown' or 'finetuned'")

    rows = connection.execute(
        """
        SELECT
            u.user_id,
            u.name,
            u.target,
            ue.user_embedding_id,
            e.embedding
        FROM user_embeddings AS ue
        JOIN users AS u
            ON u.user_id = ue.user_id
        JOIN embeddings AS e
            ON e.rowid = ue.embedding_id
        WHERE ue.model_type = ?
        ORDER BY ue.user_embedding_id
        """,
        (model_type,),
    ).fetchall()
    return [
        StoredUserEmbedding(
            user_id=int(row[0]),
            name=str(row[1]),
            target=bool(row[2]),
            user_embedding_id=int(row[3]),
            embedding=deserialize_embedding(row[4]),
        )
        for row in rows
    ]


def serialize_embedding(embedding: Iterable[float]) -> bytes:
    """
    Validate and serialize a 256-dimensional embedding for sqlite-vec storage.

    Args:
        embedding: Iterable containing speaker embedding float values.

    Returns:
        sqlite-vec binary representation of the embedding.

    Raises:
        ValueError: If the embedding does not contain exactly 256 values.
    """
    values = [float(value) for value in embedding]
    if len(values) != EMBEDDING_DIM:
        raise ValueError(f"Embedding must contain exactly {EMBEDDING_DIM} values")
    return _get_sqlite_vec().serialize_float32(values)


def deserialize_embedding(serialized_embedding: bytes) -> tuple[float, ...]:
    """
    Deserialize a sqlite-vec float32 embedding blob into Python float values.

    Args:
        serialized_embedding: Raw bytes read from the sqlite-vec embedding column.

    Returns:
        Tuple containing exactly 256 speaker embedding values.

    Raises:
        ValueError: If the serialized embedding does not contain exactly 256 float values.
    """
    values = np.frombuffer(serialized_embedding, dtype=np.float32).astype(float)
    if values.size != EMBEDDING_DIM:
        raise ValueError(f"Serialized embedding must contain exactly {EMBEDDING_DIM} values")
    return tuple(float(value) for value in values)
