# Biometric Speech Recognition Nerf Blaster Activation

## Authors

- Mason Barlow
- Alex Searle

## Description

This project is a biometric speaker recognition system that identifies people from speech input and determines whether the speaker is registered in the project database. The system is designed around a controlled Nerf blaster activation demo: if a speaker is not recognized as a registered user, the application will not start the nerf blaster motors.

The project treats speech as biometric data, so the system is designed to separate model inference, speaker database lookup, safety validation, and any hardware-control behavior. Motor activation should remain disabled by default during development and testing, and real hardware should only be used in a supervised demo with explicit consent.

## Models Used

### Fine-Tuned Transformer

The fine-tuned model uses WavLM finetuned on some of the Voxceleb1 speaker identification dataset. This approach benefits from learned acoustic representations from a larger model and then specializes those representations for speaker identification. It is intended to provide a strong baseline for identifying known speakers and rejecting unknown speakers based on confidence thresholds.

### Homegrown Transformer

The homegrown model is a custom PyTorch transformer implemented specifically for this project and trained on the Voxceleb1 dataset. It provides a transparent architecture for experimenting with audio features, transformer layers, classification heads, and training behavior. This model is useful for understanding the speaker recognition pipeline end to end and comparing a project-built architecture against a fine-tuned pretrained model.

## Database Schema

The project uses a local SQLite database and `sqlite-vec` to store the features extracted from audio as 256-dimensional speaker embedding vectors.

The database stores registered users separately from model embeddings so that both the homegrown and fine-tuned models can store embeddings for the same person.

### `users`

| Field | Type | Description |
| --- | --- | --- |
| `user_id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique database ID for the enrolled user. |
| `name` | `TEXT NOT NULL` | Name associated with the enrolled speaker. |
| `target` | `BOOLEAN NOT NULL DEFAULT 0` | Whether the user is marked as a target in the demo logic. |

### `embeddings`

This is a `sqlite-vec` virtual table created with `vec0`.

| Field | Type | Description |
| --- | --- | --- |
| `rowid` | implicit `INTEGER` | SQLite row ID used as the embedding identifier. |
| `embedding` | `FLOAT[256]` | 256-dimensional speaker feature vector extracted from audio. |

### `user_embeddings`

| Field | Type | Description |
| --- | --- | --- |
| `user_embedding_id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique ID for the link between a user and a stored embedding. |
| `user_id` | `INTEGER NOT NULL` | Foreign key referencing `users.user_id`; deleted automatically when the user is deleted. |
| `embedding_id` | `INTEGER NOT NULL` | ID of the vector stored in the `embeddings` virtual table. |
| `model_type` | `TEXT NOT NULL` | Model family that produced the embedding; must be `homegrown` or `finetuned`. |
| `created_at` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | Timestamp for when the user embedding record was created. |

## Project Structure

```text
biometrics_final_project/
  README.md
  AGENTS.md
  pyproject.toml
  requirments.txt
  notebooks/
    Homegrown Audio Transformer.ipynb
    Voice Identificaton.ipynb
  scripts/
    create_graphs.py
    test_audio_similarity.py
  src/
    app/
      database.py
      enrollment.py
      identify.py
      inference.py
      load.py
      main.py
      utils.py
    my_engine/
      __init__.py
      audio.py
      config.py
      data.py
      model.py
      sweep.py
      text.py
      trainer.py
      utils.py
```

### Directory Overview

- `notebooks/`: Exploratory experiments, model development, and analysis.
- `scripts/`: Utility scripts for testing audio similarity and generating graphs.
- `src/app/`: Application-level code for enrollment, database access, inference, and speaker identification.
- `src/my_engine/`: Custom machine learning engine code for audio processing, model definitions, training, configuration, and utilities.
