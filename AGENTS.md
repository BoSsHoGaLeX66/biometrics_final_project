# AGENTS.md

## Project Overview

This project is a biometrics final project focused on speaker recognition. The system classifies a person from speech input and determines whether the speaker exists in the registered user database.

The project will compare two modeling approaches:

1. A homegrown transformer model implemented in PyTorch.
2. A fine-tuned transformer model from Hugging Face.

If the speaker is not recognized as a registered user, the system may trigger a physical Nerf blaster turret prototype. Any physical actuation code must prioritize safety, testing controls, and consent.

## Safety Requirements

- Do not fire the Nerf turret at anyone without explicit consent.
- Turret actuation must be disabled by default in development and testing.
- Include a manual safety override for any code that controls hardware.
- Hardware-control code must be isolated from model inference code.
- Use mock hardware interfaces in tests.
- Never optimize the system for causing harm, pain, or injury.
- Treat the Nerf turret as a controlled demo device, not as a security weapon.

## Project Structure

```text
data/
models/
notebooks/
src/
  my_engine/
  app/
test/
pyproject.toml
README.md
````

### `data/`

Stores datasets, processed audio features, metadata, and train/test splits.

Guidelines:

* Do not commit large raw datasets unless explicitly allowed.
* Keep personally identifiable biometric data private.
* Use clear filenames for processed datasets.
* Document preprocessing assumptions.

### `models/`

Stores trained model weights, checkpoints, and model configuration files.

Guidelines:

* Use descriptive checkpoint names.
* Save metadata with each trained model when possible.
* Do not overwrite strong model checkpoints without backing them up.

### `notebooks/`

Stores exploratory work, experiments, and visualization notebooks.

Guidelines:

* Keep notebooks readable and organized.
* Move reusable code into `src/`.
* Avoid relying on notebook-only state for important results.

### `src/engine/`

Contains the machine learning engine.

Guidelines:

* Do not modify this directory unless explicitly instructed.
* Reuse existing engine abstractions where possible.
* Keep training loops, evaluation logic, and model utilities compatible with the existing engine.

### `src/app/`

Contains application-level code for inference, speaker registration, database lookup, and hardware-control coordination.

Suggested modules:

```text
src/app/
  audio_capture.py
  preprocessing.py
  inference.py
  speaker_database.py
  turret_controller.py
  safety.py
  config.py
```

### `test/`

Contains unit tests and integration tests.

Guidelines:

* Use `pytest`.
* Test preprocessing, model input shapes, inference outputs, and database lookup logic.
* Mock hardware-control behavior.
* Do not run real turret actuation during automated tests.

## Core Libraries

Primary libraries:

* `numpy`
* `pandas`
* `torch`
* `transformers`
* `datasets`

Additional libraries may be introduced when necessary, especially for:

* Audio loading and feature extraction
* Model evaluation
* Visualization
* Hardware communication
* Testing

Avoid adding new dependencies unless they clearly improve the project.

## Code Style

* Use clear, descriptive variable names.
* Prefer small, focused functions.
* Keep functions responsible for one main task.
* Avoid deeply nested logic when helper functions would improve readability.
* Use type hints for all public functions and methods.
* All classes and methods must have docstrings.
* Comments should be clear and concise.
* Do not comment code that already explains itself.

Good comment:

```python
# Convert raw logits to class probabilities for threshold-based rejection.
probs = torch.softmax(logits, dim=-1)
```

Avoid:

```python
# Add one to i.
i += 1
```

## Docstring Guidelines

Use concise docstrings that explain purpose, inputs, and outputs.

Example:

```python
def predict_speaker(audio_path: str, model: torch.nn.Module, threshold: float) -> dict:
    """
    Predict the speaker identity from an audio file.

    Args:
        audio_path: Path to the input audio file.
        model: Trained speaker classification model.
        threshold: Minimum confidence required to accept a known speaker.

    Returns:
        Dictionary containing the predicted speaker, confidence, and database status.
    """
```

## Modeling Guidelines

* Keep homegrown transformer code separate from Hugging Face model code.
* Use consistent train/validation/test splits.
* Track accuracy, precision, recall, F1-score, and confusion matrices.
* Include a rejection threshold for unknown speakers.
* Do not assume every input speaker belongs to a known class.
* Evaluate both closed-set classification and open-set rejection behavior.

## Speaker Recognition Logic

The system should distinguish between:

1. Known speaker correctly identified.
2. Known speaker incorrectly identified.
3. Unknown speaker correctly rejected.
4. Unknown speaker incorrectly accepted.

The unknown-speaker path should be handled carefully and conservatively.

## Hardware-Control Guidelines

* Hardware code should live behind a safe interface.
* Use a mock turret controller by default.
* Require an explicit config flag before enabling real hardware.
* Separate these steps:

    1. Speaker inference
    2. Database lookup
    3. Safety validation
    4. Turret actuation

Example design:

```python
if result["is_registered"]:
    grant_access(result)
else:
    safety_controller.validate_demo_mode()
    turret_controller.activate_mock()
```

## Testing Guidelines

Tests should cover:

* Audio preprocessing output shape
* Dataset loading
* Transformer forward passes
* Hugging Face model inference
* Speaker database lookup
* Unknown-speaker thresholding
* Mock turret activation
* Safety override behavior

Do not write tests that require live hardware.

## Data Privacy

Speech data is biometric data.

Guidelines:

* Do not expose raw voice recordings unnecessarily.
* Avoid committing private recordings.
* Use anonymized speaker IDs.
* Keep metadata minimal.
* Document how users are registered and removed from the database.

## Development Priorities

1. Build a reliable audio preprocessing pipeline.
2. Train and evaluate the homegrown transformer.
3. Fine-tune and evaluate a Hugging Face transformer.
4. Implement unknown-speaker rejection.
5. Integrate database lookup.
6. Add safe mock turret behavior.
7. Add real hardware integration only after the software pipeline is tested.

## Final Reminder

Prioritize correctness, safety, readability, and reproducibility. This is a biometrics research project, so the code should be easy to understand, easy to test, and careful with biometric data.

