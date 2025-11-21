# Testing Guide

This project uses `pytest` for unit testing. The test suite covers configuration management, training logic, and inference processing.

## Running Tests

To run the full test suite, execute the following command from the project root:

```bash
poetry run pytest tests/ -v
```

## Test Structure

- **`tests/conftest.py`**: Shared fixtures for tests (mock configs, sample images, etc.).
- **`tests/test_config_manager.py`**: Tests for loading, saving, and validating configurations.
- **`tests/test_trainer.py`**: Tests for the `YOLOTrainer` class, including initialization and parameter preparation.
- **`tests/test_predictor.py`**: Tests for the `YOLOPredictor` class, including image and batch prediction.

## Writing New Tests

When adding new features, please ensure to add corresponding tests.
- Use `pytest` fixtures from `conftest.py` where possible.
- Mock external dependencies like `ultralytics.YOLO` to avoid running actual heavy computations during tests.
- Ensure all tests pass before submitting a pull request.
