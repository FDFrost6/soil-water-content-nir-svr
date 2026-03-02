# Quick Start Guide

## Setup

1. **Activate virtual environment:**
```bash
source venv/bin/activate
```

2. **Verify installation:**
```bash
python main.py --help
```

## Common Commands

### Run Complete Pipeline
```bash
# Full pipeline: data ingestion → training → evaluation
python main.py full
```

### Individual Steps
```bash
# 1. Parse NIRQuest text files into CSV
python main.py ingest

# 2. Train water SVR + nitrogen RF models
python main.py train

# 3. Generate evaluation plots (6 plots)
python main.py evaluate
```

### Feature Engineering Analysis
```bash
# Compare 4 feature engineering approaches
python main.py compare

# Test PCA component counts (10, 25, 50, 100, 200)
python main.py pca-sweep

# Validate model stability across 5 random splits
python main.py test-seeds
```

### Make Predictions
```bash
# Predict on new spectrum with safety layer
python main.py predict --spectrum path/to/spectrum.csv

# Custom confidence threshold
python main.py predict --spectrum data.csv --confidence 0.70
```

### Run Tests
```bash
# Run all unit tests
python main.py test

# With coverage report
python main.py test --coverage
```

## Quick Examples

**Skip steps you've already completed:**
```bash
# Skip ingestion (use existing CSV)
python main.py full --skip-ingest

# Skip training (use existing models)
python main.py full --skip-train
```

**Quick training (skip re-training):**
```bash
python main.py train --quick
```

**Full pipeline with feature comparison:**
```bash
python main.py full --with-comparison
```

## Command Reference

| Command | Description |
|---------|-------------|
| `ingest` | Parse NIRQuest .txt files → CSV |
| `train` | Train water SVR + nitrogen RF |
| `evaluate` | Generate 6 evaluation plots |
| `predict` | Make predictions with safety checks |
| `compare` | Compare feature engineering (4 variants) |
| `pca-sweep` | Analyze PCA components (plot9-11) |
| `test-seeds` | Cross-split validation (5 seeds) |
| `test` | Run pytest unit tests |
| `full` | Complete pipeline (ingest→train→eval) |

## Help System

Get detailed help for any command:
```bash
python main.py <command> --help
```

Examples:
```bash
python main.py predict --help
python main.py full --help
python main.py test --help
```

## Troubleshooting

**ModuleNotFoundError:**
- Make sure virtual environment is activated: `source venv/bin/activate`

**File not found:**
- Check you're in project root: `cd /path/to/spectrometer\ trial`

**Models missing:**
- Run training first: `python main.py train`
- Or run full pipeline: `python main.py full`
