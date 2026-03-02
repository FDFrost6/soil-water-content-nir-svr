#!/usr/bin/env python3
"""
Soil Water & Nitrogen Prediction Pipeline
==========================================
Main command-line interface for the entire ML pipeline.

Usage:
    python main.py <command> [options]

Commands:
    ingest          Parse NIRQuest text files into CSV
    train           Train water SVR + nitrogen RF models
    evaluate        Generate evaluation plots
    predict         Make predictions on new spectra (with safety checks)
    compare         Compare feature engineering approaches
    pca-sweep       Analyze optimal PCA component count
    test-seeds      Validate model stability across random splits
    test            Run all unit tests
    full            Run complete pipeline (ingest → train → evaluate)

Examples:
    python main.py ingest
    python main.py train --quick
    python main.py evaluate
    python main.py predict --spectrum data.csv
    python main.py compare
    python main.py full

For detailed help on each command:
    python main.py <command> --help
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline modules
import ingest_records
import train_water_nitrogen
import evaluate_water_nitrogen
import compare_feature_engineering
import pca_sweep_analysis
import test_random_seeds
from protection_layer import load_models, predict_safe

import numpy as np
import pandas as pd
import pickle
import subprocess


def cmd_ingest(args):
    """Parse NIRQuest text files into CSV."""
    print("=" * 70)
    print("STEP 1: DATA INGESTION")
    print("=" * 70)
    ingest_records.main()
    print("\n✓ Ingestion complete!")


def cmd_train(args):
    """Train water SVR + nitrogen RF models."""
    print("=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)
    
    if args.quick:
        print("(Quick mode: skipped - using existing models)")
        return
    
    train_water_nitrogen.main()
    print("\n✓ Training complete!")


def cmd_evaluate(args):
    """Generate all evaluation plots."""
    print("=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)
    evaluate_water_nitrogen.main()
    print("\n✓ Evaluation complete!")


def cmd_predict(args):
    """Make predictions on new spectra with safety layer."""
    print("=" * 70)
    print("PREDICTION WITH SAFETY LAYER")
    print("=" * 70)
    
    # Load models
    print("Loading models...")
    water_pkg, nitrogen_pkg, ood_model = load_models()
    print("✓ Models loaded")
    
    # Load input spectrum
    if args.spectrum.endswith('.csv'):
        df = pd.read_csv(args.spectrum)
        # Assume CSV has 512 columns or rows with reflectance values
        if df.shape[1] == 512:
            X_new = df.values
        elif df.shape[0] == 512:
            X_new = df.values.T
        else:
            print(f"Error: CSV must have 512 wavelengths (got {df.shape})")
            return
    elif args.spectrum.endswith('.npy'):
        X_new = np.load(args.spectrum)
    else:
        print("Error: Input must be .csv or .npy file")
        return
    
    print(f"Input spectrum shape: {X_new.shape}")
    
    # Make prediction
    print("\nMaking prediction...")
    thresholds = {
        "nitrogen_confidence": args.confidence,
        "water_min": args.water_min,
        "water_max": args.water_max,
    }
    
    result = predict_safe(X_new, water_pkg, nitrogen_pkg, ood_model, thresholds)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Status:     {result['status'].upper()}")
    print(f"Water:      {result['water_pred']:.2f}%")
    
    if result['nitrogen_pred'] is not None:
        print(f"Nitrogen:   {result['nitrogen_pred']} mg/kg")
        print(f"Confidence: {result['max_prob']:.2f}")
    else:
        print(f"Nitrogen:   REJECTED ({result['reason']})")
    
    print(f"\nReason:     {result['reason']}")
    print("=" * 70)


def cmd_compare(args):
    """Compare feature engineering approaches."""
    print("=" * 70)
    print("FEATURE ENGINEERING COMPARISON")
    print("=" * 70)
    compare_feature_engineering.main()
    print("\n✓ Comparison complete!")


def cmd_pca_sweep(args):
    """Analyze optimal PCA component count."""
    print("=" * 70)
    print("PCA COMPONENT SWEEP ANALYSIS")
    print("=" * 70)
    pca_sweep_analysis.main()
    print("\n✓ PCA sweep complete!")


def cmd_test_seeds(args):
    """Validate model stability across random splits."""
    print("=" * 70)
    print("RANDOM SEED STABILITY VALIDATION")
    print("=" * 70)
    test_random_seeds.main()
    print("\n✓ Stability validation complete!")


def cmd_test(args):
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    test_args = ["pytest", "tests/", "-v"]
    if args.verbose:
        test_args.append("-vv")
    if args.coverage:
        test_args.extend(["--cov=src", "--cov-report=term-missing"])
    
    result = subprocess.run(test_args)
    sys.exit(result.returncode)


def cmd_full(args):
    """Run complete pipeline: ingest → train → evaluate."""
    print("=" * 80)
    print(" " * 20 + "FULL PIPELINE EXECUTION")
    print("=" * 80)
    print()
    
    # Step 1: Ingest
    if not args.skip_ingest:
        cmd_ingest(args)
        print()
    else:
        print("Skipping ingestion (using existing data)")
        print()
    
    # Step 2: Train
    if not args.skip_train:
        cmd_train(args)
        print()
    else:
        print("Skipping training (using existing models)")
        print()
    
    # Step 3: Evaluate
    cmd_evaluate(args)
    print()
    
    # Step 4: Feature comparison (optional)
    if args.with_comparison:
        cmd_compare(args)
        print()
    
    print("=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  Models:  models/svr_water_records.pkl")
    print("           models/rf_nitrogen_records.pkl")
    print("           models/ood_detector.pkl")
    print("  Plots:   results/plot1_water_actual_vs_predicted.png")
    print("           results/plot2_nitrogen_confusion_matrix.png")
    print("           results/plot3_spectral_signatures.png")
    print("           results/plot4_water_residuals.png")
    print("           results/plot5_guardrail_coverage.png")
    print("           results/plot6_nitrogen_by_water_level.png")
    if args.with_comparison:
        print("           results/plot7_feature_comparison_confusion.png")
        print("           results/plot8_feature_comparison_barplot.png")
    print("  Reports: results/training_summary.txt")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Soil Water & Nitrogen Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py full

  # Just train and evaluate
  python main.py train && python main.py evaluate

  # Compare feature engineering approaches
  python main.py compare

  # Make prediction on new spectrum
  python main.py predict --spectrum my_spectrum.csv

  # Run tests with coverage
  python main.py test --coverage
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    parser_ingest = subparsers.add_parser(
        'ingest',
        help='Parse NIRQuest text files into CSV'
    )
    parser_ingest.set_defaults(func=cmd_ingest)
    
    # Train command
    parser_train = subparsers.add_parser(
        'train',
        help='Train water SVR + nitrogen RF models'
    )
    parser_train.add_argument(
        '--quick',
        action='store_true',
        help='Skip training (use existing models)'
    )
    parser_train.set_defaults(func=cmd_train)
    
    # Evaluate command
    parser_eval = subparsers.add_parser(
        'evaluate',
        help='Generate all evaluation plots (6 plots)'
    )
    parser_eval.set_defaults(func=cmd_evaluate)
    
    # Predict command
    parser_pred = subparsers.add_parser(
        'predict',
        help='Make predictions on new spectra with safety layer'
    )
    parser_pred.add_argument(
        '--spectrum', '-s',
        required=True,
        help='Path to spectrum file (.csv or .npy with 512 wavelengths)'
    )
    parser_pred.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.60,
        help='Nitrogen confidence threshold (default: 0.60)'
    )
    parser_pred.add_argument(
        '--water-min',
        type=float,
        default=-2.0,
        help='Minimum valid water content %% (default: -2.0)'
    )
    parser_pred.add_argument(
        '--water-max',
        type=float,
        default=40.0,
        help='Maximum valid water content %% (default: 40.0)'
    )
    parser_pred.set_defaults(func=cmd_predict)
    
    # Compare command
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare 4 feature engineering approaches'
    )
    parser_compare.set_defaults(func=cmd_compare)
    
    # PCA sweep command
    parser_pca = subparsers.add_parser(
        'pca-sweep',
        help='Analyze optimal PCA component count (10, 25, 50, 100, 200)'
    )
    parser_pca.set_defaults(func=cmd_pca_sweep)
    
    # Test seeds command
    parser_seeds = subparsers.add_parser(
        'test-seeds',
        help='Validate model stability across 5 random splits'
    )
    parser_seeds.set_defaults(func=cmd_test_seeds)
    
    # Test command
    parser_test = subparsers.add_parser(
        'test',
        help='Run all unit tests with pytest'
    )
    parser_test.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser_test.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser_test.set_defaults(func=cmd_test)
    
    # Full pipeline command
    parser_full = subparsers.add_parser(
        'full',
        help='Run complete pipeline: ingest → train → evaluate'
    )
    parser_full.add_argument(
        '--skip-ingest',
        action='store_true',
        help='Skip data ingestion (use existing CSV)'
    )
    parser_full.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training (use existing models)'
    )
    parser_full.add_argument(
        '--with-comparison',
        action='store_true',
        help='Also run feature engineering comparison'
    )
    parser_full.set_defaults(func=cmd_full)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
