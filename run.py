# -*- coding: utf-8 -*-
"""
Dataset Pipeline V2 CLI - Fully Automatic

Usage:
    python run.py

Pipeline will automatically:
1. Check each step's completion status
2. Skip completed steps
3. Run remaining steps
4. Handle errors and auto-retry

Additional options:
    python run.py --force              # Force restart from beginning
    python run.py --step extract       # Run single step (debug/test)
    python run.py -c custom.yaml       # Use different config
"""

import argparse
import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DatasetPipeline
from core.logger import get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Pipeline V2 - Auto Q&A Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  python run.py                    # Run full pipeline (auto)
  python run.py --force            # Force restart
  python run.py --step evaluate    # Run single step (debug/test)
  python run.py -c custom.yaml     # Use different config

Pipeline Steps (auto-skip if completed):
  1. extract    - Extract text from PDF/documents
  2. generate   - Generate Q&A pairs using LLM
  3. evaluate   - Evaluate + Rescue + Regenerate (all-in-one)
  4. split      - Split by DOCUMENT (prevent data leakage)
  5. export     - Export final dataset
        """
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    parser.add_argument(
        "--step",
        type=str,
        choices=["extract", "generate", "evaluate", "split", "export"],
        help="Run single step only (for debug/test)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force restart, ignore checkpoints"
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed chunks in generate step"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed logs"
    )

    args = parser.parse_args()

    # Check config exists
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print("   Create config.yaml or specify path with -c")
        sys.exit(1)

    try:
        # Initialize pipeline
        print(f"Loading config: {args.config}")
        pipeline = DatasetPipeline(args.config)

        if args.retry_failed:
            # Retry failed chunks
            print("Retry failed chunks...")
            pipeline.generator.retry_failed()

        elif args.step:
            # Run single step (debug/test mode)
            print(f"Running single step: {args.step}")
            result = pipeline.run(steps=[args.step])

        elif args.force:
            # Force run from scratch
            print("Force running pipeline from scratch...")
            result = pipeline.run()

        else:
            # Auto mode - pipeline checks and skips completed steps
            print("Starting Pipeline V2 (auto mode)...")
            result = pipeline.run_auto()

        # Print summary
        print("\n" + "="*50)
        print("PIPELINE V2 SUMMARY")
        print("="*50)

        if isinstance(result, dict):
            print(f"  Version: {result.get('version', '2.0')}")
            print(f"  Project: {result.get('project', 'N/A')}")
            print(f"  Documents: {result.get('documents', 0)}")
            print(f"  Q&A Generated: {result.get('qa_pairs_generated', 0)}")
            print(f"  Good Q&A: {result.get('good_qa', 0)}")
            rescued = result.get('rescued_qa', 0)
            if rescued:
                print(f"  Rescued Q&A: {rescued}")
            print(f"  Bad Q&A: {result.get('bad_qa', 0)}")
            print(f"  Success Rate: {result.get('success_rate', 0):.1f}%")

            # V2: Show splits info
            splits = result.get('splits', {})
            if splits:
                print(f"\n  Document-based Splits:")
                print(f"    Train: {splits.get('train', 0)} samples")
                print(f"    Validation: {splits.get('validation', 0)} samples")
                print(f"    Test: {splits.get('test', 0)} samples")

        print("\nPipeline V2 completed!")

    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nPipeline error: {e}")
        logger.exception("Pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
