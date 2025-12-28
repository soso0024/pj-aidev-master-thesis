#!/usr/bin/env python3
"""
Batch Test Case Generator for HumanEval Problems

This script automates running the run_test_case_generator.py for multiple HumanEval task IDs
in a specified range, making it easy to generate test cases for many problems at once.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import time

# Constants for script paths
SCRIPT_DIR = Path(__file__).parent
MAIN_GENERATOR_PATH = SCRIPT_DIR.parent / "run_test_case_generator.py"

# Add parent directory to path to import model_utils
sys.path.insert(0, str(SCRIPT_DIR.parent))
from model_utils import get_available_models, get_default_model


class BatchTestGenerator:
    def __init__(
        self,
        start_id: int = 0,
        end_id: int = 50,
        models: List[str] = None,
        include_docstring: bool = False,
        include_ast: bool = False,
        ast_fix: bool = False,
        disable_evaluation: bool = False,
        max_fix_attempts: int = 3,
        quiet_evaluation: bool = False,
        task_timeout: int = 300,
        output_dir: str = "generated_tests",
        dataset: str = "dataset/HumanEval.jsonl",
    ):
        """Initialize the batch generator with configuration."""
        self.start_id = start_id
        self.end_id = end_id
        self.models = models if models else [get_default_model()]
        self.include_docstring = include_docstring
        self.include_ast = include_ast
        self.ast_fix = ast_fix
        self.disable_evaluation = disable_evaluation
        self.max_fix_attempts = max_fix_attempts
        self.quiet_evaluation = quiet_evaluation
        self.task_timeout = task_timeout
        self.output_dir = (
            output_dir  # Just store as string, no need to create directory
        )
        self.dataset = dataset

        # Statistics tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.skipped_tasks = 0

    def build_command(self, task_id: str) -> List[str]:
        """Build the command to run for a specific task ID."""
        cmd = [
            sys.executable,
            str(MAIN_GENERATOR_PATH),
            "--task-id",
            task_id,
            "--dataset",
            self.dataset,
            "--output-dir",
            str(self.output_dir),
            "--max-fix-attempts",
            str(self.max_fix_attempts),
            "--models",
        ]

        # Add all models
        cmd.extend(self.models)

        if self.include_docstring:
            cmd.append("--include-docstring")
        if self.include_ast:
            cmd.append("--include-ast")
        if self.ast_fix:
            cmd.append("--ast-fix")
        if self.disable_evaluation:
            cmd.append("--disable-evaluation")
        if self.quiet_evaluation:
            cmd.append("--quiet-evaluation")
        # Always force non-interactive mode for batch runs
        cmd.append("--no-show-prompt")

        return cmd

    def run_single_task(self, task_id: str) -> bool:
        """Run test generation for a single task ID."""
        cmd = self.build_command(task_id)

        print(f"\n{'='*60}")
        print(f"🚀 Processing {task_id}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")

        try:
            # Run the command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.task_timeout
            )

            # Print output
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            # Check if successful
            if result.returncode == 0:
                print(f"✅ Successfully completed {task_id}")
                return True
            else:
                print(
                    f"❌ Failed to complete {task_id} (exit code: {result.returncode})"
                )
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ Task {task_id} timed out after {self.task_timeout} seconds")
            return False
        except Exception as e:
            print(f"💥 Error running task {task_id}: {e}")
            return False

    def run_batch(self, task_ids: Optional[List[str]] = None) -> None:
        """Run test generation for a batch of task IDs."""
        if task_ids is None:
            # Generate task IDs from range
            task_ids = [f"HumanEval/{i}" for i in range(self.start_id, self.end_id + 1)]

        self.total_tasks = len(task_ids)

        print(f"🎯 Starting batch generation for {self.total_tasks} tasks")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔧 Configuration:")
        print(f"  - Models: {', '.join(self.models)}")
        print(f"  - Include docstrings: {self.include_docstring}")
        print(f"  - Include AST: {self.include_ast}")
        print(f"  - AST-based fixing: {self.ast_fix}")
        print(f"  - Evaluation disabled: {self.disable_evaluation}")
        print(f"  - Max fix attempts: {self.max_fix_attempts}")

        start_time = time.time()

        for i, task_id in enumerate(task_ids, 1):
            print(
                f"\n📊 Progress: {i}/{self.total_tasks} ({i/self.total_tasks*100:.1f}%)"
            )

            success = self.run_single_task(task_id)

            if success:
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1

                # Handle failure based on quiet mode
                if not self.quiet_evaluation:
                    while True:
                        response = (
                            input(
                                f"\n❓ Task {task_id} failed. Continue with remaining tasks? (y/n/q): "
                            )
                            .lower()
                            .strip()
                        )
                        if response in ["y", "yes"]:
                            break
                        elif response in ["n", "no"]:
                            print("⏹️  Stopping batch processing...")
                            self.skipped_tasks = self.total_tasks - i
                            break
                        elif response in ["q", "quit"]:
                            print("🛑 Quitting batch processing...")
                            self.skipped_tasks = self.total_tasks - i
                            return
                        else:
                            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")

                    if response in ["n", "no"]:
                        self.skipped_tasks = self.total_tasks - i
                        break
                else:
                    # In quiet mode, automatically continue on failure
                    print(
                        f"⚠️  Task {task_id} failed. Continuing with next task (quiet mode)..."
                    )

        # Print final summary
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        print(f"🏁 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Summary:")
        print(f"  Total tasks: {self.total_tasks}")
        print(f"  ✅ Successful: {self.successful_tasks}")
        print(f"  ❌ Failed: {self.failed_tasks}")
        print(f"  ⏭️  Skipped: {self.skipped_tasks}")
        print(f"  ⏱️  Duration: {duration:.1f} seconds")
        print(f"  📁 Output directory: {self.output_dir}")

        if self.successful_tasks > 0:
            print(
                f"\n🎉 Successfully generated test cases for {self.successful_tasks} problems!"
            )
            print(f"💡 To run all tests: pytest {self.output_dir}/ -v")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate test cases for multiple HumanEval problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tests for HumanEval/0 through HumanEval/10
  python run_batch_test_case_generator.py --start 0 --end 10

  # Generate with docstrings and AST info
  python run_batch_test_case_generator.py --start 0 --end 5 --include-docstring --include-ast

  # Generate specific task IDs
  python run_batch_test_case_generator.py --task-ids "HumanEval/0,HumanEval/5,HumanEval/10"

  # Disable evaluation for faster generation
  python run_batch_test_case_generator.py --start 0 --end 20 --disable-evaluation
        """,
    )

    # Range-based generation
    parser.add_argument(
        "--start", type=int, default=0, help="Start task ID number (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=50, help="End task ID number (default: 50)"
    )

    # Specific task IDs
    parser.add_argument(
        "--task-ids",
        help="Comma-separated list of specific task IDs (e.g., 'HumanEval/0,HumanEval/5')",
    )

    # Generator options (passed through to run_test_case_generator.py)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[get_default_model()],
        choices=get_available_models(),
        help="Claude model(s) to use for test generation (can specify multiple)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/HumanEval.jsonl",
        help="Path to HumanEval dataset file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/generated_tests",
        help="Output directory for test files",
    )
    parser.add_argument(
        "--include-docstring",
        action="store_true",
        help="Include function docstring in prompt",
    )
    parser.add_argument(
        "--include-ast",
        action="store_true",
        help="Include AST of canonical solution in prompt",
    )
    parser.add_argument(
        "--ast-fix",
        action="store_true",
        help="Enable AST-focused error fixing in the retry loop",
    )
    parser.add_argument(
        "--disable-evaluation",
        action="store_true",
        help="Disable automatic evaluation and fixing of generated tests",
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=3,
        help="Maximum number of LLM fix attempts (default: 3)",
    )
    parser.add_argument(
        "--quiet-evaluation",
        action="store_true",
        help="Disable verbose output during error fixing process",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each individual task (default: 300)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not MAIN_GENERATOR_PATH.exists():
        print(
            f"❌ Error: run_test_case_generator.py not found at {MAIN_GENERATOR_PATH}"
        )
        return 1

    if not Path(args.dataset).exists():
        print(f"❌ Error: Dataset file {args.dataset} not found")
        return 1

    # Parse task IDs
    task_ids = None
    if args.task_ids:
        task_ids = [tid.strip() for tid in args.task_ids.split(",")]
        print(f"🎯 Using specific task IDs: {task_ids}")
    else:
        if args.start > args.end:
            print("❌ Error: Start ID must be less than or equal to end ID")
            return 1
        print(f"🎯 Using range: HumanEval/{args.start} to HumanEval/{args.end}")

    # Create and run batch generator
    try:
        batch_gen = BatchTestGenerator(
            start_id=args.start,
            end_id=args.end,
            models=args.models,
            include_docstring=args.include_docstring,
            include_ast=args.include_ast,
            ast_fix=args.ast_fix,
            disable_evaluation=args.disable_evaluation,
            max_fix_attempts=args.max_fix_attempts,
            quiet_evaluation=args.quiet_evaluation,
            task_timeout=args.task_timeout,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )

        batch_gen.run_batch(task_ids)

        return 0

    except KeyboardInterrupt:
        print("\n🛑 Batch processing interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
