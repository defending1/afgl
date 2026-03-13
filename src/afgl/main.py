# src/afgl/main.py
import sys


def run():
    """Main execution function."""
    print("Starting the Accelerated Filtering Graphs Lanczos (AFGL) process...")


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
