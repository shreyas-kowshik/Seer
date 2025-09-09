import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, help="Task name to filter sequences by")
    parser.add_argument("--input", default="input.json", help="Input JSON file (default: input.json)")
    parser.add_argument("--output", default=None, help="Output JSON file (default: evaluate_sequences_<task_name>.json)")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r") as f:
        data = json.load(f)

    # Filter sequences
    filtered = [entry for entry in data if args.task_name in entry[1]]

    # Set output filename in same directory as input
    input_dir = os.path.dirname(args.input)
    output_file = args.output or os.path.join(input_dir, f"evaluate_sequences_{args.task_name}.json")

    # Write filtered sequences
    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=4)

    print(f"Saved {len(filtered)} sequences to {output_file}")

if __name__ == "__main__":
    main()
