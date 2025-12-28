#!/usr/bin/env python3
"""
Generate sample researcher data for testing.

Usage:
    python generate_samples.py --output samples.jsonl --count 1000
"""

import argparse
import json
import random
from pathlib import Path

# Sample data
FIRST_NAMES = [
    "Geoffrey", "Yann", "Yoshua", "Andrew", "Fei-Fei", "Demis", "Ilya",
    "Michael", "Christopher", "Stuart", "Peter", "Judea", "Leslie",
    "Barbara", "Daphne", "Pieter", "Eric", "Ian", "Jürgen", "Alex",
]

LAST_NAMES = [
    "Hinton", "LeCun", "Bengio", "Ng", "Li", "Hassabis", "Sutskever",
    "Jordan", "Manning", "Russell", "Norvig", "Pearl", "Kaelbling",
    "Liskov", "Koller", "Abbeel", "Horvitz", "Goodfellow", "Schmidhuber", "Graves",
]

FIELDS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Reinforcement Learning",
    "Deep Learning",
    "Computer Science",
    "Computational Biology",
    "Data Science",
]

INSTITUTIONS = [
    "MIT", "Stanford University", "UC Berkeley", "CMU", "University of Toronto",
    "Oxford University", "Cambridge University", "ETH Zurich", "NYU", "Princeton",
    "Harvard University", "Google DeepMind", "OpenAI", "Meta AI", "Microsoft Research",
]


def generate_researcher(index: int) -> dict:
    """Generate a single researcher sample."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"

    return {
        "researcher_id": f"researcher_{index:06d}",
        "name": name,
        "field": random.choice(FIELDS),
        "known_info": {
            "affiliation": random.choice(INSTITUTIONS),
        },
        "priority": random.randint(0, 10),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sample researcher data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print(f"Generating {args.count} researcher samples...")

    # Generate samples
    with open(args.output, "w") as f:
        for i in range(args.count):
            researcher = generate_researcher(i + 1)
            f.write(json.dumps(researcher) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} samples...")

    print(f"✓ Generated {args.count} samples to {args.output}")

    # Print sample
    with open(args.output) as f:
        first_line = f.readline()
        print("\nSample record:")
        print(json.dumps(json.loads(first_line), indent=2))


if __name__ == "__main__":
    main()
