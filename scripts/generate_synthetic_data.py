"""CLI entry point: generate the synthetic placeholder dataset into data/synthetic/."""

from canids.data.synthetic import generate_default_dataset

if __name__ == "__main__":
    generate_default_dataset()
    print("Synthetic dataset written to data/synthetic/")
