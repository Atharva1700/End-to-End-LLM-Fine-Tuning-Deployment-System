#!/usr/bin/env python3
"""
Sample Data Generator
Creates sample training data for testing the pipeline.
"""

import json
import os
import argparse
from pathlib import Path


# Sample QA pairs for different domains
SAMPLE_DATA = {
    "machine_learning": [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make decisions with minimal human intervention."
        },
        {
            "instruction": "Explain supervised learning",
            "response": "Supervised learning is a machine learning approach where the model is trained on labeled data. The algorithm learns to map inputs to outputs by learning from example input-output pairs. Common applications include classification and regression tasks."
        },
        {
            "instruction": "What is a neural network?",
            "response": "A neural network is a computing system inspired by biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that process and transmit information. Neural networks can learn complex patterns from data through training."
        },
        {
            "instruction": "Define deep learning",
            "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). It can automatically learn hierarchical representations from data, making it particularly effective for tasks like image recognition, natural language processing, and speech recognition."
        },
        {
            "instruction": "What is overfitting?",
            "response": "Overfitting occurs when a machine learning model learns the training data too well, including noise and random fluctuations. This results in poor performance on new, unseen data. Techniques like regularization, cross-validation, and dropout help prevent overfitting."
        },
    ],
    "programming": [
        {
            "instruction": "What is Python?",
            "response": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, AI, and automation."
        },
        {
            "instruction": "Explain object-oriented programming",
            "response": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and code (methods). Key principles include encapsulation, inheritance, and polymorphism. OOP helps organize code and makes it more modular and reusable."
        },
        {
            "instruction": "What is an API?",
            "response": "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. It defines the methods and data formats that applications can use to request and exchange information."
        },
    ],
    "science": [
        {
            "instruction": "What is quantum computing?",
            "response": "Quantum computing is a type of computing that uses quantum mechanical phenomena like superposition and entanglement to perform calculations. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously, potentially solving certain problems much faster."
        },
        {
            "instruction": "Explain photosynthesis",
            "response": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose. This process uses carbon dioxide and water, producing oxygen as a byproduct. It's fundamental to life on Earth."
        },
    ],
    "general": [
        {
            "instruction": "How does a blockchain work?",
            "response": "A blockchain is a distributed ledger technology that records transactions across multiple computers. Each block contains a set of transactions and is linked to the previous block through cryptographic hashes, forming a chain. This structure makes the data tamper-resistant and transparent."
        },
        {
            "instruction": "What is cloud computing?",
            "response": "Cloud computing is the delivery of computing services (servers, storage, databases, networking, software) over the internet. Instead of owning physical infrastructure, users can access these resources on-demand, paying only for what they use. Major providers include AWS, Google Cloud, and Azure."
        },
    ]
}


def create_dataset(output_dir, num_samples_per_category=5):
    """
    Create sample dataset files.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples_per_category: Number of samples to use from each category
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all samples
    all_samples = []
    for category, samples in SAMPLE_DATA.items():
        all_samples.extend(samples[:num_samples_per_category])
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    # Split into train/val/test (80/10/10)
    n = len(all_samples)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]
    
    # Save files
    def save_jsonl(samples, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved {len(samples)} samples to {filepath}")
    
    save_jsonl(train_samples, 'train.jsonl')
    save_jsonl(val_samples, 'val.jsonl')
    save_jsonl(test_samples, 'test.jsonl')
    
    # Save statistics
    stats = {
        'total_samples': n,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'categories': list(SAMPLE_DATA.keys()),
    }
    
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Total: {stats['total_samples']}")
    print(f"  Train: {stats['train_samples']}")
    print(f"  Val: {stats['val_samples']}")
    print(f"  Test: {stats['test_samples']}")
    print(f"\nDataset created successfully in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for dataset")
    parser.add_argument("--samples_per_category", type=int, default=5,
                       help="Number of samples per category")
    
    args = parser.parse_args()
    
    create_dataset(args.output_dir, args.samples_per_category)


if __name__ == "__main__":
    main()
