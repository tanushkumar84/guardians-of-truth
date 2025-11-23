#!/usr/bin/env python3
"""
Check dataset to diagnose 100% accuracy issue
"""

from datasets import load_dataset
import random

print("Loading dataset...")
dataset = load_dataset("JamieWithofs/Deepfake-and-real-images", split="train")
print(f"Total samples: {len(dataset)}")

# Check label distribution
print("\nChecking first 100 samples:")
labels = []
for i in range(100):
    item = dataset[i]
    labels.append(item['label'])
    if i < 10:
        print(f"Sample {i}: label = {item['label']}, image size = {item['image'].size}")

# Count real vs fake
real_count = sum(1 for l in labels if l == 'real')
fake_count = sum(1 for l in labels if l == 'fake')
print(f"\nFirst 100 samples: Real={real_count}, Fake={fake_count}")

# Check random samples
print("\nChecking 100 random samples:")
indices = random.sample(range(len(dataset)), 100)
labels_random = []
for i in indices:
    item = dataset[i]
    labels_random.append(item['label'])

real_random = sum(1 for l in labels_random if l == 'real')
fake_random = sum(1 for l in labels_random if l == 'fake')
print(f"Random 100 samples: Real={real_random}, Fake={fake_random}")

# Check if all images are same size
print("\nChecking image diversity:")
sizes = set()
for i in random.sample(range(len(dataset)), 20):
    sizes.add(dataset[i]['image'].size)
print(f"Unique image sizes in 20 random samples: {sizes}")

print("\n✅ Dataset check complete!")
