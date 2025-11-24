"""
Test the frequency probability computation
"""

import numpy as np

def compute_frequency_probs(X, Y):
    """
    Compute empirical frequency probabilities for each unique choice set.
    """
    # Create dictionary mapping choice set to list of indices
    unique_dict = {}
    
    for i in range(len(X)):
        # Convert row to tuple (hashable) for dictionary key
        key = tuple(X[i].tolist())
        if key not in unique_dict:
            unique_dict[key] = []
        unique_dict[key].append(i)
    
    # Compute average Y for each unique choice set
    Y_freq = Y.copy()
    for indices in unique_dict.values():
        avg_y = Y[indices].mean(axis=0)
        Y_freq[indices] = avg_y
    
    return Y_freq


# Test it
print("Testing frequency probability computation...")

# Create test data: 3 unique choice sets, 2 samples each
X = np.array([
    [1, 1, 0, 0],  # Set A
    [1, 1, 0, 0],  # Set A (duplicate)
    [1, 0, 1, 0],  # Set B
    [1, 0, 1, 0],  # Set B (duplicate)
    [0, 1, 1, 0],  # Set C
    [0, 1, 1, 0],  # Set C (duplicate)
], dtype=np.float32)

# Different choices for each sample
Y = np.array([
    [1, 0, 0, 0],  # Set A, chose item 0
    [0, 1, 0, 0],  # Set A, chose item 1
    [1, 0, 0, 0],  # Set B, chose item 0
    [0, 0, 1, 0],  # Set B, chose item 2
    [0, 1, 0, 0],  # Set C, chose item 1
    [0, 0, 1, 0],  # Set C, chose item 2
], dtype=np.float32)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

Y_freq = compute_frequency_probs(X, Y)

print(f"Y_freq shape: {Y_freq.shape}")
print("\nExpected behavior:")
print("- Rows 0,1 (same choice set) should have same Y_freq")
print("- Rows 2,3 (same choice set) should have same Y_freq")
print("- Rows 4,5 (same choice set) should have same Y_freq")

print("\nY_freq:")
print(Y_freq)

# Verify correctness
assert np.allclose(Y_freq[0], Y_freq[1]), "Rows 0,1 should match"
assert np.allclose(Y_freq[2], Y_freq[3]), "Rows 2,3 should match"
assert np.allclose(Y_freq[4], Y_freq[5]), "Rows 4,5 should match"

# Check values
expected_0 = np.array([0.5, 0.5, 0, 0])  # Average of [1,0,0,0] and [0,1,0,0]
expected_2 = np.array([0.5, 0, 0.5, 0])  # Average of [1,0,0,0] and [0,0,1,0]
expected_4 = np.array([0, 0.5, 0.5, 0])  # Average of [0,1,0,0] and [0,0,1,0]

assert np.allclose(Y_freq[0], expected_0), "Set A average incorrect"
assert np.allclose(Y_freq[2], expected_2), "Set B average incorrect"
assert np.allclose(Y_freq[4], expected_4), "Set C average incorrect"

print("\n✓ All tests passed!")
print("Frequency computation works correctly.")
