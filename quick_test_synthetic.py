"""
Quick test of the synthetic experiment replication (smaller scale for demonstration)
"""

import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/claude')
from deephalo_choicelearn import DeepHaloFeatureless

print("="*70)
print("Quick Test: Synthetic Data Experiment")
print("="*70)

# Generate small test data
np.random.seed(20)
n_items = 20
choice_set_size = 15
n_choice_sets = 100  # Much smaller for testing

print(f"\nGenerating test data: {n_choice_sets} choice sets")

X_list = []
Y_list = []

for _ in range(n_choice_sets):
    # Random choice set
    items = np.random.choice(n_items, size=choice_set_size, replace=False)
    binary = np.zeros(n_items, dtype=np.float32)
    binary[items] = 1.0
    
    # Random probabilities
    probs = np.zeros(n_items, dtype=np.float32)
    probs[items] = np.random.dirichlet(np.ones(choice_set_size))
    
    # Sample 10 choices per set
    for _ in range(10):
        chosen = np.random.choice(n_items, p=probs)
        one_hot = np.zeros(n_items, dtype=np.float32)
        one_hot[chosen] = 1.0
        
        X_list.append(binary.copy())
        Y_list.append(one_hot)

X_data = np.array(X_list)
Y_data = np.array(Y_list)

print(f"X shape: {X_data.shape}")
print(f"Y shape: {Y_data.shape}")

# Split train/test
n_train = int(0.8 * len(X_data))
X_train, X_test = X_data[:n_train], X_data[n_train:]
Y_train, Y_test = Y_data[:n_train], Y_data[n_train:]

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Test different depths
print("\n" + "="*70)
print("Testing Different Model Depths")
print("="*70)

depths_to_test = [3, 4, 5]
width = 50  # Small width for quick testing

results = []

for depth in depths_to_test:
    print(f"\nDepth {depth}, Width {width}")
    
    # Create model
    model = DeepHaloFeatureless(
        n_items=n_items,
        depth=depth,
        width=width,
        block_types=['qua'] * (depth - 1)
    )
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices({
        'items': X_train,
        'choices': Y_train
    }).batch(32)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    # Train for 20 epochs
    for epoch in range(20):
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                inputs = {'items': batch['items']}
                loss = model.compute_negative_log_likelihood(inputs, batch['choices'])
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Evaluate
    train_inputs = {'items': tf.constant(X_train, dtype=tf.float32)}
    test_inputs = {'items': tf.constant(X_test, dtype=tf.float32)}
    
    train_outputs = model(train_inputs, training=False)
    test_outputs = model(test_inputs, training=False)
    
    train_mse = tf.reduce_mean(tf.keras.losses.MSE(Y_train, train_outputs['probabilities']))
    test_mse = tf.reduce_mean(tf.keras.losses.MSE(Y_test, test_outputs['probabilities']))
    
    train_rmse = tf.sqrt(train_mse).numpy()
    test_rmse = tf.sqrt(test_mse).numpy()
    
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    
    results.append({
        'depth': depth,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    })

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"{'Depth':<10} {'Train RMSE':<15} {'Test RMSE':<15}")
print("-"*40)
for r in results:
    print(f"{r['depth']:<10} {r['train_rmse']:<15.4f} {r['test_rmse']:<15.4f}")

print("\n" + "="*70)
print("Key Finding:")
print("As model depth increases, RMSE should decrease (up to the point")
print("where 2^(depth-1) exceeds the choice set size of 15)")
print("="*70)

print("\nQuick test completed! ✓")
print("\nFor full replication matching the paper:")
print("  - Use full dataset (1.24M training samples)")
print("  - Test parameter budgets 200k and 500k")
print("  - Train for 500 epochs with early stopping")
print("  - See: replicate_synthetic_experiment.py")
