"""
Replicate the Synthetic High-Order Data Experiments from Zhang et al. (2025)

This script replicates the exact experimental setup from the paper:
- 20 items universe
- Choice sets of size 15 (fixed)
- Random probability distributions per choice set
- Train: 80 samples per choice set (1,240,320 total)
- Test: 20 samples per choice set (310,080 total)
- Comparison across different model depths (3-7) with fixed parameter budgets
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/home/claude')
from deephalo_choicelearn import DeepHaloFeatureless


def generate_synthetic_data(n_items=20, choice_set_size=15, 
                           train_samples_per_set=80, test_samples_per_set=20,
                           seed=20):
    """
    Generate synthetic data following the exact procedure from the paper.
    
    This creates all possible choice sets of size 15 from 20 items,
    assigns random probabilities to each set using Dirichlet distribution,
    and samples choices accordingly.
    
    Args:
        n_items: Number of items in universe (20)
        choice_set_size: Size of each choice set (15)
        train_samples_per_set: Samples per choice set for training (80)
        test_samples_per_set: Samples per choice set for testing (20)
        seed: Random seed
        
    Returns:
        X_train, Y_train, X_test, Y_test as numpy arrays
    """
    np.random.seed(seed)
    
    print(f"Generating synthetic data with {n_items} items, choice sets of size {choice_set_size}")
    
    # Generate all choice sets of the specified size
    offer_set = list(range(n_items))
    all_choice_sets = list(itertools.combinations(offer_set, choice_set_size))
    
    print(f"Total number of choice sets: {len(all_choice_sets)}")
    
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []
    
    for choice_set in tqdm(all_choice_sets, desc="Generating data"):
        # Create binary indicator for this choice set
        binary_indicator = np.zeros(n_items, dtype=np.float32)
        binary_indicator[list(choice_set)] = 1.0
        
        # Generate random probabilities using Dirichlet distribution
        # This assigns random probabilities to available items that sum to 1
        probabilities = np.zeros(n_items, dtype=np.float32)
        dirichlet_probs = np.random.dirichlet(np.ones(choice_set_size))
        probabilities[list(choice_set)] = dirichlet_probs
        
        # Sample training data
        for _ in range(train_samples_per_set):
            # Sample one choice according to probabilities
            chosen_idx = np.random.choice(n_items, p=probabilities)
            one_hot = np.zeros(n_items, dtype=np.float32)
            one_hot[chosen_idx] = 1.0
            
            X_train_list.append(binary_indicator.copy())
            Y_train_list.append(one_hot)
        
        # Sample test data
        for _ in range(test_samples_per_set):
            chosen_idx = np.random.choice(n_items, p=probabilities)
            one_hot = np.zeros(n_items, dtype=np.float32)
            one_hot[chosen_idx] = 1.0
            
            X_test_list.append(binary_indicator.copy())
            Y_test_list.append(one_hot)
    
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    X_test = np.array(X_test_list)
    Y_test = np.array(Y_test_list)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    
    return X_train, Y_train, X_test, Y_test


def compute_frequency_probs(X, Y):
    """
    Compute empirical frequency probabilities for each unique choice set.
    
    This matches the calc_freq function from the PyTorch notebook.
    For each unique choice set (row in X), compute the average of corresponding Y values.
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


def count_parameters(depth, width, n_items):
    """
    Count parameters in the featureless model.
    
    Formula: (depth - 1) * (width^2 + width) + 2 * n_items * width
    """
    # Input layer: n_items -> width
    # Output layer: width -> n_items
    # Residual blocks: (depth - 1) blocks, each with width^2 + width parameters
    return (depth - 1) * (width**2 + width) + n_items * width + width * n_items


def find_width_for_budget(depth, target_budget, n_items, tolerance=0.003):
    """
    Find the width that gives approximately the target parameter budget.
    """
    min_budget = int(target_budget * (1 - tolerance))
    max_budget = int(target_budget * (1 + tolerance))
    
    for width in range(50, 1000):
        params = count_parameters(depth, width, n_items)
        if min_budget <= params <= max_budget:
            return width, params
    
    return None, None


def train_model(model, X_train, Y_train, X_test, Y_test, 
                Y_train_freq, Y_test_freq,
                epochs=500, batch_size=1024, learning_rate=1e-4,
                patience=None):
    """
    Train the DeepHalo model on synthetic data.
    """
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({
        'items': X_train,
        'choices': Y_train
    }).shuffle(10000).batch(batch_size)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Tracking
    train_loss_list = []
    train_freq_loss_list = []
    test_loss_list = []
    test_freq_loss_list = []
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training loop
        epoch_losses = []
        
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                inputs = {'items': batch['items']}
                loss = model.compute_negative_log_likelihood(inputs, batch['choices'])
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_losses.append(loss.numpy())
        
        # Evaluation (no need for tf.no_grad in TensorFlow)
        # Training set
        train_inputs = {'items': tf.constant(X_train, dtype=tf.float32)}
        train_outputs = model(train_inputs, training=False)
        
        train_loss_orig = tf.keras.losses.MSE(Y_train, train_outputs['probabilities'])
        train_loss_orig = tf.sqrt(tf.reduce_mean(train_loss_orig)).numpy()
        
        train_loss_freq = tf.keras.losses.MSE(Y_train_freq, train_outputs['probabilities'])
        train_loss_freq = tf.sqrt(tf.reduce_mean(train_loss_freq)).numpy()
        
        # Test set
        test_inputs = {'items': tf.constant(X_test, dtype=tf.float32)}
        test_outputs = model(test_inputs, training=False)
        
        test_loss_orig = tf.keras.losses.MSE(Y_test, test_outputs['probabilities'])
        test_loss_orig = tf.sqrt(tf.reduce_mean(test_loss_orig)).numpy()
        
        test_loss_freq = tf.keras.losses.MSE(Y_test_freq, test_outputs['probabilities'])
        test_loss_freq = tf.sqrt(tf.reduce_mean(test_loss_freq)).numpy()
        
        train_loss_list.append(train_loss_orig)
        train_freq_loss_list.append(train_loss_freq)
        test_loss_list.append(test_loss_orig)
        test_freq_loss_list.append(test_loss_freq)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss_orig:.4f}, "
                  f"Train Freq: {train_loss_freq:.6f}, "
                  f"Test Loss: {test_loss_orig:.4f}, "
                  f"Test Freq: {test_loss_freq:.6f}")
        
        # Early stopping
        if patience is not None:
            if train_loss_orig < best_loss - 1e-5:
                best_loss = train_loss_orig
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return {
        'train_loss': train_loss_list,
        'train_freq_loss': train_freq_loss_list,
        'test_loss': test_loss_list,
        'test_freq_loss': test_freq_loss_list,
        'final_train_loss': train_loss_list[-1],
        'final_train_freq': train_freq_loss_list[-1],
        'final_test_loss': test_loss_list[-1],
        'final_test_freq': test_freq_loss_list[-1]
    }


def run_experiments(budgets=None, epochs=500, save_results=True):
    """
    Run the full experimental suite matching the paper.
    """
    if budgets is None:
        budgets = {
            '200k': 200000,
            '500k': 500000
        }
    
    n_items = 20
    
    # Generate data (once)
    print("="*70)
    print("Generating Synthetic Data")
    print("="*70)
    
    X_train, Y_train, X_test, Y_test = generate_synthetic_data(
        n_items=20,
        choice_set_size=15,
        train_samples_per_set=80,
        test_samples_per_set=20,
        seed=20
    )
    
    print("\nComputing frequency probabilities...")
    Y_train_freq = compute_frequency_probs(X_train, Y_train)
    Y_test_freq = compute_frequency_probs(X_test, Y_test)
    
    results = {}
    
    for budget_name, budget_value in budgets.items():
        print(f"\n{'='*70}")
        print(f"Testing Budget: {budget_name} ({budget_value:,} parameters)")
        print(f"{'='*70}")
        
        results[budget_name] = {}
        
        # Test depths 3-7
        for depth in range(3, 8):
            width, actual_params = find_width_for_budget(depth, budget_value, n_items)
            
            if width is None:
                print(f"Could not find width for depth {depth}")
                continue
            
            print(f"\nDepth: {depth}, Width: {width}, Parameters: {actual_params:,}")
            
            # Create model
            model = DeepHaloFeatureless(
                n_items=n_items,
                depth=depth,
                width=width,
                block_types=['qua'] * (depth - 1)
            )
            
            # Train model
            history = train_model(
                model, X_train, Y_train, X_test, Y_test,
                Y_train_freq, Y_test_freq,
                epochs=epochs,
                batch_size=1024,
                learning_rate=1e-4,
                patience=50  # Early stopping
            )
            
            results[budget_name][depth] = {
                'width': width,
                'params': actual_params,
                'history': history
            }
            
            print(f"Final Results - Depth {depth}:")
            print(f"  Train RMSE: {history['final_train_loss']:.4f}")
            print(f"  Train Freq RMSE: {history['final_train_freq']:.6f}")
            print(f"  Test RMSE: {history['final_test_loss']:.4f}")
            print(f"  Test Freq RMSE: {history['final_test_freq']:.6f}")
    
    if save_results:
        # Save results
        import json
        results_serializable = {}
        for budget in results:
            results_serializable[budget] = {}
            for depth in results[budget]:
                results_serializable[budget][depth] = {
                    'width': int(results[budget][depth]['width']),
                    'params': int(results[budget][depth]['params']),
                    'final_train_loss': float(results[budget][depth]['history']['final_train_loss']),
                    'final_train_freq': float(results[budget][depth]['history']['final_train_freq']),
                    'final_test_loss': float(results[budget][depth]['history']['final_test_loss']),
                    'final_test_freq': float(results[budget][depth]['history']['final_test_freq'])
                }
        
        with open('/mnt/user-data/outputs/synthetic_experiment_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print("\nResults saved to synthetic_experiment_results.json")
    
    return results


def plot_results(results):
    """
    Plot the results comparing different depths and budgets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for budget_name, budget_results in results.items():
        depths = sorted(budget_results.keys())
        train_losses = [budget_results[d]['history']['final_train_loss'] for d in depths]
        test_losses = [budget_results[d]['history']['final_test_loss'] for d in depths]
        
        axes[0].plot(depths, train_losses, marker='o', label=f'{budget_name} Train')
        axes[0].plot(depths, test_losses, marker='s', label=f'{budget_name} Test', linestyle='--')
    
    axes[0].set_xlabel('Model Depth')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Effect of Model Depth on Approximation Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot frequency loss
    for budget_name, budget_results in results.items():
        depths = sorted(budget_results.keys())
        train_freq = [budget_results[d]['history']['final_train_freq'] for d in depths]
        test_freq = [budget_results[d]['history']['final_test_freq'] for d in depths]
        
        axes[1].plot(depths, train_freq, marker='o', label=f'{budget_name} Train')
        axes[1].plot(depths, test_freq, marker='s', label=f'{budget_name} Test', linestyle='--')
    
    axes[1].set_xlabel('Model Depth')
    axes[1].set_ylabel('Frequency RMSE')
    axes[1].set_title('Effect of Depth on Frequency Approximation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/synthetic_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPlot saved to synthetic_experiment_results.png")


if __name__ == "__main__":
    print("="*70)
    print("DeepHalo Synthetic Data Experiment Replication")
    print("Following Zhang et al. (2025) Section 5.1")
    print("="*70)
    
    # Run experiments with 200k and 500k parameter budgets
    # Using fewer epochs for testing - increase for full replication
    results = run_experiments(
        budgets={
            '200k': 200000,
            '500k': 500000
        },
        epochs=100,  # Use 500 for full replication
        save_results=True
    )
    
    # Plot results
    plot_results(results)
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Budget':<10} {'Depth':<8} {'Width':<8} {'Params':<12} {'Train RMSE':<12} {'Test RMSE':<12}")
    print("-"*70)
    
    for budget_name in results:
        for depth in sorted(results[budget_name].keys()):
            r = results[budget_name][depth]
            print(f"{budget_name:<10} {depth:<8} {r['width']:<8} {r['params']:<12,} "
                  f"{r['history']['final_train_loss']:<12.4f} "
                  f"{r['history']['final_test_loss']:<12.4f}")
    
    print("="*70)
    print("Experiment completed successfully!")
    print("="*70)
