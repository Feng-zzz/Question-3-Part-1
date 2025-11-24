"""
Training and Evaluation Example for DeepHalo Model

This script demonstrates how to train and evaluate the DeepHalo model
on synthetic data, replicating the experiments from Zhang et al. (2025).
"""

import tensorflow as tf
import numpy as np
from deephalo_choicelearn import DeepHaloFeatureBased, DeepHaloFeatureless
import matplotlib.pyplot as plt


def generate_synthetic_feature_data(n_samples=1000, n_items=10, n_features=20, seed=42):
    """
    Generate synthetic choice data with features.
    
    Args:
        n_samples: Number of choice observations
        n_items: Maximum number of items per choice set
        n_features: Number of features per item
        seed: Random seed
        
    Returns:
        features, availabilities, choices
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Generate random features
    features = np.random.randn(n_samples, n_items, n_features).astype(np.float32)
    
    # Generate random availabilities (70-100% items available)
    availabilities = np.random.rand(n_samples, n_items) > 0.3
    availabilities = availabilities.astype(np.float32)
    
    # Ensure at least one item is available
    for i in range(n_samples):
        if availabilities[i].sum() == 0:
            availabilities[i, 0] = 1.0
    
    # Generate choices based on simple utility function (for ground truth)
    # u_j = β^T x_j + ε where ε ~ Gumbel(0, 1)
    beta = np.random.randn(n_features, 1).astype(np.float32)
    utilities = features @ beta  # (n_samples, n_items, 1)
    utilities = utilities.squeeze(-1)  # (n_samples, n_items)
    
    # Add Gumbel noise
    gumbel = np.random.gumbel(0, 1, utilities.shape).astype(np.float32)
    noisy_utilities = utilities + gumbel
    
    # Mask unavailable items
    noisy_utilities = np.where(availabilities > 0, noisy_utilities, -1e9)
    
    # Choose item with highest utility
    choice_indices = np.argmax(noisy_utilities, axis=1)
    choices = np.eye(n_items)[choice_indices].astype(np.float32)
    
    return features, availabilities, choices


def generate_synthetic_featureless_data(n_samples=2000, n_items=4, seed=42):
    """
    Generate synthetic choice data for featureless setting.
    
    This replicates the beverage experiment from Section 5.1 of the paper.
    
    Args:
        n_samples: Number of choice observations
        n_items: Number of items in the universe
        seed: Random seed
        
    Returns:
        item_indicators, choices
    """
    np.random.seed(seed)
    
    # Define market shares for all possible choice sets (Table 1 in paper)
    # Items: 1=Pepsi, 2=Coke, 3=7-Up, 4=Sprite
    market_shares = {
        (1, 2): [0.98, 0.02],
        (1, 3): [0.50, 0.50],
        (1, 4): [0.50, 0.50],
        (2, 3): [0.50, 0.50],
        (2, 4): [0.50, 0.50],
        (3, 4): [0.90, 0.10],
        (1, 2, 3): [0.49, 0.01, 0.50],
        (1, 2, 4): [0.49, 0.01, 0.50],
        (1, 3, 4): [0.50, 0.45, 0.05],
        (2, 3, 4): [0.50, 0.45, 0.05],
        (1, 2, 3, 4): [0.49, 0.01, 0.45, 0.05],
    }
    
    # Generate data
    all_indicators = []
    all_choices = []
    
    for choice_set, probs in market_shares.items():
        # Number of samples for this choice set
        n_set_samples = n_samples // len(market_shares)
        
        for _ in range(n_set_samples):
            # Create indicator vector
            indicator = np.zeros(n_items, dtype=np.float32)
            for item_idx in choice_set:
                indicator[item_idx - 1] = 1.0  # -1 because items are 1-indexed
            
            # Sample choice based on market share
            choice_idx = np.random.choice(len(choice_set), p=probs)
            actual_item = choice_set[choice_idx] - 1
            
            choice = np.zeros(n_items, dtype=np.float32)
            choice[actual_item] = 1.0
            
            all_indicators.append(indicator)
            all_choices.append(choice)
    
    return np.array(all_indicators), np.array(all_choices)


def train_feature_based_model(model, train_data, val_data=None, epochs=100, 
                               batch_size=32, learning_rate=1e-3, verbose=1):
    """
    Train the feature-based DeepHalo model.
    
    Args:
        model: DeepHaloFeatureBased instance
        train_data: Tuple of (features, availabilities, choices)
        val_data: Optional validation data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        verbose: Verbosity level
        
    Returns:
        Training history
    """
    features_train, avail_train, choices_train = train_data
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({
        'features': features_train,
        'availability': avail_train,
        'choices': choices_train
    })
    train_dataset = train_dataset.shuffle(10000).batch(batch_size)
    
    if val_data is not None:
        features_val, avail_val, choices_val = val_data
        val_dataset = tf.data.Dataset.from_tensor_slices({
            'features': features_val,
            'availability': avail_val,
            'choices': choices_val
        })
        val_dataset = val_dataset.batch(batch_size)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': []}
    if val_data is not None:
        history['val_loss'] = []
        history['val_acc'] = []
    
    for epoch in range(epochs):
        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                inputs = {
                    'features': batch['features'],
                    'availability': batch['availability']
                }
                loss = model.compute_negative_log_likelihood(inputs, batch['choices'])
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Compute accuracy
            outputs = model(inputs, training=False)
            predicted = tf.argmax(outputs['probabilities'], axis=-1)
            actual = tf.argmax(batch['choices'], axis=-1)
            acc = tf.reduce_mean(tf.cast(predicted == actual, tf.float32))
            
            epoch_loss += loss.numpy()
            epoch_acc += acc.numpy()
            n_batches += 1
        
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation
        if val_data is not None:
            val_loss = 0.0
            val_acc = 0.0
            n_val_batches = 0
            
            for batch in val_dataset:
                inputs = {
                    'features': batch['features'],
                    'availability': batch['availability']
                }
                loss = model.compute_negative_log_likelihood(inputs, batch['choices'])
                
                outputs = model(inputs, training=False)
                predicted = tf.argmax(outputs['probabilities'], axis=-1)
                actual = tf.argmax(batch['choices'], axis=-1)
                acc = tf.reduce_mean(tf.cast(predicted == actual, tf.float32))
                
                val_loss += loss.numpy()
                val_acc += acc.numpy()
                n_val_batches += 1
            
            val_loss /= n_val_batches
            val_acc /= n_val_batches
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            log_str = f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
            if val_data is not None:
                log_str += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            print(log_str)
    
    return history


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative Log-Likelihood')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("DeepHalo Model Training Example")
    print("=" * 70)
    
    # Experiment 1: Feature-based model
    print("\n1. Training Feature-Based DeepHalo Model")
    print("-" * 70)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    features_train, avail_train, choices_train = generate_synthetic_feature_data(
        n_samples=5000, n_items=10, n_features=20, seed=42
    )
    features_val, avail_val, choices_val = generate_synthetic_feature_data(
        n_samples=1000, n_items=10, n_features=20, seed=123
    )
    
    print(f"Training samples: {len(features_train)}")
    print(f"Validation samples: {len(features_val)}")
    
    # Create model
    model_fb = DeepHaloFeatureBased(
        input_dim=20,
        n_items_max=10,
        embed_dim=64,
        n_layers=4,
        n_heads=8,
        dropout=0.1
    )
    
    # Train model
    print("\nTraining model...")
    history_fb = train_feature_based_model(
        model_fb,
        train_data=(features_train, avail_train, choices_train),
        val_data=(features_val, avail_val, choices_val),
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        verbose=1
    )
    
    print(f"\nFinal Training Loss: {history_fb['train_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history_fb['train_acc'][-1]:.4f}")
    print(f"Final Validation Loss: {history_fb['val_loss'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history_fb['val_acc'][-1]:.4f}")
    
    # Experiment 2: Featureless model (beverage example)
    print("\n2. Training Featureless DeepHalo Model (Beverage Example)")
    print("-" * 70)
    
    # Generate synthetic beverage data
    print("Generating beverage choice data...")
    items_train, choices_train_fl = generate_synthetic_featureless_data(
        n_samples=2000, n_items=4, seed=42
    )
    
    print(f"Training samples: {len(items_train)}")
    
    # Create model with quadratic activation
    model_fl = DeepHaloFeatureless(
        n_items=4,
        depth=3,
        width=20,
        block_types=['qua', 'qua']
    )
    
    # Simple training for featureless model
    print("\nTraining model...")
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    train_dataset_fl = tf.data.Dataset.from_tensor_slices({
        'items': items_train,
        'choices': choices_train_fl
    }).shuffle(1000).batch(32)
    
    for epoch in range(100):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_dataset_fl:
            with tf.GradientTape() as tape:
                inputs = {'items': batch['items']}
                loss = model_fl.compute_negative_log_likelihood(inputs, batch['choices'])
            
            gradients = tape.gradient(loss, model_fl.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_fl.trainable_variables))
            
            epoch_loss += loss.numpy()
            n_batches += 1
        
        epoch_loss /= n_batches
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 - Loss: {epoch_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
