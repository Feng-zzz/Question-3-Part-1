"""
Quick test to verify all TensorFlow issues are resolved
"""

import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/claude')
from deephalo_choicelearn import DeepHaloFeatureless

print("Testing TensorFlow operations...")

# Test 1: Basic model creation
print("\n1. Creating model...")
model = DeepHaloFeatureless(n_items=10, depth=3, width=20, block_types=['qua', 'qua'])
print("   ✓ Model created successfully")

# Test 2: Forward pass
print("\n2. Testing forward pass...")
X = np.random.rand(5, 10).astype(np.float32)
X = (X > 0.5).astype(np.float32)  # Binary
inputs = {'items': tf.constant(X)}
outputs = model(inputs, training=False)
print(f"   ✓ Forward pass successful, output shape: {outputs['probabilities'].shape}")

# Test 3: Loss computation
print("\n3. Testing loss computation...")
Y = np.eye(10)[np.random.randint(0, 10, 5)].astype(np.float32)
loss = model.compute_negative_log_likelihood(inputs, Y)
print(f"   ✓ Loss computed: {loss.numpy():.4f}")

# Test 4: Gradient computation
print("\n4. Testing gradient computation...")
with tf.GradientTape() as tape:
    loss = model.compute_negative_log_likelihood(inputs, Y)
gradients = tape.gradient(loss, model.trainable_variables)
print(f"   ✓ Gradients computed, {len(gradients)} variables")

# Test 5: Training step
print("\n5. Testing training step...")
optimizer = tf.keras.optimizers.Adam(1e-3)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
print("   ✓ Training step successful")

# Test 6: Evaluation mode (no tf.no_grad needed)
print("\n6. Testing evaluation mode...")
outputs = model(inputs, training=False)
mse = tf.keras.losses.MSE(Y, outputs['probabilities'])
rmse = tf.sqrt(tf.reduce_mean(mse)).numpy()
print(f"   ✓ Evaluation successful, RMSE: {rmse:.4f}")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nNo TensorFlow errors. Implementation working correctly.")
