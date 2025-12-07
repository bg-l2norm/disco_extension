import collections
import numpy as np
import jax
import rlax
from disco_rl import types
from disco_rl import utils

class SimpleReplayBuffer:
  """A simple FIFO replay buffer for JAX arrays."""

  def __init__(self, capacity: int, seed: int):
    """Initializes the buffer.
    
    Args:
      capacity: The maximum number of items the buffer can hold.
      seed: Seed for the random number generator.
    """
    self.buffer = collections.deque(maxlen=capacity)
    self.capacity = capacity
    self.np_rng = np.random.default_rng(seed)

  def add(self, rollout: types.ActorRollout) -> None:
    """Appends a batch of trajectories to the buffer.
    
    Args:
      rollout: The actor rollout containing trajectories to add.
    """
    rollout = jax.device_get(rollout)
    # Split the tree across the batch dimension (axis 2)
    split_tree = rlax.tree_split_leaves(rollout, axis=2)
    self.buffer.extend(split_tree)

  def sample(self, batch_size: int) -> types.ActorRollout | None:
    """Samples a batch of trajectories from the buffer.
    
    Args:
      batch_size: The number of trajectories to sample.
      
    Returns:
      A batch of sampled trajectories or None if buffer is empty.
    """
    buffer_size = len(self.buffer)
    if buffer_size == 0:
      print("Warning: Trying to sample from an empty buffer.")
      return None

    indices = self.np_rng.integers(buffer_size, size=batch_size)
    batched_samples = utils.tree_stack(
        [self.buffer[i] for i in indices], axis=2
    )
    return batched_samples

  def __len__(self) -> int:
    """Returns the current number of transitions in the buffer."""
    return len(self.buffer)