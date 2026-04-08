import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from modules.rewards import init_scorer, get_self_critical_reward


def test_reward_shape():
    """Reward tensor should match [batch, seq_len]."""
    init_scorer()

    batch_size = 2
    seq_len = 10

    # fake token id sequences (non-zero = real token, 0 = end)
    greedy_res = torch.randint(1, 100, (batch_size, seq_len))
    gen_result  = torch.randint(1, 100, (batch_size, seq_len))
    data_gts    = torch.randint(1, 100, (batch_size, seq_len))

    rewards = get_self_critical_reward(greedy_res, data_gts, gen_result, alpha=0.5)

    assert rewards.shape == (batch_size, seq_len), \
        f"Expected shape {(batch_size, seq_len)}, got {rewards.shape}"
    print(f"Reward shape: {rewards.shape}  ✓")


def test_reward_range():
    """Rewards are advantages (can be negative), but should be finite."""
    init_scorer()

    batch_size = 2
    seq_len = 15

    greedy_res = torch.randint(1, 50, (batch_size, seq_len))
    gen_result  = torch.randint(1, 50, (batch_size, seq_len))
    data_gts    = torch.randint(1, 50, (batch_size, seq_len))

    rewards = get_self_critical_reward(greedy_res, data_gts, gen_result, alpha=0.5)

    assert np.all(np.isfinite(rewards)), "Rewards contain NaN or Inf"
    print(f"Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]  ✓")


if __name__ == '__main__':
    test_reward_shape()
    test_reward_range()
    print("\nAll reward tests passed!")
