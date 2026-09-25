import torch
import torch.nn as nn


class NeighborhoodAggregator(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, v_next):
        # Accept either (B,H,W) or (B,1,H,W)
        if v_next.ndim == 3:
            v_next = v_next.unsqueeze(1)
        elif v_next.ndim != 4:
            raise ValueError(f"Unexpected shape: {v_next.shape}")

        return self.conv(v_next).squeeze(1)


class PixelEnv:
    """
    Stateless helper implementing the dynamic iterative update policy.
    Keeps no internal state itself; the rollout loop in train.py owns
    the (image, mask, ground_truth) tensors and calls these static-like
    methods at every time step.
    """

    def __init__(self, device):
        self.device = device

    @staticmethod
    def init_mask(image):
        """m^(0) = ones_like(image); s_i^(0) = I_i as stated in the paper."""
        return torch.ones_like(image[:, :1, :, :])

    @staticmethod
    def temp_input(image, mask):
        """X^(t) = image * m^(t)"""
        return image * mask

    @staticmethod
    def step(mask, action):
        """
        Apply actions to obtain m^(t+1).
          action == 0 -> background (0)
          action == 1 -> do nothing (keep current mask value)
        action: (B, H, W) long tensor in {0, 1}
        mask:   (B, H, W) float tensor
        """
        return action.float()
        # keep = action.float()           # 1 where "do nothing", 0 where "set background"
        # new_mask = mask * keep
        # return new_mask

    @staticmethod
    def reward(prev_mask, new_mask, gt):
        """
        Eq. (12): r^(t) = ||f^(t-1)-G||^2 - ||f^(t)-G||^2, computed per-pixel
        (squared error at each pixel location, not summed), which is the
        natural pixel-level analogue used with Eqs. (9)/(11)'s per-pixel
        averaging over N pixels.
        """
        prev_err = (prev_mask - gt) ** 2
        new_err = (new_mask - gt) ** 2
        return prev_err - new_err       # (B, H, W)


def compute_targets(rewards, values, gamma, aggregator: NeighborhoodAggregator):
    """
    Computes R_i^(t) (Eq. 8) for every step of a rolled-out episode, in
    matrix form, following Algorithm 1's backward accumulation (lines 17-19)
    combined with the neighbourhood bootstrapping of Eq. (8).

    rewards: list of length T, each (B, H, W)         -- r^(t), t = 0..T-1
    values:  list of length T+1, each (B, H, W)        -- V(s^(t)), t = 0..T
             (values[T] is the value of the terminal state, used for
             bootstrapping the last step; for a non-terminal rollout this is
             simply V(s^(T)) from a final forward pass, matching Algorithm 1
             line 17: R_i = V(s_i^{(t)}; theta') for non-terminal states.)
    gamma:   discount factor
    aggregator: NeighborhoodAggregator module implementing sum_j w_{i-j} V_j

    Returns:
        returns: list of length T, each (B, H, W) -- R^(t) for t = 0..T-1
    """
    T = len(rewards)
    returns = [None] * T
    # Bootstrap from the last value estimate (Algorithm 1, line 17-19)
    R = values[T].detach()
    for t in reversed(range(T)):
        R = rewards[t] + gamma * aggregator(R)
        returns[t] = R
    return returns
