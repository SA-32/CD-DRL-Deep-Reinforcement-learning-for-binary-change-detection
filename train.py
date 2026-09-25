"""
train.py
--------
Implements Algorithm 1 ("Training PixelDRL-MG Using a Dynamic Iterative
Update Policy") from the paper.

Mapping from the paper's Algorithm 1 to this code:

  Input: T_max, t_max, (X, G), segmentation network M(theta_s)          -> `train()` args
  Line 1: sample a training batch (x, g)                                -> dataloader batch
  Line 2/6: reset gradients                                             -> optimizer.zero_grad()
  Line 3/7: thread-specific params theta_p', theta_v', theta_s'         -> (single synchronous
             synchronized with global theta_p, theta_v, theta_s            copy of the model;
                                                                             true A3C's per-thread
                                                                             copies are replaced
                                                                             by a batched rollout,
                                                                             see module docstring
                                                                             in env.py)
  Line 9: obtain state s_i^(t) for all i                                 -> PixelEnv.init_mask /
                                                                             temp_input
  Lines 10-16: rollout for t_start..t_max, performing actions, storing   -> rollout loop below
               rewards/new states, computing segmentation metrics
  Line 17: bootstrap R_i from terminal or non-terminal value             -> compute_targets()
  Lines 18-26: backward accumulation of discounted returns & gradients   -> policy/value losses
               for theta_p, theta_v, theta_s                                + one backward() call
  Line 27-28: update parameters, update M via gradient descent           -> optimizer.step()
"""

import copy
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import LEVIRCDPatchDataset
from env import PixelEnv, NeighborhoodAggregator, compute_targets
from metrics import compute_all_metrics, average_metrics
from model import PixelDRLMG
from torchvision import transforms


def rollout_episode(model, env, aggregator, pre_change_image, post_change_image, gt, t_max, gamma, device):

    B, _, H, W = pre_change_image.shape
    mask = env.init_mask(pre_change_image)

    log_probs, entropies, values, rewards = [], [], [], []

    # NEW
    mask_history = [mask.clone()]

    for t in range(t_max):
        x_t = torch.cat([pre_change_image, post_change_image, mask], dim = 1)

        action, log_prob, entropy, value, _ = model.act(x_t, t)

        (action==0).float().mean()

        new_mask_2d = env.step(mask.squeeze(1), action)

        r = env.reward(mask.squeeze(1), new_mask_2d, gt.squeeze(1))

        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        rewards.append(r)

        mask = new_mask_2d.unsqueeze(1)

        # NEW
        mask_history.append(mask.clone())

    with torch.no_grad():
        x_T = torch.cat([pre_change_image, post_change_image, mask], dim = 1)
        _, _, _, _, v_T = model.act(x_T, t=t_max, greedy=True)
    values.append(v_T)

    return log_probs, entropies, values, rewards, mask, mask_history


def compute_losses(log_probs, entropies, values, returns, entropy_coef, value_loss_coef):
    """
    PA3C losses in matrix form (Eqs. 13-16):

        A(a,s) = R - V(s)                                    (Eq. 14)
        dtheta_v ~ (R - V(s))^2                               (Eq. 13/9)
        dtheta_p ~ -log(pi(a|s)) * A(a,s)                     (Eq. 15/11)
        dtheta_s = dtheta_p + dtheta_v (shared backbone)      (Eq. 16)
    """
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    T = len(returns)

    for t in range(T):
        R = returns[t].detach()
        V = values[t]
        advantage = R - V                                    # Eq. (14) / (10)

        value_loss   = value_loss + (advantage ** 2).mean()     # Eq. (13) / (9)
        policy_loss  = policy_loss + (-log_probs[t] * advantage.detach()).mean()  # Eq. (15)/(11)
        entropy_loss = entropy_loss - entropies[t].mean()

    policy_loss = policy_loss / T
    value_loss = value_loss / T
    entropy_loss = entropy_loss / T

    total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss.item(), value_loss.item()

import matplotlib.pyplot as plt

def plot_rollout(image, gt, mask_history, epoch):

    n = len(mask_history)

    plt.figure(figsize=(3*n, 6))

    # Input image
    plt.subplot(2, n+2, 1)
    plt.imshow(image[0,0].cpu(), cmap='gray')
    plt.title("Input")
    plt.axis("off")

    # Ground truth
    plt.subplot(2, n+2, 2)
    plt.imshow(gt[0,0].cpu(), cmap='gray')
    plt.title("GT")
    plt.axis("off")

    # Masks
    for i, mask in enumerate(mask_history):

        plt.subplot(2, n+2, i+3)
        plt.imshow(mask[0,0].detach().cpu(), cmap='gray', vmin=0, vmax=1)
        plt.title(f"t={i}")
        plt.axis("off")

    os.makedirs("rollouts", exist_ok=True)

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"rollouts/epoch_{epoch}.png")
    plt.close()

def train(cfg: Config,  k_shot=None):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])
    
    root_dir = "../../levir-cd-256"

    train_ds = LEVIRCDPatchDataset(root_dir,mask_transform = transform)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)

    model = PixelDRLMG().to(device)

    aggregator = NeighborhoodAggregator().to(device)
    env = PixelEnv(device)

    params = list(model.parameters()) + list(aggregator.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_every,
                                                 gamma=cfg.lr_decay_gamma)

    global_step = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_start = time.time()
        running_loss, running_dice = 0.0, 0.0

        for batch_idx, (pre_change_image, post_change_image, gt) in enumerate(train_loader):
            pre_change_image, post_change_image, gt = pre_change_image.to(device), post_change_image.to(device), gt.to(device)

            optimizer.zero_grad()
            
            log_probs, entropies, values, rewards, final_mask, mask_history = rollout_episode(
                model, env, aggregator, pre_change_image, post_change_image, gt, cfg.t_max, cfg.gamma, device
            )

            returns = compute_targets(rewards, values, cfg.gamma, aggregator)

            total_loss, p_loss, v_loss = compute_losses(
                log_probs, entropies, values, returns, cfg.entropy_coef, cfg.value_loss_coef
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                pred = (final_mask > 0.5).float()
                dice = 0.0
                for b in range(pred.shape[0]):
                    m = compute_all_metrics(pred[b, 0], gt[b, 0])
                    dice += m["DICE"]
                dice /= pred.shape[0]

            running_loss += total_loss.item()
            running_dice += dice
            global_step += 1

            if batch_idx % cfg.log_every == 0:
                print(f"[epoch {epoch:03d}][batch {batch_idx:04d}] "
                      f"loss={total_loss.item():.4f} (p={p_loss:.4f}, v={v_loss:.4f}) "
                      f"dice={dice:.4f}")

            if batch_idx == 0:
                plot_rollout(pre_change_image, gt, mask_history, epoch)

        scheduler.step()
        n_batches = len(train_loader)
        print(f"== Epoch {epoch:03d} done in {time.time()-epoch_start:.1f}s | "
              f"avg_loss={running_loss/n_batches:.4f} avg_dice={running_dice/n_batches:.4f} ==")

        if (epoch + 1) % 25 == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"pixeldrl_mg_epoch{epoch+1}.pt")
            torch.save({"model": model.state_dict(),
                        "aggregator": aggregator.state_dict(),
                        "epoch": epoch}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    return model, aggregator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_sam", action="store_true")
    parser.add_argument("--no_dc", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.no_sam:
        cfg.use_sam = False
    if args.no_dc:
        cfg.use_dc = False

    train(cfg,  k_shot= None)
