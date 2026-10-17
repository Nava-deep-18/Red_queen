import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Linear layers match exact checkpoint state_dict keys:
        # actor.0 (Linear), actor.2 (Linear), actor.4 (Linear)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def _get_masked_logits(self, state, action_mask=None):
        logits = self.actor(state)
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            else:
                mask = action_mask.to(dtype=torch.bool, device=logits.device)
            # Mask out invalid action logits with a large negative number
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
        return logits

    def act(self, state, action_mask=None, evaluate=False):
        logits = self._get_masked_logits(state, action_mask)
        dist = Categorical(logits=logits)

        if evaluate:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, action_mask=None):
        logits = self._get_masked_logits(state, action_mask)
        dist = Categorical(logits=logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        minibatch_size=64,
        max_grad_norm=0.5,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # Rollout Buffer
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

    def select_action(self, state, action_mask=None, evaluate=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(
                state_tensor, action_mask=action_mask, evaluate=evaluate
            )

        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        if action_mask is not None:
            self.masks.append(torch.tensor(action_mask, dtype=torch.bool))
        else:
            self.masks.append(None)

        return action.item()

    def predict(self, state, action_mask=None):
        """Pure greedy action selection with optional action masking for clean evaluation."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            logits = self.policy.actor(state_tensor)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
                logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device))
            return torch.argmax(logits, dim=-1).item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def update(self):
        # Guard: nothing to learn from an empty buffer
        if not self.rewards:
            return

        # 1. True Monte Carlo returns (unnormalized)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        old_states = torch.stack(self.states, dim=0).detach()
        old_actions = torch.stack(self.actions, dim=0).detach()
        old_logprobs = torch.stack(self.logprobs, dim=0).detach()

        has_masks = any(m is not None for m in self.masks)
        if has_masks:
            default_mask = torch.ones(self.policy.actor[-1].out_features, dtype=torch.bool)
            old_masks = torch.stack(
                [m if m is not None else default_mask for m in self.masks], dim=0
            ).detach()
        else:
            old_masks = None

        # 2. Proper PPO Advantage Calculation & Normalization
        with torch.no_grad():
            old_values = self.policy.critic(old_states).squeeze(-1)
            raw_advantages = returns - old_values
            adv_std = raw_advantages.std(unbiased=False)
            if adv_std > 1e-7:
                advantages = (raw_advantages - raw_advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = raw_advantages - raw_advantages.mean()

        # 3. Minibatch PPO Optimization for K epochs
        batch_size = old_states.size(0)
        minibatch_sz = min(self.minibatch_size, batch_size)

        for _ in range(self.K_epochs):
            indices = torch.randperm(batch_size)
            for start_idx in range(0, batch_size, minibatch_sz):
                batch_idx = indices[start_idx : start_idx + minibatch_sz]

                b_states = old_states[batch_idx]
                b_actions = old_actions[batch_idx]
                b_logprobs = old_logprobs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_masks = old_masks[batch_idx] if old_masks is not None else None

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    b_states, b_actions, b_masks
                )
                state_values = state_values.squeeze(-1)

                ratios = torch.exp(logprobs - b_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.MseLoss(state_values, b_returns)
                entropy_loss = -0.01 * dist_entropy.mean()

                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                # 4. Essential PPO Gradient Clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)

                self.optimizer.step()

        # Sync old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.masks.clear()
