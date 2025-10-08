import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import cv2
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import logging

from mad_drift_rl.mad_drift_env import MadDriftEnv, ActionSpace
from mad_drift_rl.nox_player import NoxPlayer
from mad_drift_rl.models.agent_model import AgentModel

console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)]
)

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 10
    mini_batch_size: int = 64
    episodes_per_batch: int = 10
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    interval_loss_coef: float = 0.1
    interval_entropy_coef: float = 0.01
    interval_noise_std: float = 0.02  # Gaussian noise std for interval exploration

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.memory_states = []
        self.intervals = []
        self.episode_rewards = []  # Track which episode each timestep belongs to

    def add(self, state, action, reward, value, log_prob, done, memory_state, interval, episode_reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.memory_states.append(memory_state)
        self.intervals.append(interval)
        self.episode_rewards.append(episode_reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.memory_states.clear()
        self.intervals.clear()
        self.episode_rewards.clear()

    def compute_advantages(self, config: PPOConfig):
        """Compute GAE advantages"""
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + config.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + config.gamma * config.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        return np.array(advantages), np.array(returns)

class Trainer:
    def __init__(self, config: PPOConfig = PPOConfig()):
        self.config = config

        # Initialize Nox emulator
        if not NoxPlayer.check_availability():
            raise RuntimeError("NoxConsole not available")

        emulators = NoxPlayer.list_emulators()
        if not emulators:
            raise RuntimeError("No Nox emulators found")

        self.emulator = emulators[0]
        if not self.emulator.is_running:
            NoxPlayer.launch(self.emulator)

        logging.info(f"Using emulator: {self.emulator.name}")

        # Initialize environment and model
        self.env = MadDriftEnv(self.emulator)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # n_latent = 208 (16x13 spatial patches from deeper CNN)
        self.model = AgentModel(feature_dim=128, n_latent=208, n_layer=2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize model with dummy input to setup lazy layers
        dummy_obs = torch.zeros(1, 2, 133, 108).to(self.device)
        dummy_memory = torch.zeros(1, 208, 128).to(self.device)
        with torch.no_grad():
            self.model(dummy_obs, dummy_memory)

        self.memory = PPOMemory()

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Checkpoints
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_avg_score = -float('inf')

    def normalize_reward(self, reward):
        """Online reward normalization"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt((self.reward_std ** 2 * (self.reward_count - 1) + delta * delta2) / self.reward_count)

        return (reward - self.reward_mean) / (self.reward_std + 1e-8)

    def collect_episode(self, retry_count=0):
        """Collect one episode of experience"""
        import time as time_module
        start_time = time_module.time()

        obs = self.env.reset()
        # obs shape: (133, 108, 2) -> permute to (2, 133, 108) -> add batch dim: (1, 2, 133, 108)
        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        memory_state = torch.zeros(1, self.model.n_latent, self.model.feature_dim).to(self.device)

        episode_reward = 0
        episode_length = 0
        episode_intervals = []
        inference_times = []
        episode_transitions = []  # Store transitions for this episode

        while True:
            with torch.no_grad():
                inference_start = time_module.time()
                memory_state, action_logits, value, pooled_features = self.model(obs_tensor, memory_state)
                memory_state = memory_state.detach()  # Detach to prevent memory leak
                inference_time = time_module.time() - inference_start
                inference_times.append(inference_time)

                # Sample action from policy
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Compute action interval using cached pooled features
                interval = self.model.get_interval(pooled_features, action)

                # Add Gaussian noise for exploration
                noise = torch.randn_like(interval) * self.config.interval_noise_std
                interval_noisy = torch.clamp(interval + noise, 0.0, 0.05)  # Clamp to [0, 0.05]

            action_value = action.item()
            interval_value = interval_noisy.item()
            action_name = ActionSpace(action_value).name

            action_start = time_module.time()
            reward, next_obs, done, info = self.env.step(ActionSpace(action_value), interval_value)
            action_duration = time_module.time() - action_start

            logging.info(f"Action: {action_name}, Interval: {interval_value:.3f}s, Inference: {inference_time*1000:.2f}ms, Duration: {action_duration:.3f}s")

            # Visualize observation AFTER step (only if not done, to avoid showing game over screen)
            # Note: we visualize the observation that was used to make the decision
            if not done:
                # obs shape: (133, 108, 2) - channel 0 is car, channel 1 is environment
                car_channel = obs[:, :, 0]  # (133, 108) binary car mask
                env_channel = obs[:, :, 1]  # (133, 108) environment grayscale

                # Create RGB visualization: Bright green for car, White for environment
                obs_display = np.zeros((133, 108, 3), dtype=np.float32)

                # Create car mask (boolean)
                car_mask = car_channel > 0.5

                # Set environment to white (where there's no car)
                obs_display[:, :, 0] = np.where(car_mask, 0, env_channel)  # Red = environment only
                obs_display[:, :, 1] = np.where(car_mask, 1.0, env_channel)  # Green = car (full intensity) or environment
                obs_display[:, :, 2] = np.where(car_mask, 0, env_channel)  # Blue = environment only

                # Convert to uint8 and resize
                obs_display = (obs_display * 255).astype(np.uint8)
                # Resize to match original game view aspect ratio (540x665, cropped from screenshot[90:755, :])
                obs_display = cv2.resize(obs_display, (540, 665), interpolation=cv2.INTER_NEAREST)

                # Add action text with interval
                cv2.putText(obs_display, f"Action: {action_name}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(obs_display, f"Interval: {interval_value:.3f}s", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Agent View", obs_display)
                cv2.waitKey(1)

            # Store transition temporarily (will add to memory after episode completes)
            # Store the ORIGINAL interval (without noise) for training
            episode_transitions.append({
                'obs_tensor': obs_tensor.cpu().numpy(),
                'action': action.item(),
                'reward': reward,
                'value': value.item(),
                'log_prob': log_prob.item(),
                'done': done,
                'memory_state': memory_state.cpu().numpy(),
                'interval': interval.item()  # Store original interval (without noise)
            })

            episode_reward += reward
            episode_length += 1
            episode_intervals.append(interval_value)

            if done:
                # Check if OCR failed - if so, discard this episode and resample (with retry limit)
                if info.get('ocr_failed', False):
                    if retry_count < 3:
                        logging.warning(f"OCR failed to read score, resampling episode (retry {retry_count + 1}/3)...")
                        return self.collect_episode(retry_count + 1)  # Retry with incremented count
                    else:
                        logging.error("OCR failed 3 times, using score=0 and continuing...")
                        info['final_score'] = 0
                        info['ocr_failed'] = True  # Keep flag for logging

                # Add all transitions from this episode to memory with episode reward
                for transition in episode_transitions:
                    self.memory.add(
                        transition['obs_tensor'],
                        transition['action'],
                        transition['reward'],
                        transition['value'],
                        transition['log_prob'],
                        transition['done'],
                        transition['memory_state'],
                        transition['interval'],
                        episode_reward  # Tag each timestep with total episode reward
                    )

                elapsed_time = time_module.time() - start_time
                actions_per_second = episode_length / elapsed_time if elapsed_time > 0 else 0
                info['actions_per_second'] = actions_per_second
                info['avg_interval'] = np.mean(episode_intervals)
                info['avg_inference_time'] = np.mean(inference_times)
                info['max_inference_time'] = np.max(inference_times)
                return episode_reward, episode_length, info

            # Safety check: ensure next observation is valid (not None from game over screen)
            # This should never happen due to early return in env.step(), but guard against it
            if next_obs is None:
                raise RuntimeError("Received None observation but episode not marked as done - this should not happen")

            obs = next_obs
            # obs shape: (133, 108, 2) -> permute to (2, 133, 108) -> add batch dim: (1, 2, 133, 108)
            obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def train_step(self):
        """Perform PPO update"""
        advantages, returns = self.memory.compute_advantages(self.config)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.concatenate(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        memory_states = torch.FloatTensor(np.concatenate(self.memory.memory_states)).to(self.device)
        stored_intervals = torch.FloatTensor(self.memory.intervals).to(self.device)

        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_interval_loss = 0
        num_updates = 0

        for _ in range(self.config.epochs_per_batch):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_memory_states = memory_states[batch_indices]
                batch_stored_intervals = stored_intervals[batch_indices]

                # Forward pass
                _, action_logits, values, pooled_features = self.model(batch_states, batch_memory_states)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)

                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()

                # Interval loss - only train on successful episodes
                # Compute reward threshold (e.g., top 50% of episodes)
                episode_rewards_tensor = torch.FloatTensor(self.memory.episode_rewards).to(self.device)
                batch_episode_rewards = episode_rewards_tensor[batch_indices]
                reward_threshold = torch.median(episode_rewards_tensor)

                # Create mask for successful timesteps
                success_mask = batch_episode_rewards >= reward_threshold

                if success_mask.sum() > 0:
                    # Only compute interval loss on successful episodes
                    predicted_intervals = self.model.get_interval(pooled_features, batch_actions)
                    interval_mse = (predicted_intervals - batch_stored_intervals) ** 2
                    interval_loss = (interval_mse * success_mask).sum() / (success_mask.sum() + 1e-8)
                else:
                    # No successful episodes in this batch, skip interval training
                    interval_loss = torch.tensor(0.0, device=self.device)

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)

                # Total loss
                loss = (policy_loss +
                       self.config.value_loss_coef * value_loss -
                       self.config.entropy_coef * entropy +
                       self.config.interval_loss_coef * interval_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_interval_loss += interval_loss.item()
                num_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "interval_loss": total_interval_loss / num_updates,
        }

        self.memory.clear()
        return metrics

    def save_checkpoint(self, iteration, avg_score, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'avg_score': avg_score,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }

        path = self.checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with avg score: {avg_score:.2f}")

    def train(self):
        """Main training loop"""
        wandb.init(
            project="mad-drift-rl",
            config={
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "clip_epsilon": self.config.clip_epsilon,
                "epochs_per_batch": self.config.epochs_per_batch,
                "mini_batch_size": self.config.mini_batch_size,
                "episodes_per_batch": self.config.episodes_per_batch,
            }
        )

        iteration = 0
        avg_score = 0  # Initialize for early interruption handling

        try:
            while True:
                # Collect episodes
                episode_rewards = []
                episode_lengths = []
                final_scores = []

                logging.info(f"Collecting {self.config.episodes_per_batch} episodes...")
                for ep in range(self.config.episodes_per_batch):
                    reward, length, info = self.collect_episode()
                    episode_rewards.append(reward)
                    episode_lengths.append(length)
                    if 'final_score' in info:
                        final_scores.append(info['final_score'])

                    wandb.log({
                        "episode/reward": reward,
                        "episode/length": length,
                        "episode/final_score": info.get('final_score', 0),
                        "episode/timeout": info.get('timeout', False),
                        "episode/actions_per_second": info.get('actions_per_second', 0),
                        "episode/avg_interval": info.get('avg_interval', 0),
                        "episode/avg_inference_time": info.get('avg_inference_time', 0),
                        "episode/max_inference_time": info.get('max_inference_time', 0),
                    })

                    logging.info(f"Episode {ep+1}/{self.config.episodes_per_batch}: reward={reward:.2f}, length={length}, score={info.get('final_score', 0)}, aps={info.get('actions_per_second', 0):.2f}, avg_interval={info.get('avg_interval', 0):.3f}s, avg_inference={info.get('avg_inference_time', 0)*1000:.2f}ms")

                # Train
                logging.info("Training model...")
                metrics = self.train_step()

                avg_reward = np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths)
                avg_score = np.mean(final_scores) if final_scores else 0

                wandb.log({
                    "train/policy_loss": metrics['policy_loss'],
                    "train/value_loss": metrics['value_loss'],
                    "train/entropy": metrics['entropy'],
                    "train/interval_loss": metrics['interval_loss'],
                    "train/avg_reward": avg_reward,
                    "train/avg_length": avg_length,
                    "train/avg_score": avg_score,
                    "train/iteration": iteration,
                })

                logging.info(f"Iteration {iteration}: avg_reward={avg_reward:.2f}, avg_score={avg_score:.2f}, policy_loss={metrics['policy_loss']:.4f}, interval_loss={metrics['interval_loss']:.4f}")

                # Save checkpoint
                if iteration % 50 == 0:
                    is_best = avg_score > self.best_avg_score
                    if is_best:
                        self.best_avg_score = avg_score
                    self.save_checkpoint(iteration, avg_score, is_best)

                iteration += 1

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint(iteration, avg_score)
        finally:
            cv2.destroyAllWindows()
            wandb.finish()

if __name__ == "__main__":
    Trainer().train()