"""
Advantage-weighted regression (AWR) for diffusion policy.

Advantage = discounted-reward-to-go - V(s)

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
from copy import deepcopy

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts


def td_values(
    states,
    rewards,
    dones,
    state_values,
    gamma=0.99,
    alpha=0.95,
    lam=0.95,
):
    """
    Gives a list of TD estimates for a given list of samples from an RL environment.
    The TD(λ) estimator is used for this computation.

    :param replay_buffers: The replay buffers filled by exploring the RL environment.
    Includes: states, rewards, "final state?"s.
    :param state_values: The currently estimated state values.
    :return: The TD estimates.
    """
    sample_count = len(states)
    tds = np.zeros_like(state_values, dtype=np.float32)
    dones[-1] = 1
    next_value = 1 - dones[-1]

    val = 0.0
    for i in range(sample_count - 1, -1, -1):
        # next_value = 0.0 if dones[i] else state_values[i + 1]

        # get next_value for vectorized
        if i < sample_count - 1:
            next_value = state_values[i + 1]
            next_value = next_value * (1 - dones[i])

        state_value = state_values[i]
        error = rewards[i] + gamma * next_value - state_value
        val = alpha * error + gamma * lam * (1 - dones[i]) * val

        tds[i] = val + state_value
    return tds


class TrainAWRDiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Reward exponential
        self.beta = cfg.train.beta

        # Max weight for AWR
        self.max_adv_weight = cfg.train.max_adv_weight

        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_update_ratio = cfg.train.critic_update_ratio

    def run(self):

        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        done_buffer = deque(maxlen=self.buffer_size)
        first_buffer = deque(maxlen=self.buffer_size)

        # Start training loop
        timer = Timer()
        run_results = []
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

            # Reset env at the beginning of an iteration
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = (
                    done_venv  # if done at the end of last iteration, then the envs are just reset
                )
            last_itr_eval = eval_mode
            reward_trajs = np.empty((0, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    samples = (
                        self.model(
                            cond=torch.from_numpy(prev_obs_venv)
                            .float()
                            .to(self.device),
                            deterministic=eval_mode,
                        )
                        .cpu()
                        .numpy()
                    )  # n_env x horizon x act
                action_venv = samples[:, : self.act_steps]

                # Apply multi-step action
                obs_venv, reward_venv, done_venv, info_venv = self.venv.step(
                    action_venv
                )
                reward_trajs = np.vstack((reward_trajs, reward_venv[None]))

                # add to buffer
                obs_buffer.append(prev_obs_venv)
                action_buffer.append(action_venv)
                reward_buffer.append(reward_venv * self.scale_reward_factor)
                done_buffer.append(done_venv)
                first_buffer.append(firsts_trajs[step])

                firsts_trajs[step + 1] = done_venv
                prev_obs_venv = obs_venv

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update
            if not eval_mode:

                obs_trajs = np.array(deepcopy(obs_buffer))
                reward_trajs = np.array(deepcopy(reward_buffer))
                dones_trajs = np.array(deepcopy(done_buffer))

                obs_t = einops.rearrange(
                    torch.from_numpy(obs_trajs).float().to(self.device),
                    "s e h d -> (s e) h d",
                )
                values_t = np.array(self.model.critic(obs_t).detach().cpu().numpy())
                values_trajs = values_t.reshape(-1, self.n_envs)
                td_trajs = td_values(obs_trajs, reward_trajs, dones_trajs, values_trajs)

                # flatten
                obs_trajs = einops.rearrange(
                    obs_trajs,
                    "s e h d -> (s e) h d",
                )
                td_trajs = einops.rearrange(
                    td_trajs,
                    "s e -> (s e)",
                )

                # Update policy and critic
                num_batch = int(
                    self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
                )
                for _ in range(num_batch // self.critic_update_ratio):

                    # Sample batch
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = torch.from_numpy(obs_trajs[inds]).float().to(self.device)
                    td_b = torch.from_numpy(td_trajs[inds]).float().to(self.device)

                    # Update critic
                    loss_critic = self.model.loss_critic(obs_b, td_b)
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic_optimizer.step()

                obs_trajs = np.array(deepcopy(obs_buffer))
                samples_trajs = np.array(deepcopy(action_buffer))
                reward_trajs = np.array(deepcopy(reward_buffer))
                dones_trajs = np.array(deepcopy(done_buffer))

                obs_t = einops.rearrange(
                    torch.from_numpy(obs_trajs).float().to(self.device),
                    "s e h d -> (s e) h d",
                )
                values_t = np.array(self.model.critic(obs_t).detach().cpu().numpy())
                values_trajs = values_t.reshape(-1, self.n_envs)
                td_trajs = td_values(obs_trajs, reward_trajs, dones_trajs, values_trajs)
                advantages_trajs = td_trajs - values_trajs

                # flatten
                obs_trajs = einops.rearrange(
                    obs_trajs,
                    "s e h d -> (s e) h d",
                )
                samples_trajs = einops.rearrange(
                    samples_trajs,
                    "s e h d -> (s e) h d",
                )
                advantages_trajs = einops.rearrange(
                    advantages_trajs,
                    "s e -> (s e)",
                )

                for _ in range(num_batch):

                    # Sample batch
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = torch.from_numpy(obs_trajs[inds]).float().to(self.device)
                    actions_b = (
                        torch.from_numpy(samples_trajs[inds]).float().to(self.device)
                    )
                    advantages_b = (
                        torch.from_numpy(advantages_trajs[inds]).float().to(self.device)
                    )
                    advantages_b = (advantages_b - advantages_b.mean()) / (
                        advantages_b.std() + 1e-6
                    )
                    advantages_b_scaled = torch.exp(self.beta * advantages_b)
                    advantages_b_scaled.clamp_(max=self.max_adv_weight)

                    # Update policy with collected trajectories
                    loss = self.model.loss(
                        actions_b,
                        obs_b,
                        advantages_b_scaled.detach(),
                    )
                    self.actor_optimizer.zero_grad()
                    loss.backward()
                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor.parameters(), self.max_grad_norm
                            )
                        self.actor_optimizer.step()

            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                }
            )
            if self.itr % self.log_freq == 0:
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: loss {loss:8.4f} | reward {avg_episode_reward:8.4f} |t:{timer():8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "loss": loss,
                                "loss - critic": loss_critic,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["loss"] = loss
                    run_results[-1]["loss_critic"] = loss_critic
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                run_results[-1]["time"] = timer()
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1