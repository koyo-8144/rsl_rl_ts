#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl_ts.modules import ActorCritic
from rsl_rl_ts.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        adaptation_module_learning_rate = 1.e-3,
        num_adaptation_module_substeps = 1,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        # device="cpu",
        device="cuda:0",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.num_adaptation_module_substeps = num_adaptation_module_substeps

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),lr=adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    # def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    #     self.storage = RolloutStorage(
    #         num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
    #     )

    # def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape):
    #     self.storage = RolloutStorage(
    #         num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, self.device
    #         )

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, privileged_obs_shape, obs_history_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, self.device
            )
        
    # def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, privileged_obs_shape, obs_history_shape, action_shape):
    #     self.storage = RolloutStorage(
    #         num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, self.device
    #         )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    #def act(self, obs, critic_obs):
    def act(self, obs, critic_obs, privileged_obs, obs_history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    # def compute_returns(self, last_critic_obs):
    #     last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    #     self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            privileged_obs_batch, 
            obs_history_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            # actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # value_batch = self.actor_critic.evaluate(
            #     critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            # )

             # ────────────────────────────────────────────────
            # 1) SANITIZE all observation inputs
            obs_batch            = torch.nan_to_num(obs_batch,            nan=0.0, posinf=0.0, neginf=0.0)
            privileged_obs_batch = torch.nan_to_num(privileged_obs_batch, nan=0.0, posinf=0.0, neginf=0.0)
            critic_obs_batch     = torch.nan_to_num(critic_obs_batch,     nan=0.0, posinf=0.0, neginf=0.0)

            # ────────────────────────────────────────────────
            # 2) Forward pass through actor‐critic
            self.actor_critic.act(obs_batch, privileged_obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, privileged_obs_batch, masks=masks_batch
            )
            mu_batch    = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ────────────────────────────────────────────────
            # 3) SANITIZE the distribution parameters before any KL / surrogate loss
            dist = self.actor_critic.distribution
            loc   = torch.nan_to_num(dist.mean,   nan=0.0, posinf=0.0, neginf=0.0)
            scale = torch.nan_to_num(dist.stddev, nan=1e-3, posinf=1e-3, neginf=1e-3)
            # rebuild a clean Normal
            self.actor_critic.distribution = Normal(loc, scale)

            # self.actor_critic.act(obs_batch, privileged_obs_batch, masks=masks_batch)
            # actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # value_batch = self.actor_critic.evaluate(critic_obs_batch, privileged_obs_batch, masks=masks_batch)
            # mu_batch = self.actor_critic.action_mean
            # sigma_batch = self.actor_critic.action_std
            # entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        #self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        #self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                        self.learning_rate = max(1e-7, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        #self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        #self.learning_rate = min(1e-3, self.learning_rate * 1.5)
                        self.learning_rate = min(1e-4, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)

                # Check for inf and NaN values in value_losses
                inf_mask = torch.isinf(value_losses)
                #nan_mask = torch.isnan(value_losses)
                
                if inf_mask.any():
                # if inf_mask.any() or nan_mask.any():
                    print("Inf is detected")
                    # print("value_batch ", value_batch) #normal
                    # print("retunrs_batch ", returns_batch) #normal
                    # print("value_losses ", value_losses)
                    # print("mu_batch:", mu_batch)
                    # print("sigma_batch:", sigma_batch)
                    # sigma_batch = torch.clamp(sigma_batch, min=1e-5)
                    # print("sigma_batch after clamp:", sigma_batch)
                    # print("actions_log_prob_batch:", actions_log_prob_batch)
                    # print("KL ", kl)
                    # print("value_losses ", value_losses)
                    # Replace inf and NaN values with large finite values or zero
                    value_losses[inf_mask] = value_losses_clipped[inf_mask]
                    #value_losses[nan_mask] = value_losses_clipped[nan_mask]
                    #print("value_losses_clipped ", value_losses_clipped)
       
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
                #print("value_loss ", value_loss)
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()


            # Adaptation module gradient step
            for epoch in range(self.num_adaptation_module_substeps):
                adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                with torch.no_grad():
                    adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch)
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()

                mean_adaptation_loss += adaptation_loss.item()



        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_loss
