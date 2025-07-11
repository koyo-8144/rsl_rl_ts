#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl_ts
from rsl_rl_ts.algorithms import PPO
from rsl_rl_ts.env import VecEnv
from rsl_rl_ts.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl_ts.utils import store_code_state

import copy


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cuda:0"):
        # print("OnPolicyRunner")
        # breakpoint()
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # obs, extras = self.env.get_observations()
        # num_obs = obs.shape[1]
        # if "critic" in extras["observations"]:
        #     num_critic_obs = extras["observations"]["critic"].shape[1]
        # else:
        #     num_critic_obs = num_obs
        # if "adaptation" in extras["observations"] and "privilege" in extras["observations"]:
        #     num_obs_history = extras["observations"]["adaptation"].shape[1]
        #     num_privileged_obs = extras["observations"]["privilege"].shape[1]
        #     print("Adaptation module is used")
        #     breakpoint()
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        if self.env.num_domain_randomization is not None:
            num_privileged_obs = self.env.num_domain_randomization 
            # num_domain_randomization = self.env.num_domain_randomization 
        if self.env.num_obs_history is not None:
            num_obs_history = self.env.num_obs_history 
        num_obs = self.env.num_obs    
        # num_critic_obs = self.env.num_obs

        # print("num_obs (actor): ", num_obs)
        # print("num_critic_obs (critic): ", num_critic_obs)
        # print("num_privileged_obs (DR params): ", num_privileged_obs)
        # print("num_obs_history (15*num_obs): ", num_obs_history)
        # breakpoint()

        # actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic_class = eval(self.cfg["policy_class_name"])
        # actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
        #     num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        # ).to(self.device)
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, num_privileged_obs, num_obs_history, self.env.num_actions, **self.policy_cfg,
        ).to(self.device)
        # actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
        #     num_obs, num_critic_obs, num_domain_randomization, num_obs_history, self.env.num_actions, **self.policy_cfg,
        # ).to(self.device)
        # alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        # self.empirical_normalization = self.cfg["empirical_normalization"]
        # if self.empirical_normalization:
        #     print("Normalisation")
        #     breakpoint()
        #     self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
        #     self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        #     self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(self.device)
        #     self.obs_history_normalizer = EmpiricalNormalization(shape=[num_obs_history], until=1.0e8).to(self.device)
        # else:
        self.obs_normalizer = torch.nn.Identity()  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [num_privileged_obs],
            [num_obs_history],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl_ts.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl_ts.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl_ts.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # obs, extras = self.env.get_observations()
        # critic_obs = extras["observations"].get("critic", obs)
        # privileged_obs = extras["observations"].get("privilege", obs)
        # obs_history = extras["observations"].get("adaptation", obs)
        obs = self.env.get_observations()
        critic_obs = self.env.get_critic_observations()
        privileged_obs = self.env.get_domain_randomizations()
        obs_history = self.env.get_observations_history()
        #obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs, critic_obs, privileged_obs, obs_history = obs.to(self.device), critic_obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # print("obs ", obs.shape)
                    # print("privileged_obs ", privileged_obs.shape)
                    # print("obs_history ", obs_history.shape)
                    # breakpoint()
                    actions = self.alg.act(obs, critic_obs, privileged_obs, obs_history)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    # if "critic" in infos["observations"]:
                    #     critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    # else:
                    #     critic_obs = obs
                    # privileged_obs = infos["observations"]["privilege"]
                    # obs_history = infos["observations"]["adaptation"] 
                    critic_obs = self.env.get_critic_observations()
                    privileged_obs = self.env.get_domain_randomizations()
                    obs_history = self.env.get_observations_history()

                    # print("obs: ", obs.shape)
                    # print("critic_obs: ", critic_obs.shape)
                    # print("privileged_obs: ", privileged_obs.shape)
                    # print("obs_history: ", obs_history.shape)
                    # breakpoint()

                    # obs, critic_obs, rewards, dones = (
                    #     obs.to(self.device),
                    #     critic_obs.to(self.device),
                    #     rewards.to(self.device),
                    #     dones.to(self.device),
                    # )
                    obs, critic_obs, privileged_obs, obs_history, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device), 
                        privileged_obs.to(self.device), 
                        obs_history.to(self.device), 
                        rewards.to(self.device), 
                        dones.to(self.device)
                    )
                    # obs, critic_obs, privileged_obs, obs_history, rewards, dones = (
                    #     obs.to(self.device), 
                    #     critic_obs.to(self.device),
                    #     privileged_obs.to(self.device), 
                    #     obs_history.to(self.device), 
                    #     rewards.to(self.device), 
                    #     dones.to(self.device)
                    # )

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, privileged_obs)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                # self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                # self.save_actor_body(os.path.join(self.log_dir, f"body_{it}.pt")) 
                # self.save_adaptation_module(os.path.join(self.log_dir, f"adaptation_{it}.pt"))

                # torch.save(self.alg.actor_critic.actor.state_dict(), os.path.join(self.log_dir, f"body_{it}.pt"))
                # torch.save(self.alg.actor_critic.adaptation_module.state_dict(), os.path.join(self.log_dir, f"adaptation_{it}.pt"))
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                self.save(os.path.join(self.log_dir, f"adaptation_module_{it}.pt"))
                #torch.save(self.alg.actor_critic.actor.state_dict(), os.path.join(self.log_dir, f"actor_body_{it}.pt"))
                #torch.save(self.alg.actor_critic.adaptation_module.state_dict(), os.path.join(self.log_dir, f"adaptation_module_{it}.pt"))
            
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        # self.save_actor_body(os.path.join(self.log_dir, f"body_{self.current_learning_iteration}.pt")) 
        # self.save_adaptation_module(os.path.join(self.log_dir, f"adaptation_{self.current_learning_iteration}.pt")) 
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.save(os.path.join(self.log_dir, f"adaptation_module_{self.current_learning_iteration}.pt"))
        # torch.save(self.alg.actor_critic.actor.state_dict(), os.path.join(self.log_dir, f"body_{self.current_learning_iteration}.pt"))
        # torch.save(self.alg.actor_critic.adaptation_module.state_dict(), os.path.join(self.log_dir, f"adaptation_{self.current_learning_iteration}.pt"))
        #self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        # torch.save(self.alg.actor_critic.actor.state_dict(), os.path.join(self.log_dir, f"actor_body_{it}.pt"))
        # torch.save(self.alg.actor_critic.adaptation_module.state_dict(), os.path.join(self.log_dir, f"adaptation_module_{self.current_learning_iteration}.pt"))
        #self.save(os.path.join(self.log_dir, f"adaptation_module_{self.current_learning_iteration}.pt"))


    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        locs["mean_pitch_error_normalized"], locs["mean_accel_norm_normalized"], locs["smoothed_ax_mean"], locs["smoothed_az_mean"], locs["env0_command_x"], locs["env0_command_y"], locs["env0_command_z"], locs["env1_command_x"], locs["env1_command_y"], locs["env1_command_z"], locs["env2_command_x"], locs["env2_command_y"], locs["env2_command_z"] = self.env.get_data()
        # locs["mean_pitch_error_normalized"], locs["mean_roll_error_normalized"], locs["mean_accel_norm_normalized"], locs["smoothed_ax_mean"], locs["smoothed_az_mean"], locs["env0_command_x"], locs["env0_command_y"], locs["env0_command_z"], locs["env1_command_x"], locs["env1_command_y"], locs["env1_command_z"], locs["env2_command_x"], locs["env2_command_y"], locs["env2_command_z"] = self.env.get_data()

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/adaptation', locs['mean_adaptation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # self.writer.add_scalar('KL Divergence/mean_kl', locs['mean_kl'], locs['it'])
        # self.writer.add_scalar('Clipped surrogate objective/mean_advantages_batch', locs['mean_advantages_batch'], locs['it'])
        
        self.writer.add_scalar('Slosh Free Reward/mean_pitch_error_normalized', locs["mean_pitch_error_normalized"], locs['it'])
        self.writer.add_scalar('Slosh Free Reward/mean_accel_norm_normalized', locs["mean_accel_norm_normalized"], locs['it'])

        self.writer.add_scalar('Track ACC_X Reward/smoothed_ax_mean', locs["smoothed_ax_mean"], locs['it'])
        self.writer.add_scalar('Track ACC_X Reward/smoothed_az_mean', locs["smoothed_az_mean"], locs['it'])
        # self.writer.add_scalar('Track ACC_X Reward/smoothed_desired_ax_mean', locs["smoothed_desired_ax_mean"], locs['it'])
        
        self.writer.add_scalar('Velocity Resampling/env0_command_x', locs["env0_command_x"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env0_command_y', locs["env0_command_y"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env0_command_z', locs["env0_command_z"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env1_command_x', locs["env1_command_x"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env1_command_y', locs["env1_command_y"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env1_command_z', locs["env1_command_z"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env2_command_x', locs["env2_command_x"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env2_command_y', locs["env2_command_y"], locs['it'])
        self.writer.add_scalar('Velocity Resampling/env2_command_z', locs["env2_command_z"], locs['it'])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Adaptation loss:':>{pad}} {locs['mean_adaptation_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # if self.empirical_normalization:
        #     saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        #     saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        # print(saved_dict)
        # breakpoint()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def save_actor_body(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.actor.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def save_adaptation_module(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.adaptation_module.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        #self.alg.actor_critic.actor.load_state_dict(loaded_dict["actor_body"])
        # if self.empirical_normalization:
        #     self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        #     self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        #return loaded_dict["infos"]
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        # if self.cfg["empirical_normalization"]:
        #     if device is not None:
        #         self.obs_normalizer.to(device)
        #     policy = lambda x, x_h: self.alg.actor_critic.act_inference(self.obs_normalizer(x), self.obs_history_normalizer(x_h))  # noqa: E731
        #     #policy = self.alg.actor_critic.act_inference

        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        # if self.empirical_normalization:
        #     self.obs_normalizer.train()
        #     self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        # if self.empirical_normalization:
        #     self.obs_normalizer.eval()
        #     self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
