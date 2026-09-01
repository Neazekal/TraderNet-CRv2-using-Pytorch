# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch PPO Agent wrapper using Stable Baselines 3.

This module provides a PPOAgent class that wraps Stable Baselines 3's PPO
implementation for easy training and evaluation in reinforcement learning tasks.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv



class PPOAgent:
    """PPO Agent wrapper using Stable Baselines 3.
    
    This class provides a simplified interface for training and evaluating
    PPO (Proximal Policy Optimization) agents using the Stable Baselines 3
    library.
    
    Attributes:
        model: The underlying SB3 PPO model.
        env: The training environment.
    """

    def __init__(
            self,
            env: Union[VecEnv, Any],
            policy: Union[str, type] = "MlpPolicy",
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            tensorboard_log: Optional[str] = None,
            verbose: int = 1,
            seed: Optional[int] = None,
            device: str = "auto",
            policy_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        """Initialize the PPO Agent.

        Args:
            env: The environment to train on (Gym or VecEnv).
            policy: The policy model to use ("MlpPolicy", "CnnPolicy", or custom).
            learning_rate: Learning rate for the optimizer.
            n_steps: Number of steps to run per environment per update.
            batch_size: Minibatch size for each gradient update.
            n_epochs: Number of epochs when optimizing the surrogate loss.
            gamma: Discount factor for rewards.
            gae_lambda: Factor for trade-off of bias vs variance in GAE.
            clip_range: Clipping parameter for PPO objective.
            ent_coef: Entropy coefficient for loss calculation.
            vf_coef: Value function coefficient for loss calculation.
            max_grad_norm: Maximum value for gradient clipping.
            tensorboard_log: Directory for TensorBoard logs.
            verbose: Verbosity level (0: none, 1: info, 2: debug).
            seed: Random seed for reproducibility.
            device: Device to use ("cpu", "cuda", or "auto").
            **kwargs: Additional arguments passed to SB3 PPO.
        """
        self.env = env
        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            **kwargs,
        )

    def train(self, total_timesteps: int, **kwargs) -> "PPOAgent":
        """Train the PPO agent.

        Args:
            total_timesteps: Total number of environment steps to train for.
            **kwargs: Additional arguments passed to model.learn().

        Returns:
            self: The trained agent instance.
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def act(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Select an action given an observation.

        Args:
            observation: The current observation from the environment.
            deterministic: If True, use deterministic actions (no exploration).

        Returns:
            A tuple of (action, states). States is None for non-recurrent policies.
        """
        action, states = self.model.predict(
            observation, deterministic=deterministic
        )
        return action, states

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: File path to save the model (without extension).
        """
        self.model.save(path)

    @classmethod
    def load(
            cls,
            path: str,
            env: Optional[Union[VecEnv, Any]] = None,
            **kwargs,
    ) -> "PPOAgent":
        """Load a trained model from disk.

        Args:
            path: File path to load the model from.
            env: Optional environment to set for the loaded model.
            **kwargs: Additional arguments passed to PPO.load().

        Returns:
            A new PPOAgent instance with the loaded model.
        """
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env, **kwargs)
        agent.env = env
        return agent

    def get_env(self) -> Optional[VecEnv]:
        """Get the current environment.

        Returns:
            The current training environment.
        """
        return self.model.get_env()

    def set_env(self, env: Union[VecEnv, Any]) -> None:
        """Set a new environment for the agent.

        Args:
            env: The new environment to use.
        """
        self.env = env
        self.model.set_env(env)
