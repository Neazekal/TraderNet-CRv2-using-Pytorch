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

"""PyTorch DQN Agent wrapper using Stable Baselines 3.

This module provides a DQNAgent class that wraps Stable Baselines 3's DQN
implementation for easy training and evaluation in reinforcement learning tasks.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv



class DQNAgent:
    """DQN Agent wrapper using Stable Baselines 3.
    
    This class provides a simplified interface for training and evaluating
    DQN (Deep Q-Network) agents using the Stable Baselines 3 library.
    
    Attributes:
        model: The underlying SB3 DQN model.
        env: The training environment.
    """

    def __init__(
            self,
            env: Union[VecEnv, Any],
            policy: Union[str, type] = "MlpPolicy",
            learning_rate: float = 1e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 50000,
            batch_size: int = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: int = 4,
            gradient_steps: int = 1,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10.0,
            tensorboard_log: Optional[str] = None,
            verbose: int = 1,
            seed: Optional[int] = None,
            device: str = "auto",
            policy_kwargs: Optional[dict] = None,
            **kwargs,
    ):
        """Initialize the DQN Agent.

        Args:
            env: The environment to train on (Gym or VecEnv).
            policy: The policy model to use ("MlpPolicy", "CnnPolicy", or custom).
            learning_rate: Learning rate for the optimizer.
            buffer_size: Size of the replay buffer.
            learning_starts: Number of steps before learning starts.
            batch_size: Minibatch size for each gradient update.
            tau: Soft update coefficient (1 for hard update).
            gamma: Discount factor for rewards.
            train_freq: Update the model every train_freq steps.
            gradient_steps: Number of gradient steps per update.
            target_update_interval: Update target network every N steps.
            exploration_fraction: Fraction of training for epsilon decay.
            exploration_initial_eps: Initial exploration epsilon.
            exploration_final_eps: Final exploration epsilon.
            max_grad_norm: Maximum value for gradient clipping.
            tensorboard_log: Directory for TensorBoard logs.
            verbose: Verbosity level (0: none, 1: info, 2: debug).
            seed: Random seed for reproducibility.
            device: Device to use ("cpu", "cuda", or "auto").
            **kwargs: Additional arguments passed to SB3 DQN.
        """
        self.env = env
        self.model = DQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            **kwargs,
        )

    def train(self, total_timesteps: int, **kwargs) -> "DQNAgent":
        """Train the DQN agent.

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
            deterministic: If True, use greedy action selection (no exploration).

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
    ) -> "DQNAgent":
        """Load a trained model from disk.

        Args:
            path: File path to load the model from.
            env: Optional environment to set for the loaded model.
            **kwargs: Additional arguments passed to DQN.load().

        Returns:
            A new DQNAgent instance with the loaded model.
        """
        agent = cls.__new__(cls)
        agent.model = DQN.load(path, env=env, **kwargs)
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
