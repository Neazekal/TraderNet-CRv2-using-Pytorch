#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from database.datasets.utils import prepare_eval_dataset
from environments.factory import build_trading_environment


def build_eval_env(
        dataset_filepath,
        timeframe_size,
        target_horizon_len,
        num_eval_samples,
        fees,
        reward_fn_instance,
        position_size=1.0,
        leverage=1.0,
        n_consecutive_window=None,
        reward_wrapper=None,
        **kwargs
):
    x_eval, eval_reward_fn = prepare_eval_dataset(
        dataset_path=config.dataset_save_filepath.format(dataset_filepath),
        feature_columns=config.regression_features,
        timeframe_size=timeframe_size,
        target_horizon_len=target_horizon_len,
        num_eval_samples=num_eval_samples,
        fees=fees,
        reward_fn_factory=reward_fn_instance,
        position_size=position_size,
        leverage=leverage,
        reward_wrapper=reward_wrapper
    )

    def make_env():
        env = build_trading_environment(
            states=x_eval,
            reward_fn=eval_reward_fn,
            episode_steps=x_eval.shape[0] - 1,
            n_consecutive_window=n_consecutive_window
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    return env


def load_agent(agent_class, checkpoint_filepath, env):
    model_path = os.path.join(checkpoint_filepath, "model")
    if not os.path.exists(model_path + ".zip"):
        print(f"Warning: Model not found at {model_path}.zip")
        return None

    try:
        agent = agent_class.load(model_path, env=env)
        return agent
    except Exception as e:
        print(f"Error loading agent from {model_path}: {e}")
        return None


def eval_tradernet(agent, env):
    obs = env.reset()
    cumulative_rewards = 0.0
    cumulative_pnls = 0.0
    pnls = []

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        reward_val = float(reward[0])
        log_pnl = float(info[0]['log_pnl'])

        cumulative_rewards += reward_val
        cumulative_pnls += log_pnl
        pnls.append(cumulative_pnls)

        if done[0]:
            break

    return cumulative_rewards, pnls


if __name__ == "__main__":
    for agent_name, agent_params in config.agent_config.items():
        for dataset_name, dataset_filepath in config.datasets_dict.items():
            for reward_fn_name, reward_fn_instance in config.reward_config.items():
                print(f"Evaluating {agent_name} on {dataset_name} with {reward_fn_name}...")

                env = build_eval_env(
                    dataset_filepath=dataset_filepath,
                    reward_fn_instance=reward_fn_instance,
                    **config.env_config
                )

                checkpoint_path = f'database/storage/checkpoints/experiments/tradernet/{agent_name}/{dataset_name}/{reward_fn_name}/'
                agent = load_agent(
                    agent_class=agent_params['agent_class'],
                    checkpoint_filepath=checkpoint_path,
                    env=env
                )

                if agent is None:
                    print(f"Skipping {agent_name} on {dataset_name} due to missing model.")
                    continue

                average_returns, pnls = eval_tradernet(
                    agent=agent.model,
                    env=env
                )

                base_env = env.envs[0].unwrapped
                episode_metrics = base_env.get_metrics()

                metrics = {
                    'average_returns': [average_returns],
                    **{
                        metric.name: [float(np.mean(metric.episode_metrics))] if len(metric.episode_metrics) > 0 else [float(metric.result())]
                        for metric in episode_metrics
                    }
                }
                results_df = pd.DataFrame(metrics)

                output_metrics_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{reward_fn_name}_metrics.csv'
                os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
                results_df.to_csv(output_metrics_path, index=False)

                print(results_df, '\n')

                episode_pnls_df = pd.DataFrame(pnls, columns=['cumulative_pnl'])
                output_pnls_path = f'experiments/tradernet/{agent_name}/{dataset_name}_{reward_fn_name}_eval_cumul_pnls.csv'
                episode_pnls_df.to_csv(output_pnls_path, index=False)

                print(episode_pnls_df.tail(5))
