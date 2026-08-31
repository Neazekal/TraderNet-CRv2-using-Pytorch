#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import config
from environments.actions import Action
from eval import build_eval_env, load_agent


def eval_tradernet_smurf(tradernet_agent, smurf_agent, env):
    obs = env.reset()
    cumulative_rewards = 0.0
    step_records = []

    while True:
        # 1. Ask Smurf Agent ("Should I HOLD?")
        smurf_action, _ = smurf_agent.predict(obs, deterministic=True)
        smurf_action_val = int(smurf_action[0])

        # 2. Decide Action
        if smurf_action_val == Action.HOLD.value:
            final_action = smurf_action
        else:
            # If Smurf says "Not HOLD", ask TraderNet for the specific trade action
            tradernet_action, _ = tradernet_agent.predict(obs, deterministic=True)
            final_action = tradernet_action

        # 3. Step Environment
        obs, reward, done, info = env.step(final_action)

        reward_val = float(reward[0])
        cumulative_rewards += reward_val
        step_records.append(info[0])

        if done[0]:
            break

    return cumulative_rewards, step_records


if __name__ == "__main__":
    scenario_name = "Portfolio-Simulator"

    agent_pairs = [
        {
            'name': 'PPO_TraderNet_DDQN_Smurf',
            'tradernet': {'class': config.agent_config['PPO']['agent_class'], 'name': 'PPO'},
            'smurf': {'class': config.agent_config['DDQN']['agent_class'], 'name': 'DDQN'}
        },
    ]

    for pair in agent_pairs:
        tradernet_conf = pair['tradernet']
        smurf_conf = pair['smurf']

        for dataset_name, dataset_filepath in config.datasets_dict.items():
            print(f"Evaluating Integrated {pair['name']} on {dataset_name} with {scenario_name}...")

            env = build_eval_env(
                dataset_filepath=dataset_filepath,
                **config.env_config
            )

            tradernet_path = (
                f'database/storage/checkpoints/experiments/tradernet/{tradernet_conf["name"]}/{dataset_name}/{scenario_name}/'
            )
            tradernet_agent = load_agent(tradernet_conf['class'], tradernet_path, env)

            smurf_path = (
                f'database/storage/checkpoints/experiments/smurf/{smurf_conf["name"]}/{dataset_name}/{scenario_name}/'
            )
            smurf_agent = load_agent(smurf_conf['class'], smurf_path, env)

            if tradernet_agent is None or smurf_agent is None:
                print(f"Skipping {pair['name']} due to missing/incompatible models.")
                if tradernet_agent is None:
                    print(f"Missing/incompatible TraderNet model at {tradernet_path}")
                if smurf_agent is None:
                    print(f"Missing/incompatible Smurf model at {smurf_path}")
                continue

            average_returns, step_records = eval_tradernet_smurf(
                tradernet_agent=tradernet_agent.model,
                smurf_agent=smurf_agent.model,
                env=env
            )

            base_env = env.envs[0].unwrapped
            episode_metrics = base_env.get_metrics()

            final_equity = step_records[-1]['equity'] if step_records else config.env_config['initial_equity']
            cumulative_return = step_records[-1]['cumulative_return'] if step_records else 0.0

            metrics = {
                'average_returns': [average_returns],
                'final_equity': [final_equity],
                'cumulative_return': [cumulative_return],
                **{
                    metric.name: [float(np.mean(metric.episode_metrics))] if len(metric.episode_metrics) > 0 else [float(metric.result())]
                    for metric in episode_metrics
                }
            }
            results_df = pd.DataFrame(metrics)

            output_dir = f'experiments/integrated/{tradernet_conf["name"]}'
            output_metrics_path = f'{output_dir}/{dataset_name}_{scenario_name}_metrics.csv'
            os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
            results_df.to_csv(output_metrics_path, index=False)

            print(results_df, '\n')

            episode_pnls_df = pd.DataFrame(step_records)
            output_pnls_path = f'{output_dir}/{dataset_name}_{scenario_name}_eval_cumul_pnls.csv'
            episode_pnls_df.to_csv(output_pnls_path, index=False)

            print(episode_pnls_df[['step_index', 'position', 'equity', 'cumulative_pnl', 'cumulative_return']].tail(5))
