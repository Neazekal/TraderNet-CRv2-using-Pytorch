#!/usr/bin/env python
# coding: utf-8

from environments.rewards.smurf import SmurfRewardFunction
from train import run_experiments

if __name__ == "__main__":
    run_experiments('smurf', reward_wrapper=SmurfRewardFunction)
