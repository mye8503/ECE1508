---
title: Final Project

---

# ECE1508

## Project Description
In recent years, with the progress and development of deep neural networks and its integration with reinforcement learning (RL), there has been an ever growing interest in leveraging DRL in the field of finance, mainly for portfolio optimization. In this project, we investigate the performance of deep reinforcement learning (DRL) in the field of portfolio management. The efficacy of four different DRL models is measured by comparing the results obtained via DRL with the results obtained through three simple and non-RL heuristics and models.


## Setup
Download all packages listed in **requirements.txt**.


## Demo

### Input
To view the data we are inputting into our models, run the following in a Python shell:
```
>> import pickle
>> with open(${dataset_filepath}$, 'rb') as f:
>>     data = pickle.load(f)
>> data
```

This will print the Open, Close, High, Low, Volume, and Log return values for each company in the dataset, over the date range. To view specific aspects of the data, you can index into it (ex: `data['Open']`, `data['GOOG']`, `data['2021':'2024']`).

### Dataset
There are two datasets in this GitHub. One (in **data/**) contains MAANG (Microsoft, Amazon, Apple, Nvidia, Google) assets from 2005-2025, and the other (in **new_data/**) contains 33 S&P 500 assets from 2005-2025 (for the full company list, see our paper). The data has already been partitioned into train/val/test folders. If you want to change the split or pull data from different dates, change the dates in **get_data.py** and run the file.


### Environment
We used the same environment over all four models and three baseline heuristics. It can be found in **ppo_env.py**. You can modify the environment as you wish. For example, if you want to change the reward function to differential Sharpe Ratio from simple return, comment out line 97 and uncomment line 100.


### Models
There are four models: PPO, DDPG, A2C, and ACER. They can be found in **PPO.py**, **DDPG.py**, **a2c.py**, and **acer.py** respectively. For PPO and DDPG, you can modify any hyperparameter, as well as the dataset path and save path, before running  the file. For example, each model trains a certain number of agents (seeds) on each training window, then saves the best performing agent as a zip file (A2C and ACER use a .pt file). You can change the number of seeds by modifying the range in `seeds = [i*111 for i in range(1, 3)]`. A2C and ACER's hyperparameters (ex. learning rate, gamma) can be modified by adding an argument in the terminal command.


### Output
To evaluate a trained model on the test sets, run **backtest_ppo.py** with the appropriate `best_model_paths`, `testset_paths`, and `save_results_path`. The Equal Weighting, Buy-and-Hold, and MVO baselines' results can be found by running **Equal_Weighting.py**, **Buy_and_Hold.py**, and **backtest_MVO.py** respectively and changing `testset_paths` and `save_results_path` in the code. These will produce graphs showing the model's annual returns, annual Sharpe Ratio, and monthly returns.

The raw outputs of each model (i.e. action taken at each time step in the dataset, i.e. daily portfolio allocation) can be found in `portfolio_vals.pkl` in the model's `save_results_path`. You can view them in a Python shell by following the same steps from **Input** to load the pickle file.

To compare the results of any model against each other, run **compare_results.py** and change the appropriate values in `data_dict` (line 33). This file plots the annual Sharpe Ratio and Maximum Drawdown for each model. To get the numerical results, run **get_results.py**.