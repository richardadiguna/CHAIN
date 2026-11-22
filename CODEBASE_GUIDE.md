# How the CHAIN repository works

This repository runs reinforcement learning experiments. Each experiment trains an agent to act in a game or simulator, then measures how well it performs. It includes standard learning algorithms and versions that add **CHAIN**, a penalty that discourages the agent’s predictions or actions from changing too much on reference inputs.

## The four experiment groups

Yes, four is a useful way to group the code. The groups below follow **what is being tested**. PPO spans two task suites, and the `mujoco/` directory contains both PPO and the separate TD3/SAC experiments. Each group contains several runs across tasks, algorithms, or CHAIN variants.

| Group | Where the agent acts | Where the training data comes from | What the comparison tests |
| --- | --- | --- | --- |
| **1. MinAtar value learning** | Arcade-style games such as Breakout, with a short list of actions | The agent plays and adds steps to a replay buffer | Double DQN versus Double DQN with a penalty on changes in action scores. Plain DQN and random-action runners are also available. |
| **2. Continuous control with PPO** | Gymnasium MuJoCo tasks such as Hopper and DeepMind Control tasks such as ball-in-cup | The current policy collects a fresh rollout for each round of updates | PPO versus PPO with a penalty on changes in the mean action. |
| **3. Continuous control with TD3 and SAC** | Gymnasium MuJoCo tasks such as HalfCheetah | The agent collects steps continually and can reuse older steps from replay | Whether stabilizing the actor, the critics, or both changes learning and churn. |
| **4. Offline RL with CORL and D4RL** | D4RL tasks such as AntMaze and Pen | A fixed file of previously recorded experience. The simulator is used to evaluate the trained policy. | Whether policy or critic stabilization helps AWAC and IQL when collecting more training data is unavailable. |

### Why use DQN for MinAtar and PPO for continuous control?

The distinction is **which actions the agent can choose**. DQN already outputs continuous numbers. Those numbers are scores for a finite set of choices, such as moving left, moving right, or doing nothing. The network can output every score in one pass, and the agent picks the largest.

In a continuous task, an action might be a vector of joint controls such as `[0.12, -0.43, 0.87]`. A critic can still assign a score to that vector. The difficult part is searching for the highest-scoring vector among infinitely many possibilities. You can discretize each control, but even 10 choices per joint lead to one million combinations for six joints. Coarse bins can also remove useful actions. [OpenAI Spinning Up on continuous-action Q-learning](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

PPO takes a direct route. Its **actor** produces a distribution over continuous actions, and the agent samples an action from it. The update increases the probability of actions that turned out better than expected and limits how far the policy update can go through a clipped objective. PPO also works with discrete actions when its actor outputs a distribution over choices. Here it provides a continuous-control baseline for studying churn in the policy itself. [DMC PPO runner](dmc/run_ppo_dmc.py) · [MuJoCo PPO runner](mujoco/run_ppo_mujoco.py)

### What is DeepMind Control, and how does it relate to MuJoCo?

**MuJoCo is the physics engine.** It simulates bodies, joints, actuators, and contacts. **DeepMind Control Suite**, or DMC, is a collection of tasks built on that engine. A task defines the goal, the observations available to the agent, and the reward. Gymnasium’s MuJoCo environments are another collection of tasks using the same engine. For example, Hopper asks a simulated robot to move forward, while DMC ball-in-cup asks the agent to catch a ball in a cup. The two PPO runner pairs let the repository test the same learning approach on both collections. [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) · [Gymnasium MuJoCo environments](https://gymnasium.farama.org/v0.29.0/environments/mujoco/)

### Why are TD3 and SAC in the MuJoCo folder too?

The same simulator can train different algorithms. **PPO** is *on-policy*: it learns from a rollout collected by the current policy, performs several passes over that rollout, and collects a fresh one. **TD3** and **SAC** are *off-policy*: they repeatedly sample a buffer containing steps taken by earlier versions of the policy. All three are *online* here because the agents keep interacting with the simulator to collect training data.

- **TD3** (Twin Delayed DDPG) learns an actor that outputs one action for a given observation. Two critics score actions, and the actor updates less often than the critics. The CHAIN policy penalty compares the current and historical actor outputs. [TD3](mujoco/algs/td3.py) · [TD3 policy penalty](mujoco/algs/td3_pcr.py)
- **SAC** (Soft Actor-Critic) learns an actor that outputs an action distribution. Its objective rewards both return and entropy, which favors keeping a range of plausible actions. The CHAIN policy penalty compares distributions. [SAC](mujoco/algs/sac.py) · [SAC policy penalty](mujoco/algs/sac_pcr.py)

Both algorithms have a distinct actor and a pair of critics. This makes it possible to add a CHAIN penalty to just the actor (`PCR`), just the critics (`VCR`), or both (`DCR`) and study each choice. Their runners share the same basic structure. Their update files differ because each algorithm computes its targets, actions, and losses differently. [MuJoCo variant runner](mujoco/run_deepac_chain_mujoco.py)

### What is CORL?

**CORL is an upstream library of offline RL implementations. D4RL supplies the benchmark datasets and evaluation tasks.** The local `corl/` directory contains adapted CORL scripts for AWAC and IQL. During training, those scripts sample recorded observations, actions, and rewards from D4RL. They periodically run the learned policy in a simulator to measure its return. An online replay buffer grows as the agent explores. This offline dataset stays fixed even if the learned agent would visit different states. [CORL](https://github.com/tinkoff-ai/CORL) · [D4RL](https://github.com/tinkoff-ai/d4rl) · [Local AWAC runner](corl/algorithms/offline/awac.py)

The common question across the four groups is whether keeping predictions closer to their recent history helps learning. The implementations repeat that idea where it fits each algorithm. MinAtar matches Q scores, PPO matches mean actions on fresh rollout observations, TD3 and SAC can regularize the actor and the critics separately, and the offline versions draw all reference inputs from the fixed dataset.

## Where to start

| I want to understand… | Open this file first |
| --- | --- |
| How the code plays a game and trains an agent | [minatar/run_double_dqn.py](minatar/run_double_dqn.py) |
| Where CHAIN changes an actual loss | [minatar/agents/double_dqn_chain.py](minatar/agents/double_dqn_chain.py), method `train()` |
| How PPO collects experience and learns from it | [dmc/run_ppo_dmc.py](dmc/run_ppo_dmc.py) |
| How a MuJoCo runner chooses between TD3 and SAC | [mujoco/run_deepac_baseline_mujoco.py](mujoco/run_deepac_baseline_mujoco.py) |
| How training works with a pre-recorded dataset | [corl/algorithms/offline/awac.py](corl/algorithms/offline/awac.py), function `train(config)` |

## One training update, from the game to the loss

Follow [the fixed CHAIN Double DQN runner](minatar/run_double_dqn_chain.py) into [its agent](minatar/agents/double_dqn_chain.py). This pair gives a concrete example of what most of the other experiments are doing.

1. **Play the game.** The runner chooses an action, calls the game environment, and receives a reward and a new board. It saves the old board, action, reward, new board, and terminal flag in a replay buffer. A replay buffer is a collection of past experiences that the agent can sample later.
2. **Learn from past experiences.** The runner calls `agent.train()`. The agent samples a batch from replay. Its Q network gives one score per possible action. The ordinary Double DQN loss tells the network to bring its score for the action taken closer to a target computed from the reward and a prediction about the next board.
3. **Check what else the update could change.** The CHAIN agent draws a separate batch of boards from the same buffer. It runs those boards through both the current network and a saved older copy. The squared difference between their action scores is the CHAIN penalty. The older copy supplies fixed reference scores.
4. **Update the live network.** The optimizer uses the ordinary loss plus the weighted CHAIN penalty. The code saves a copy of the updated network for future comparisons. Separately, the runner occasionally measures return by playing evaluation games.

Imagine two actions score **1.01** and **1.00** on a particular board. A small change in either score could change the action the agent chooses. That kind of change between updates is **churn**. In Double DQN, a different greedy action on the next board also changes which value is looked up for a later training target. CHAIN targets this feedback by penalizing changes in the scores themselves. A score change can be measured even when it leaves the chosen action unchanged.

You’ll see two different saved networks in the Double DQN agent. The **target network** supplies a value for step 2. The **historical copy** supplies the scores for step 3.

## MinAtar files

MinAtar is the discrete-action setup. A runner controls the experiment and its matching agent class controls learning.

| File | Responsibility |
| --- | --- |
| [run_random_agent.py](minatar/run_random_agent.py) | Plays MinAtar with randomly selected actions and reports performance. |
| [run_dqn.py](minatar/run_dqn.py) | Runs the DQN experiment. Handles arguments, game interaction, evaluation, and logging. |
| [run_double_dqn.py](minatar/run_double_dqn.py) | Runs the standard Double DQN experiment. |
| [run_double_dqn_chain.py](minatar/run_double_dqn_chain.py) | Runs Double DQN with a user-supplied CHAIN weight. |
| [run_double_dqn_chain_auto.py](minatar/run_double_dqn_chain_auto.py) | Runs Double DQN with automatic CHAIN weighting. |
| [agents/dqn.py](minatar/agents/dqn.py) | Defines the convolutional Q network and DQN update. The target network supplies the maximum next-action score. |
| [agents/double_dqn.py](minatar/agents/double_dqn.py) | Implements Double DQN. The live network chooses the next action and the target network scores it. It also supports measuring changes in Q scores and greedy actions. |
| [agents/double_dqn_chain.py](minatar/agents/double_dqn_chain.py) | Adds a fixed-weight penalty for changes in the full vector of action scores. |
| [agents/double_dqn_chain_auto.py](minatar/agents/double_dqn_chain_auto.py) | Adds the same penalty and adjusts its weight using recent losses. |
| [agents/agent_utils.py](minatar/agents/agent_utils.py) | Defines the experience record and replay buffers. The supplied agents use uniform replay. |
| [agents/sum_tree.py](minatar/agents/sum_tree.py) | Provides the tree used by the optional prioritized buffer to sample experiences by weight. |

## DeepMind Control and MuJoCo PPO files

PPO collects a fresh batch of simulator steps, called a **rollout**, and does several passes over it. The runner stores what the agent saw, what it did, how probable those actions were, and what rewards it received. It then estimates which actions performed better than expected and uses those estimates to update the policy. The value network learns to predict future return. The CHAIN version adds a penalty for changes in the actor’s mean action on reference observations drawn from the rollout.

| File | Responsibility |
| --- | --- |
| [dmc/run_ppo_dmc.py](dmc/run_ppo_dmc.py) | Runs baseline PPO on DeepMind Control. Contains rollout collection, advantage estimation, the PPO update, evaluation, and logging. |
| [dmc/run_ppo_chain_auto_dmc.py](dmc/run_ppo_chain_auto_dmc.py) | Adds historical actor-mean matching and automatic weighting to the DMC PPO loop. |
| [dmc/utils/basic_utils.py](dmc/utils/basic_utils.py) | Builds and normalizes DMC environments, defines the PPO networks, and evaluates them with observation statistics copied from training. |
| [mujoco/run_ppo_mujoco.py](mujoco/run_ppo_mujoco.py) | Runs baseline PPO on MuJoCo. Includes model-width settings and optional environment normalization. |
| [mujoco/run_ppo_chain_auto_mujoco.py](mujoco/run_ppo_chain_auto_mujoco.py) | Adds historical actor-mean matching and automatic weighting to the MuJoCo PPO loop. |
| [mujoco/utils/ppo_basic_utils.py](mujoco/utils/ppo_basic_utils.py) | Builds MuJoCo environments for PPO, defines the actor and value networks, and runs evaluation. The supplied PPO runners construct `DSAgent` from this file. |

Both PPO baselines also **measure** policy churn. The CHAIN files additionally use churn in the optimization loss. PPO’s usual likelihood ratio compares against the policy that collected the rollout. The CHAIN penalty compares with a saved copy from the optimizer’s recent history.

## MuJoCo TD3 and SAC files

These agents learn from a replay buffer. The runner collects experience, calls the selected algorithm’s `train()` method, and evaluates the policy. The algorithm files define the networks and what the optimizer updates. An **actor** chooses actions. A **critic** estimates the future return of taking an action in a state. TD3’s actor outputs a single action and is updated less frequently than its critics. SAC’s actor outputs a distribution and its objective rewards both return and randomness.

### Runners and their settings

| File | Responsibility |
| --- | --- |
| [run_deepac_baseline_mujoco.py](mujoco/run_deepac_baseline_mujoco.py) | Runs baseline TD3 or SAC. |
| [configs/deepac_baseline_configs.py](mujoco/configs/deepac_baseline_configs.py) | Defines the baseline runner’s command-line options and defaults. |
| [run_deepac_chain_mujoco.py](mujoco/run_deepac_chain_mujoco.py) | Runs policy, value, or combined CHAIN variants of TD3 and SAC. |
| [configs/deepac_chain_configs.py](mujoco/configs/deepac_chain_configs.py) | Defines fixed-variant settings, including penalty coefficients and reference history. |
| [run_deepac_chain_auto_mujoco.py](mujoco/run_deepac_chain_auto_mujoco.py) | Runs TD3 with automatic policy churn weighting. |
| [configs/deepac_chain_auto_configs.py](mujoco/configs/deepac_chain_auto_configs.py) | Defines the automatic runner’s defaults, including the target loss ratio. |
| [run_random_policy.py](mujoco/run_random_policy.py) | Evaluates random actions to establish a performance reference. |
| [utils/basic_utils.py](mujoco/utils/basic_utils.py) | Builds environments for the replay-based and random-policy runners. PPO has its own helper. |

### Learning algorithms

`PCR` applies the penalty to the actor, `VCR` applies it to the critics, and `DCR` applies both. The actor penalty compares outputs on the same observations. The critic penalty compares Q scores for the same state-action pairs.

| File | Responsibility |
| --- | --- |
| [algs/td3.py](mujoco/algs/td3.py) | Implements baseline TD3 with two critics, smoothed target actions, and delayed actor updates. |
| [algs/td3_pcr.py](mujoco/algs/td3_pcr.py) | Penalizes squared differences between current and historical TD3 actions. |
| [algs/td3_vcr.py](mujoco/algs/td3_vcr.py) | Penalizes squared differences between current and historical TD3 critic scores. |
| [algs/td3_dcr.py](mujoco/algs/td3_dcr.py) | Applies both TD3 penalties in the same training loop. |
| [algs/td3_pcr_auto.py](mujoco/algs/td3_pcr_auto.py) | Applies the TD3 action penalty and automatically adjusts its weight. |
| [algs/sac.py](mujoco/algs/sac.py) | Implements baseline SAC with two critics, a stochastic actor, and optional automatic tuning of the entropy coefficient. |
| [algs/sac_pcr.py](mujoco/algs/sac_pcr.py) | Uses KL divergence to compare current and historical SAC action distributions before squashing them to the action range. |
| [algs/sac_vcr.py](mujoco/algs/sac_vcr.py) | Penalizes squared differences between current and historical SAC critic scores. |
| [algs/sac_dcr.py](mujoco/algs/sac_dcr.py) | Applies both SAC penalties in the same training loop. |

## CORL offline learning files

These scripts train on data collected in advance. Each Python file contains the entire experiment, including its configuration, dataset loading, networks, trainer, and evaluation. The environment is used to evaluate the policy. Training batches come from the fixed D4RL dataset.

| File | Responsibility |
| --- | --- |
| [awac.py](corl/algorithms/offline/awac.py) | Implements AWAC. The actor learns to imitate recorded actions, giving more weight to actions that the critics score higher than actions from the current policy. |
| [awac_pcr.py](corl/algorithms/offline/awac_pcr.py) | Adds a penalty when the AWAC actor’s mean action changes on dataset observations. |
| [awac_vcr.py](corl/algorithms/offline/awac_vcr.py) | Adds a penalty when AWAC critic scores change on recorded state-action pairs. |
| [iql.py](corl/algorithms/offline/iql.py) | Implements IQL. It learns a state-value model from the scores of recorded actions, uses it in the critic target, and gives better-scoring recorded actions more weight in the actor update. |
| [iql_pcr.py](corl/algorithms/offline/iql_pcr.py) | Adds a penalty when the IQL actor’s mean action changes on dataset observations. |
| [iql_vcr.py](corl/algorithms/offline/iql_vcr.py) | Adds a penalty to the two IQL critics. IQL’s separate state-value network still supplies the critic’s next-state target. |

The YAML files set task-specific defaults such as dataset ID, training length, evaluation frequency, and algorithm hyperparameters. Each row below identifies the task for both presets. The Python script and its `alg` setting determine whether it applies the CHAIN loss.

| Dataset | AWAC preset | IQL preset |
| --- | --- | --- |
| AntMaze U-maze | [AWAC](corl/configs/offline/awac/antmaze/umaze_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/umaze_v2.yaml) |
| AntMaze U-maze, diverse | [AWAC](corl/configs/offline/awac/antmaze/umaze_diverse_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/umaze_diverse_v2.yaml) |
| AntMaze medium, play | [AWAC](corl/configs/offline/awac/antmaze/medium_play_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/medium_play_v2.yaml) |
| AntMaze medium, diverse | [AWAC](corl/configs/offline/awac/antmaze/medium_diverse_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/medium_diverse_v2.yaml) |
| AntMaze large, play | [AWAC](corl/configs/offline/awac/antmaze/large_play_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/large_play_v2.yaml) |
| AntMaze large, diverse | [AWAC](corl/configs/offline/awac/antmaze/large_diverse_v2.yaml) | [IQL](corl/configs/offline/iql/antmaze/large_diverse_v2.yaml) |
| Pen, human | [AWAC](corl/configs/offline/awac/pen/human_v1.yaml) | [IQL](corl/configs/offline/iql/pen/human_v1.yaml) |
| Pen, cloned | [AWAC](corl/configs/offline/awac/pen/cloned_v1.yaml) | [IQL](corl/configs/offline/iql/pen/cloned_v1.yaml) |

## Setup files and figures

| File | Responsibility |
| --- | --- |
| [README.md](README.md) | Introduces the paper, upstream projects, and published results. |
| [requirements.txt](requirements.txt) | Lists dependencies for the Python 3.11 MinAtar, MuJoCo, and DMC setup. |
| [Dockerfile.cleanrl_mujoco](dockerfiles/Dockerfile.cleanrl_mujoco) | Reference recipe for the original Python 3.9 CleanRL MuJoCo environment. |
| [Dockerfile.clearnrl_dmc](dockerfiles/Dockerfile.clearnrl_dmc) | Reference recipe for the Python 3.9 CleanRL environment with DeepMind Control dependencies. |
| [Dockerfile.corl_d4rl](dockerfiles/Dockerfile.corl_d4rl) | Reference recipe for the Python 3.9 CORL/D4RL environment with legacy MuJoCo. |
| [chain_ddqn_fig.png](assets/chain_ddqn_fig.png) | Shows the Double DQN performance comparison. |
| [chain_ddqn_value_churn.png](assets/chain_ddqn_value_churn.png) | Shows Double DQN value churn. |
| [chain_ppo_fig.png](assets/chain_ppo_fig.png) | Shows the PPO performance comparison. |
| [chain_policy_reduce_fig.png](assets/chain_policy_reduce_fig.png) | Shows PPO policy churn. |
| [chain_scaling_fig.png](assets/chain_scaling_fig.png) | Plots the model-scaling experiments. |
| [chain_scaling_table.png](assets/chain_scaling_table.png) | Presents the scaling results in a table. |
| [.gitignore](.gitignore) | Lists files and paths Git should ignore. |
| `.DS_Store` | macOS Finder metadata. |

The Dockerfiles expect upstream files in their build contexts. The root requirements file covers local online experiments. For CORL, the README points to the upstream D4RL setup.

## How to read a CHAIN training loop

When you open a variant’s update method, look for these variables and operations in order.

1. **The original loss.** Look for `q_loss`, `actor_loss`, or `pg_loss`. This is what the underlying algorithm is trying to improve.
2. **The reference inputs.** Look for names like `reg_data`, `reg_batch`, or `mb_reg_inds`. These identify the observations or state-action pairs whose predictions are being compared.
3. **The historical model.** Look for `his_...` lists and `reg_his_idx`. The code counts backward from the latest saved copy. Check where the copy is appended to understand how many optimizer updates old it is. PPO saves after minibatch updates, while TD3 updates its actor on a delayed schedule.
4. **The penalty and its weight.** Look for `reg_loss`, `reg_coef`, or `v_reg_coef`. The fixed variants use a configured coefficient, sometimes scaled by a loss or score magnitude. The automatic variants use recent loss magnitudes. For example, if the ordinary loss averages 2, the penalty averages 0.5, and the target ratio is 0.1, the ratio gives a weight of 0.4 before any schedule multiplier. That makes the weighted penalty about 0.2.
5. **The metrics.** Look for `invest` or names containing `churn`. A run can measure change without adding a penalty. Compare metrics with the same meaning and units, such as two MinAtar greedy-action flip rates or two PPO mean-action errors.

Three details can explain confusing behavior. A history buffer can be too short for its requested reference index, leaving the penalty inactive. A very short PPO Auto rollout can hit `reg_loss.item()` while the warm-up loss is still a Python integer. IQL PCR’s mean-action comparison expects the Gaussian actor’s `get_mean()` method, which the deterministic actor lacks. These observations come from reading the source. Simulator installation and training runs were outside this review.
