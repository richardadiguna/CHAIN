# View CHAIN experiment results in Weights & Biases

Run these commands from the repository root with your virtual environment active. If the experiment has already finished, log in and skip to step 3.

## 1. Log in

```bash
wandb login
```

## 2. Run an experiment

This MinAtar example runs for 20,000 environment steps, evaluates every 5,000 steps, enables churn measurements, and saves a results file under `results/`. [MinAtar CHAIN runner](minatar/run_double_dqn_chain.py)

```bash
python minatar/run_double_dqn_chain.py \
  --env breakout --max-step 20 --eval-interval 5000 \
  --invest --is-save-data
```

For the PPO scripts, add `--track` to record TensorBoard metrics in a W&B offline run. [MuJoCo PPO + CHAIN](mujoco/run_ppo_chain_auto_mujoco.py) · [DeepMind Control PPO + CHAIN](dmc/run_ppo_chain_auto_dmc.py)

## 3. Sync the offline run

List local runs and replace the example directory with the one you want to upload.

```bash
ls -dt wandb/offline-run-*
wandb sync ./wandb/offline-run-YYYYMMDD_HHMMSS-RUN_ID
```

## 4. View the curves

Open the synced run in its W&B project workspace. Look for these metrics.

| Experiment | W&B project | Evaluation return | Policy churn |
| --- | --- | --- | --- |
| MinAtar Double DQN + CHAIN | `policy_churn_minatar` | `eval_return` | `invest/greedy_a_diff` |
| MuJoCo PPO + CHAIN | `policy_churn_mujcoo_ppo` | `charts/Eval` | `charts/policy_churn` |
| DeepMind Control PPO + CHAIN | `ppo_improv_dmc` | `charts/Eval` | `charts/policy_churn` |

## 5. Upload the results file as an Artifact

Use this optional step if you want a versioned, downloadable copy of the MinAtar `.npz` file. Find the file and substitute its path in the upload command.

```bash
find results -name '*.npz'
wandb artifact put \
  --name policy_churn_minatar/breakout-chain-results \
  --type evaluation \
  ./results/path-to-your-results.npz
```

Open the project's **Artifacts** tab to find the uploaded file and its versions. The MinAtar CHAIN runner saves the latest evaluation return and horizon in the `.npz`. Open the synced run to see the full evaluation curve. [Results export](minatar/run_double_dqn_chain.py)
