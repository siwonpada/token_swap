import gymnasium as gym
import optuna
import os
from gymnasium.envs.registration import WrapperSpec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Any, Dict


from src.callbacks.TrialEval_callback import TrialEvalCallback


EX_NAME = "candidateDistance_experiment_N4"
ENV_ID = "TokenSwapEnv"
gym.register(
    id=ENV_ID,
    entry_point="src.envs:TokenSwapEnv",
    additional_wrappers=(
        WrapperSpec(
            "candidate_distance",
            "src.wrappers:CandidateDistanceWrapper",
            {
                "candidate_num": 6,  # N candidates
            },
        ),
    ),
)

N_TRIALS = 100
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 10
N_EVAL_EPISODES = 5
N_TIMESTEPS = int(1e6)  # 1M
N_WORKERS = 16  # for parallel training
EVAL_FREQ = int(N_TIMESTEPS / (N_EVALUATIONS * N_WORKERS))

DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "policy": "MlpPolicy",
}


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    gamma: float = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm: float = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda: float = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.1, log=True)
    n_steps: int = 2 ** trial.suggest_int("exponent_n_steps", 3, 7)
    learning_rate: float = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef: float = trial.suggest_float("ent_coef", 0.0000001, 0.1, log=True)
    vf_coef: float = trial.suggest_float("vf_coef", 0.000001, 1.0, log=True)

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
    }


def objective(trial: optuna.Trial) -> float:
    def make_env():
        return Monitor(gym.make(ENV_ID, node_num=4, max_episode_steps=1000))

    env = SubprocVecEnv([make_env for _ in range(N_WORKERS)])
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(
        **kwargs,
        env=env,
        tensorboard_log=f"./result/{EX_NAME}/",
        verbose=1,
        device="cuda",
    )
    # Create env used for evaluation.
    eval_env = make_env()
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )
    nan_encountered = False
    try:
        print("start learning")
        model.learn(
            N_TIMESTEPS,
            callback=[
                eval_callback,
            ],
        )
        model.save(f"./result/{EX_NAME}/saves/rl_model_{trial.number}")
        print("Learning end")
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        if model.env is not None:
            model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    result_dir = f"./result/{EX_NAME}"
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Set up the study.
    print("Setting up the study...")
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    if not os.path.exists(f"./{result_dir}/study.db"):
        study = optuna.create_study(
            study_name=EX_NAME,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            storage=f"sqlite:///{result_dir}/study.db",
        )
    else:
        print("Loading existing study...")
        study = optuna.load_study(
            study_name=EX_NAME,
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{result_dir}/study.db",
        )

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=60 * 60 * 24)  # 1 day
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
