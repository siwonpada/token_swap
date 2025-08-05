import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec
from typing import Any, Dict


EX_NAME = "candidateGraph_experiment"
ENV_ID = "TokenSwapEnv"
gym.register(
    id=ENV_ID,
    entry_point="src.envs:TokenSwapEnv",
    additional_wrappers=(
        WrapperSpec("candidate_graph", "src.wrappers:CandidateGraphWrapper", None),
    ),
)

N_TRIALS = 2000
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 5
N_TIMESTEPS = int(1e7)
N_WORKERS = 16
EVAL_FREQ = int(N_TIMESTEPS / (N_EVALUATIONS * N_WORKERS))
N_EVAL_EPISODES = 5

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "policy_kwargs": dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    ),
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


def make_env():
    return Monitor(gym.make(ENV_ID, start_level=1, max_episode_steps=1000))


def objective(trial: optuna.Trial) -> float:
    env = SubprocVecEnv([make_env for _ in range(N_WORKERS)])
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = MaskablePPO(
        **kwargs, env=env, tensorboard_log=f"./{EX_NAME}/", verbose=1, device="cuda"
    )
    # Create env used for evaluation.
    eval_env = Monitor(gym.make(ENV_ID, start_level=1))
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
                CurriculumCallback(
                    verbose=1, success_threshold=0.9, min_training_epi=2500
                ),
            ],
        )
        model.save(f"./{EX_NAME}/saves/rl_model_{trial.number}")
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
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    if not os.path.exists(EX_NAME):
        os.makedirs(EX_NAME)

    if not os.path.exists(f"./{EX_NAME}/study.db"):
        study = optuna.create_study(
            study_name=EX_NAME,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            storage=f"sqlite:///{EX_NAME}/study.db",
        )
    else:
        print("Loading existing study...")
        study = optuna.load_study(
            study_name=EX_NAME, storage=f"sqlite:///{EX_NAME}/study.db"
        )
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=60 * 60 * 12)  # 12 hours
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
