import gymnasium as gym
import os
from gymnasium.envs.registration import WrapperSpec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from typing import Any, Dict

from src.callbacks.Curriculum_callback import CurriculumCallback

EX_NAME = "curriculum_experiment"
ENV_ID = "TokenSwapEnv-Curriculum"

# 환경 등록
gym.register(
    id=ENV_ID,
    entry_point="src.envs:TokenSwapEnv",
    additional_wrappers=(
        WrapperSpec(
            "candidate_distance",
            "src.wrappers:CandidateDistanceWrapper",
            {
                "candidate_num": 10,  # 초기 후보 수 (작게 시작)
            },
        ),
    ),
)

N_TIMESTEPS = int(2e6)  # 2M timesteps
N_WORKERS = 8  # 병렬 환경 수

# PPO 하이퍼파라미터
PPO_HYPERPARAMS: Dict[str, Any] = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}


def make_env(env_id: str, rank: int, seed: int = 0, node_num: int = 4):
    """환경 생성 함수"""

    def _init():
        env = gym.make(env_id, node_num=node_num, seed=seed + rank)
        env = Monitor(env)
        return env

    return _init


def main():
    """메인 실행 함수"""
    print(f"🚀 커리큘럼 학습 실험 시작: {EX_NAME}")

    # 결과 저장 디렉토리 생성
    save_dir = f"result/{EX_NAME}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/saves", exist_ok=True)

    # 커리큘럼 콜백 설정
    curriculum_callback = CurriculumCallback(
        initial_node_num=4,  # 4개 노드로 시작
        target_node_num=16,  # 16개 노드까지 증가
        success_threshold=0.7,  # 70% 성공률 달성 시 다음 단계로
        min_episodes_before_increase=50,  # 최소 50 에피소드 후 증가 고려
        check_freq=5000,  # 5000 스텝마다 평가
        verbose=2,
    )

    # 체크포인트 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path=f"{save_dir}/saves/", name_prefix="curriculum_model"
    )

    # 콜백 리스트
    callbacks = CallbackList([curriculum_callback, checkpoint_callback])

    # 병렬 환경 생성
    if N_WORKERS > 1:
        env = SubprocVecEnv([make_env(ENV_ID, i, node_num=4) for i in range(N_WORKERS)])
    else:
        env = DummyVecEnv([make_env(ENV_ID, 0, node_num=4)])

    # PPO 모델 생성
    model = PPO(env=env, tensorboard_log=f"{save_dir}/", **PPO_HYPERPARAMS)

    print("📊 초기 커리큘럼 상태:")
    stats = curriculum_callback.get_current_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 학습 시작
    print(f"🎯 학습 시작 - {N_TIMESTEPS:,} timesteps")
    try:
        model.learn(
            total_timesteps=N_TIMESTEPS,
            callback=callbacks,
            tb_log_name="curriculum_PPO",
        )

        # 최종 모델 저장
        model.save(f"{save_dir}/final_model")
        print(f"✅ 학습 완료! 모델이 {save_dir}/final_model.zip에 저장됨")

    except KeyboardInterrupt:
        print("\n⚠️ 학습이 중단되었습니다.")
        model.save(f"{save_dir}/interrupted_model")
        print(f"현재 모델이 {save_dir}/interrupted_model.zip에 저장됨")

    finally:
        # 최종 커리큘럼 상태 출력
        print("\n📈 최종 커리큘럼 상태:")
        final_stats = curriculum_callback.get_current_stats()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

        env.close()


if __name__ == "__main__":
    main()
