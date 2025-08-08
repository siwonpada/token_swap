from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    커리큘럼 학습을 위한 콜백 클래스
    노드 개수를 점진적으로 증가시켜 학습 난이도를 조절합니다.
    """

    def __init__(
        self,
        initial_node_num: int = 4,
        target_node_num: int = 20,
        success_threshold: float = 0.8,
        min_episodes_before_increase: int = 100,
        check_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Args:
            initial_node_num: 시작 노드 개수
            target_node_num: 목표 노드 개수
            success_threshold: 다음 단계로 넘어가기 위한 성공률 임계값
            min_episodes_before_increase: 노드 수 증가 전 최소 에피소드 수
            check_freq: 성과 평가 주기 (timesteps)
            verbose: 로그 출력 레벨
        """
        super().__init__(verbose)
        self.initial_node_num = initial_node_num
        self.current_node_num = initial_node_num
        self.target_node_num = target_node_num
        self.success_threshold = success_threshold
        self.min_episodes_before_increase = min_episodes_before_increase
        self.check_freq = check_freq

        # 통계 추적용 변수들
        self.episode_rewards = []
        self.episode_successes = []
        self.episodes_at_current_level = 0
        self.last_check_timestep = 0

        # 노드 수 증가 단계 정의 (점진적 증가)
        self.node_levels = self._generate_node_levels()
        self.current_level_idx = 0

    def _generate_node_levels(self) -> list:
        """노드 수 단계를 생성합니다."""
        if self.target_node_num <= self.initial_node_num:
            return [self.initial_node_num]

        # 2개씩 점진적으로 증가
        levels = []
        current = self.initial_node_num
        while current <= self.target_node_num:
            levels.append(current)
            current += 2

        # 마지막이 target과 다르면 target 추가
        if levels[-1] != self.target_node_num:
            levels.append(self.target_node_num)

        return levels

    def _init_callback(self) -> None:
        """콜백 초기화"""
        if self.verbose >= 1:
            print(f"커리큘럼 학습 시작: {self.initial_node_num}개 노드")
            print(f"노드 수 단계: {self.node_levels}")

    def _on_step(self) -> bool:
        """매 스텝마다 호출되는 메서드"""
        # 에피소드가 끝났는지 확인
        if hasattr(self.locals, "dones") and any(self.locals["dones"]):
            self._on_episode_end()

        # 주기적으로 성과 평가 및 난이도 조절
        if self.num_timesteps - self.last_check_timestep >= self.check_freq:
            self._evaluate_and_adjust_difficulty()
            self.last_check_timestep = self.num_timesteps

        return True

    def _on_episode_end(self) -> None:
        """에피소드 종료 시 호출"""
        self.episodes_at_current_level += 1

        # 에피소드 통계 수집
        if hasattr(self.locals, "infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    episode_success = (
                        episode_reward > 0
                    )  # 성공 여부 (보상이 양수면 성공)

                    self.episode_rewards.append(episode_reward)
                    self.episode_successes.append(episode_success)

                    # 최근 100개 에피소드만 유지
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_successes.pop(0)

    def _evaluate_and_adjust_difficulty(self) -> None:
        """성과 평가 후 난이도 조절"""
        if (
            len(self.episode_successes) < 20  # 최소 20개 에피소드 필요
            or self.episodes_at_current_level < self.min_episodes_before_increase
            or self.current_level_idx >= len(self.node_levels) - 1
        ):  # 이미 최고 단계
            return

        # 최근 성공률 계산
        recent_success_rate = sum(self.episode_successes[-50:]) / min(
            50, len(self.episode_successes)
        )

        if self.verbose >= 1:
            avg_reward = sum(self.episode_rewards[-20:]) / min(
                20, len(self.episode_rewards)
            )
            print(
                f"노드 {self.current_node_num}개 - 성공률: {recent_success_rate:.3f}, "
                f"평균 보상: {avg_reward:.3f}, 에피소드: {self.episodes_at_current_level}"
            )

        # 성공률이 임계값을 넘으면 다음 단계로
        if recent_success_rate >= self.success_threshold:
            self._increase_difficulty()

    def _increase_difficulty(self) -> None:
        """난이도 증가 (노드 수 증가)"""
        if self.current_level_idx < len(self.node_levels) - 1:
            self.current_level_idx += 1
            self.current_node_num = self.node_levels[self.current_level_idx]
            self.episodes_at_current_level = 0

            # 환경 업데이트
            self._update_environment()

            if self.verbose >= 1:
                print(f"🎯 난이도 증가! 노드 수: {self.current_node_num}개")

    def _update_environment(self) -> None:
        """환경의 노드 수 업데이트"""
        try:
            # 환경이 VecEnv인 경우
            if hasattr(self.training_env, "set_attr"):
                self.training_env.set_attr("node_num", self.current_node_num)
                if self.verbose >= 2:
                    print(f"VecEnv 노드 수를 {self.current_node_num}개로 업데이트")

            # 단일 환경인 경우 - 안전한 방식으로 속성 설정
            elif hasattr(self.training_env, "unwrapped"):
                env = self.training_env.unwrapped
                if hasattr(env, "node_num"):
                    setattr(env, "node_num", self.current_node_num)
                    if self.verbose >= 2:
                        print(
                            f"Unwrapped Env 노드 수를 {self.current_node_num}개로 업데이트"
                        )

            # 직접 접근 시도
            else:
                # 런타임에 속성 존재 여부 확인 후 안전하게 설정
                try:
                    if hasattr(self.training_env, "node_num"):
                        setattr(self.training_env, "node_num", self.current_node_num)
                        if self.verbose >= 2:
                            print(
                                f"Direct Env 노드 수를 {self.current_node_num}개로 업데이트"
                            )
                except AttributeError:
                    pass  # 속성 설정 실패는 무시

            # 환경 리셋 (새로운 노드 수 적용)
            if hasattr(self.training_env, "reset"):
                self.training_env.reset()

        except Exception as e:
            if self.verbose >= 1:
                print(f"환경 업데이트 중 오류 발생: {e}")
                print(
                    "환경 업데이트를 건너뜁니다. 수동으로 환경을 재생성해야 할 수 있습니다."
                )

    def get_current_stats(self) -> dict:
        """현재 커리큘럼 상태 반환"""
        if len(self.episode_successes) > 0:
            recent_success_rate = sum(self.episode_successes[-20:]) / min(
                20, len(self.episode_successes)
            )
            avg_reward = sum(self.episode_rewards[-20:]) / min(
                20, len(self.episode_rewards)
            )
        else:
            recent_success_rate = 0.0
            avg_reward = 0.0

        return {
            "current_node_num": self.current_node_num,
            "current_level": self.current_level_idx + 1,
            "total_levels": len(self.node_levels),
            "episodes_at_level": self.episodes_at_current_level,
            "recent_success_rate": recent_success_rate,
            "recent_avg_reward": avg_reward,
            "progress": (self.current_level_idx + 1) / len(self.node_levels),
        }
