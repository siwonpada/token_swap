"""
커리큘럼 콜백 테스트 스크립트
"""

from src.callbacks.Curriculum_callback import CurriculumCallback


def test_curriculum_callback():
    """커리큘럼 콜백의 기본 기능을 테스트"""

    print("🧪 커리큘럼 콜백 테스트 시작")

    # 콜백 인스턴스 생성
    callback = CurriculumCallback(
        initial_node_num=4,
        target_node_num=12,
        success_threshold=0.8,
        min_episodes_before_increase=10,
        check_freq=100,
        verbose=1,
    )

    print("\n📋 초기 설정:")
    print(f"  시작 노드 수: {callback.initial_node_num}")
    print(f"  목표 노드 수: {callback.target_node_num}")
    print(f"  성공 임계값: {callback.success_threshold}")
    print(f"  노드 레벨: {callback.node_levels}")

    # 초기 상태 확인
    initial_stats = callback.get_current_stats()
    print("\n📊 초기 상태:")
    for key, value in initial_stats.items():
        print(f"  {key}: {value}")

    # 가상 에피소드 시뮬레이션
    print("\n🎮 가상 에피소드 시뮬레이션:")

    # 성공적인 에피소드들 추가
    for i in range(15):
        # 높은 성공률 시뮬레이션
        success = i < 12  # 12/15 = 80% 성공률
        reward = 1.0 if success else -0.5

        callback.episode_rewards.append(reward)
        callback.episode_successes.append(success)
        callback.episodes_at_current_level += 1

        if i % 5 == 4:  # 5 에피소드마다 상태 출력
            stats = callback.get_current_stats()
            print(
                f"  에피소드 {i + 1}: 성공률 {stats['recent_success_rate']:.2f}, "
                f"노드 수 {stats['current_node_num']}"
            )

    # 난이도 증가 테스트
    print("\n🔄 난이도 증가 테스트:")
    old_node_num = callback.current_node_num
    callback._evaluate_and_adjust_difficulty()

    if callback.current_node_num > old_node_num:
        print(
            f"  ✅ 성공! 노드 수가 {old_node_num}에서 {callback.current_node_num}로 증가"
        )
    else:
        print(f"  ℹ️ 아직 증가하지 않음. 현재 노드 수: {callback.current_node_num}")

    # 최종 상태
    final_stats = callback.get_current_stats()
    print("\n📈 최종 상태:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    print("\n✅ 테스트 완료!")


def test_node_levels():
    """노드 레벨 생성 테스트"""
    print("\n🔢 노드 레벨 생성 테스트:")

    test_cases = [
        (4, 8),  # 4 -> 8
        (4, 12),  # 4 -> 12
        (6, 20),  # 6 -> 20
        (4, 4),  # 동일한 값
    ]

    for initial, target in test_cases:
        callback = CurriculumCallback(initial_node_num=initial, target_node_num=target)
        print(f"  {initial} → {target}: {callback.node_levels}")


if __name__ == "__main__":
    test_curriculum_callback()
    test_node_levels()
