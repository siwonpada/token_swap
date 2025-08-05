# Envs

Token Swapping Problem의 핵심적인 로직들만 구현한 환경입니다.  
문제의 정의대로, Tokens swap이 이루어지는 그래프와, 처음 mapping과 나중 mapping을 함께 생성하고, 이를 풀 수 있도록 환경을 구성하였습니다. observation은 그래프의 형태와 현재 mapping과 목표 mapping으로 하였으며, 여기에서 데이터를 수정하기 위해서는 env자체를 고치는 것이 아니라 wrapper로 감싸서 수정을 하는 방식으로 코드가 이루어져있다. 이는 유지 관리를 더 편리하게 하기 위해서 필수적이 것이다.