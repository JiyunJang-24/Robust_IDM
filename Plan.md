## 사용자가 하고 싶은 실험 설계

1. Diffusion Policy를 lerobot으로 Train해서 LIBERO 환경에서 Eval해 보는 것
2. 이때, Diffusion Policy는 기본 주어진 버전이 아니라 튜닝해서 사용해야 함. IDM 형태(Pi(a_t|o_t+H,o_t))로 사용할 예정, Goal Conditioned Image를 주고, Goal Conditioned Image에 도달하는 Action을 만들어낼 수 있는 IDM의 형태를 만드는 것이 목표.
3. 이때, Train data를 ReWiND(되감기) 하는 코드를 작성해서, Observation Frame과 그에 따른 GT Action Data도 역으로 만든 데이터셋을 추가로 제작.
4. 원래 데이터셋으로만 학습했을 때와, ReWiND 한 데이터셋으로 학습했을 때의 성능 차이를 비교.
5. 4번에서의 가설은, ReWiND Dataset이 같은 O_t에 대해 여러 가지 O_t+H와 그에 따른 Action Trajectory를 경험하게 되므로, O_t+H를 좀 더 충실히 따라갈 수 있는 Robust한 IDM에 도움이 될 것이라는 가설.

## 실험 명세
Task_suite : LIBERO_10
Evaluation Metric : Success rate, 얼마나 Goal Conditioned Image를 잘 따라갔는지(EEF distance from GT? 고민 필요.)