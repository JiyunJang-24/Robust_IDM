# 결과물 인덱스 (diffusion_idm / ReWiND 실험)

> "이 폴더/파일 = 이 방식으로 구현된 결과물" 매핑. 경로는 모두 `/workspace` 기준.
> 핵심 결론: **task-id 매핑 버그 + EGL hang 수정 후**, ReWiND(시간 역재생 augmentation)가
> unseen goal-following을 개선함이 success·reach·probe·attribution으로 일관 입증됨.

---

## 1. 모델 체크포인트 (`lerobot/outputs/idm_*`)

| 폴더 | 학습 방식 | 설명 |
|---|---|---|
| `idm_baseline`, `idm_rewind` | **h8weighted** (현재 최신) | `goal_offset_peak_prob=0.6` (H=8을 60% 샘플링), agentview-only. baseline=원본 270ep, rewind=combined 540ep |
| `idm_baseline_h8unif`, `idm_rewind_h8unif` | **h8 uniform** | `peak_prob=0` (H~Uniform{1..8}), agentview-only. H분포 변경 전 버전 |
| `idm_baseline_wrist`, `idm_rewind_wrist` | **uniform + wrist** | `peak_prob=0`, `image_keys=[agentview, wrist_image]` (obs+goal 둘 다 wrist 추가) |
| `idm_*_drop11`, `idm_*_h16d0`, `idm_*_H16old` | 구버전 (archive) | 초기 horizon=16 / drop_n_last_frames 실험. H16old=ReWiND reversed-only 버그 시절 |

학습 스크립트: [scripts/train_idm.sh](scripts/train_idm.sh) — env `PEAK_PROB`, `IMAGE_KEYS`, `OUT`, `CUDA_VISIBLE_DEVICES` 로 변형.

---

## 2. 평가 결과 (success rate + eef reach) — `rollout_eval_out_*`

각 폴더에 `results_{baseline,rewind}.json` (per-demo) + `summary_*.json` + per-(task/demo) mp4·reach.png.

| 폴더 | 평가 대상 | 비고 |
|---|---|---|
| `rollout_eval_out_fixedmap` | **h8unif** baseline/rewind | task-id 매핑 **수정 후** 첫 공정 평가. ReWiND: unseen reach 2.9×, success 0→20% |
| `rollout_eval_out_h8weighted` | **h8weighted** baseline/rewind | 최고 성능. rewind overall 62%, seen 70%, unseen 30% |
| `rollout_eval_out_wrist` | **wrist** baseline/rewind | wrist obs+goal 추가 (ROT 개선 가설 검증용) |
| `rollout_eval_out_drop0`, `_drop11` | 구버전 (archive) | task-id 버그 시절 — **신뢰 불가** |

생성: [scripts/rollout_eval_isolated.sh](scripts/rollout_eval_isolated.sh) (task별/unseen은 demo별 프로세스 격리 — EGL hang 회피) → [merge_eval_results.py](merge_eval_results.py) 병합.
비교: [compare_eval.py](compare_eval.py) (`--dir`, `--variants`) — success·reach(pos/rot/grip) 표.

---

## 3. "모델이 goal을 보는가" 분석 — 4가지 방식

### 3a. Goal-influence probe (정량, sim-free) — `probe_*.json`
[goal_influence_probe.py](goal_influence_probe.py): o_t 고정, **goal을 swap**했을 때 action 변화량(influence). position(init/mid)별.
- `probe_*_h8w.json` = h8weighted. **결과: mid에서 rewind influence가 baseline의 ~2.5×** (goal 의존 입증).
- `probe_*_h8.json` = h8unif, `probe_*_drop0/new.json` = 구버전.

### 3b. Input-stream attribution (정량, obs vs goal) — `input_attribution.{png,json}`
[input_attribution.py](input_attribution.py): o_t 이미지 / goal 이미지를 각각 **평균이미지로 ablate**해 sens 비교 → `goal_share`.
- **결과: mid에서 goal_share baseline ~40% vs rewind ~68%** (의존이 o_t→goal로 역전).

### 3c. SpatialSoftmax attention 영상 (representation 층위) — `rollout_attn_smoke/`
[rollout_eval.py](rollout_eval.py) `--attention`: 각 인코더의 SpatialSoftmax attention을 o_t·goal 패널에 overlay.
- "**학습된 인코더가 이미지에서 정보를 뽑는 위치**" (각 이미지 self-normalized, 모델 간 절대 비교 X).
- `rollout_attn_smoke/{baseline,rewind}/...task9...mp4`. 비교 frame: `_attn_compare_task9.png`.

### 3d. Gradient saliency 영상 (decision 층위) — `rollout_sal_smoke/`, `rollout_sal_cmp/`
[rollout_eval.py](rollout_eval.py) `--saliency`: `∂‖action‖/∂입력`을 o_t·goal **공통 절대 스케일**로 overlay.
- "**그 입력이 action을 바꾸는 정도**" (encoder→U-Net 전체 통과, 밝은 패널 = 더 의존).
- `rollout_sal_cmp/{baseline,rewind}/...task9...mp4` = 비교용. 비교 frame: `_sal_cmp_f{40,90}.png`.
- ⚠️ 현재 frame별 percentile 정규화 + gradient noise로 모델 간 시각 차이가 옅음 → **goal_share% 오버레이 + 공통 스케일로 다듬는 중**.

**층위 구분**: 3a/3b = 의존도 *정량*, 3c = representation(인코더가 보는 곳), 3d = decision(action이 쓰는 곳).

---

## 4. 인프라 / 버그 수정 스크립트

| 파일 | 역할 |
|---|---|
| [libero_task_map.py](libero_task_map.py) | dataset task_index ↔ LIBERO benchmark task_id를 **language로 매핑** (1:1 검증). success=0의 근본 버그(매핑 어긋남) 해결, 모든 suite 재사용 |
| [diag_check_success.py](diag_check_success.py) | check_success가 정상임을 GT-replay로 검증한 진단 (버그는 매핑이었음) |
| [scripts/rollout_eval_isolated.sh](scripts/rollout_eval_isolated.sh) | EGL render 누수 hang 회피 — task별(+unseen은 demo별) 프로세스 격리 |

---

## 임시 비교 이미지 (재생성 가능, 삭제해도 무방)
`_attn_frame_*.png`, `_attn_compare_task9.png`, `_sal_frame_60.png`, `_sal_cmp_f*.png`
