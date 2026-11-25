# 실시간 리얼리티 뷰 · 공개 데이터 실험

## Windows 개발환경

기존 웹캠 HTML은 실험 기준선으로 사용한다. 프로젝트 폴더에서 아래 명령을 실행한다. Python 3.10 x64와 Git이 필요하다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
python -m pip check
python -c "import jax, cv2; print(jax.__version__, jax.devices(), cv2.__version__)"
```

PowerShell에서 활성화가 제한된 경우 아래 예제의 `python`을 `.\.venv\Scripts\python.exe`로 바꿔 실행한다.

| 패키지 | 용도 |
| --- | --- |
| NumPy, Pillow | 합성 영상, 좌표 배열, 깊이 역투영, TAPIR 입력 리사이즈 |
| OpenCV | 웹캠과 동영상 입력, 향후 추적기 비교 |
| JAX / jaxlib 0.6.2 | Python 3.10용 Windows CPU 추론 |
| TAPNet, einops | Online TAPIR 모델과 유틸리티. Haiku, Chex, Optax 등은 TAPNet 의존성으로 설치 |

[JAX의 Windows 설치 안내](https://docs.jax.dev/en/latest/installation.html)에 따라 네이티브 Windows에서는 CPU로 실행한다. 이 환경의 실행 시간으로 GPU 라이브 성능을 추정하지 않는다.

```powershell
New-Item -ItemType Directory -Force checkpoints,outputs | Out-Null
curl.exe --fail --location --output checkpoints/causal_tapir_checkpoint.npy https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

StreamVGGT는 [공식 설치법](https://github.com/wzzheng/StreamVGGT#installation)에 맞춰 별도 Python 3.11 환경에서 PyTorch와 프로젝트 requirements를 구성한다. GTX 1650의 4 GB VRAM에서 순차 복원의 실행 가능 여부는 검증 전이다.

## 1. 시계열 추적 비교

정답 이동량이 있는 합성 프레임에 밝기 변화, 블러, 완전 가림을 순서대로 넣는다. 이 입력은 반복 가능한 계측 기준이다. 장면의 합성 영상 생성 모델을 검토할 때는 ReCamMaster 등으로 만든 영상에 대해 관측 영상과의 시점 충실도를 따로 평가한다.

```powershell
python reality_benchmark.py track --export synthetic.npz
```

공개 실제 영상은 [TAP-Vid-DAVIS 공식 배포](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md)의 pickle을 압축 해제해 사용한다. pickle은 공식 출처에서 받은 파일만 읽는다. 합성 장면 공개 데이터로는 [TAP-Vid-Kubric](https://tapvid.github.io/)이 있다. 현재 입력 어댑터는 DAVIS 형식이다.

```powershell
python reality_benchmark.py track --davis tapvid_davis.pkl --points 8 --export davis.npz
```

[공식 TAPIR 저장소](https://github.com/google-deepmind/tapnet)의 JAX 환경을 구성하고 공식 causal TAPIR 체크포인트를 내려받는다. 아래 어댑터는 첫 관측 시점을 기준으로 질의를 시작하고 이후 프레임을 순차 추론한다.

```powershell
python reality_tapir.py davis.npz causal_tapir_checkpoint.npy davis-ai.npz
python reality_benchmark.py track --davis tapvid_davis.pkl --points 8 --predictions davis-ai.npz
```

두 번째 명령에서 데이터, 클립, 점의 개수를 동일하게 지정한다. 보고서의 4px 이내 비율은 실제로 보이는 프레임에서 추적 성공률과 위치 오차를 함께 본다. 이 간이 평가에는 TAP-Vid 공식 Average Jaccard가 포함되지 않는다. TAPIR 실행 시간에는 컴파일이 포함된다. 라이브 적용에서는 워밍업 이후의 프레임별 지연 p95를 별도로 잰다.

## 2. RGB 기반 3D와 관측 깊이 비교

[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)의 Freiburg 1 구간부터 시작한다. RGB만 [StreamVGGT](https://github.com/wzzheng/StreamVGGT)에 넣고 같은 시점의 등록된 깊이를 비교용으로 보관한다. RGB와 깊이는 데이터의 타임스탬프로 매칭한다. 단일 프레임 기준 점군은 다음과 같이 만든다. 예시 보정값은 Freiburg 1 RGB 카메라 기준이며 사용할 프레임의 등록·왜곡 처리와 맞춰야 한다. 점군 스크립트는 핀홀 모델을 사용한다.

```powershell
python reality_benchmark.py geometry --depth matched-depth.png --rgb matched-rgb.png --output observed.ply --fx 517.3 --fy 516.5 --cx 318.6 --cy 255.3 --factor 5000
```

다음 평가에서는 일부 프레임의 참조 깊이로 모델의 거리 척도를 한 번 맞추고, 다른 프레임에서 깊이 절대오차와 유효 영역 비율을 보고한다. 전체 궤적은 모션 캡처 참조 궤적과 비교한다. TUM 깊이 영상은 센서 관측값이며 독립 측량 기준을 대신하기에 오차가 있다. 가상 시점 화질은 입력에서 제외한 실제 카메라 영상과 비교한다.

## 현재 확인 결과

- 합성 추적에서 가시 64프레임 중 18프레임 누락. 총 가시 프레임 기준 4px 이내 71.9%. 가림 종료 후 3프레임 만에 4px 이내로 복구.
- 합성 깊이 1m의 점군 역투영 확인. StreamVGGT 순차 복원은 실행 전.

- Windows CPU 가상환경에서 의존성 검사, JAX JIT, 공식 체크포인트를 사용한 합성 3프레임·1점 Online TAPIR 추론을 확인했다. 추론은 최초 컴파일 포함 8.70초였고, 라이브 처리 시간은 별도 측정이 필요하다.

## TAP-Vid-DAVIS 비교 결과

공식 배포본의 이름순 첫 클립 `bike-packing`에서 첫 8개 추적점을 사용했다. 영상은 910×480, 69프레임이며 시작점과 시작 프레임을 공유한다. Python 기준선은 원본 해상도에서 HTML의 고정 템플릿 규칙을 실행한다. TAPIR는 256×256으로 리사이즈해 추론한 뒤 좌표를 원본 해상도로 환산한다.

| 지표 | 기준선 | Online TAPIR |
| --- | ---: | ---: |
| 표시한 가시 점의 중앙 위치 오차 | 91.03 px | 7.04 px |
| 가시 점-프레임 누락 | 225/390 (57.7%) | 96/390 (24.6%) |
| 전체 가시 점-프레임 중 4 px 이내 | 12/390 (3.1%) | 91/390 (23.3%) |
| 가려진 점을 표시 | 41/154 (26.6%) | 79/154 (51.3%) |
| 가림이 끝난 뒤 4 px 이내 복구 | 0/6 | 2/6 (0, 7프레임) |

복구는 다시 가려지기 전까지 평가한다. TAPIR의 CPU 실행 시간은 최초 컴파일을 포함해 24.42초였다. 단일 클립의 파일 입력 실험이다. 전체 데이터셋 성능, 실제 웹캠 지연, StreamVGGT는 추가 검증이 필요하다. 수치 원본은 `outputs/davis-comparison.json`에 있다.
