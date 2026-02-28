# From EM to Diffusion

본 저장소는 생성형 AI(Generative AI)의 수학적 근간이 되는 고전적인 통계 알고리즘인 EM 알고리즘부터, 최근 가장 SOTA 모델인 확산 모델(Diffusion Models)까지 직접 구현하고 실험하고 기록한 저장소이다.

잠재 변수(Latent Variable)를 추정하는 기초적인 아이디어가 어떻게 딥러닝과 결합하여 고품질의 이미지를 생성해 내는지, 그 발전의 궤적을 코드로 따라간다.

## Repository Structure & Modules

본 저장소의 구조는 다음과 같이 구성되어 있다.

### 1. `EM-Algorithm` (Expectation-Maximization)
생성 모델의 통계적 기초가 되는 EM 알고리즘으로 관측되지 않은 잠재 변수 $Z$가 존재할 때, Maximum Likelihood를 EM 알고리즘을 통해 수학적으로 구하고 이를 구현하였다.

### 2. `VAE` (Variational AutoEncoder)
변분 추론(Variational Inference)과 신경망을 결합하여, 잠재 공간(Latent Space)을 연속적으로 학습하고 새로운 데이터를 생성하는 모델이다. 
ELBO(Evidence Lower Bound)를 목적 함수로 사용하여 모델을 최적화한다.

### 3. `Diffusion(DDPM)` (Denoising Diffusion Probabilistic Models)
데이터에 노이즈를 점진적으로 추가하는 확산 과정(Forward Process)과, 이를 다시 복원하는 역확산 과정(Reverse Process)을 학습하여 고해상도 이미지를 생성하는 모델이다.
노이즈를 직접 예측하는 U-Net 구조의 신경망$(ϵ_θ)$을 구축하고, 특정 타임스텝 $t$에서 데이터에 더해진 가우시안 노이즈를 추정하여 점진적으로 제거해 나가는 마르코프 체인(Markov Chain) 방식을 뼈대로 한다.

### 4. `CFG` (Classifier-Free Guidance)
조건부 생성(Conditional Generation) 시, 별도로 학습된 노이즈 분류기(Classifier) 없이도 조건(Prompt 등)의 반영률을 극대화할 수 있도록 고안된 점수 추정(Score Estimation) 기법이다.
학습 과정에서 조건을 일정 확률로 제거(Null)하여 무조건부(Unconditional) 모델과 조건부(Conditional) 모델을 동시에 학습시킨다. 추론 시에는 두 예측값의 차이를 가이던스 스케일($\gamma$)만큼 증폭시켜, 사용자가 원하는 조건에 훨씬 더 강하게 부합하는 고품질의 결과물을 유도해 낸다.