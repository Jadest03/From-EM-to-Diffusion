import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic') 
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
iris = load_iris()
xs = iris.data # (N, 4)
xs = (xs - np.mean(xs, axis=0)) / np.std(xs, axis=0) # 표준 정규분포

# 매개변수
K = 3
N, D = xs.shape
phis = np.array([1/K] * K)
mus = np.random.randn(K, D)
covs = np.array([np.eye(D) for _ in range(K)])

# 하이퍼 파라미터
MAX_ITERS = 100
THRESHOLD = 1e-8

def multivariate_normal(x, mu, cov):
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    d = len(x)
    z = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov) 
    y = z * np.exp((x - mu).T @ inv_cov @ (x - mu) / -2.0)
    return y

def gmm(x, phis, mus, covs):
    K = len(phis)
    total_p = 0
    for k in range(K):
        phi = phis[k]
        mu = mus[k]
        cov = covs[k]
        p_k = phi * multivariate_normal(x, mu, cov)
        total_p += p_k
    return total_p

# 로그가능도 평균
def likelihood(xs, phis, mus, covs):
    eps = 1e-8 # log(0) 방지
    N = len(xs)
    log_like = 0
    for x in xs:
        y = gmm(x, phis, mus, covs)
        log_like += np.log(y + eps)
    return log_like / N 

# 학습
current_likelihood = likelihood(xs, phis, mus, covs)
for iter in range(MAX_ITERS):
    # E-스텝
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)
            
    # M-스텝
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        # phis
        phis[k] = qs_sum[k] / N
        
        # mus
        z = 0
        for n in range(N):
            z += qs[n, k] * xs[n]
        mus[k] = z / qs_sum[k]
        
        z = 0
        for n in range(N):
            f = xs[n] - mus[k]
            f = f[:, np.newaxis]
            z += qs[n, k] * f @ f.T
            
        covs[k] = z / qs_sum[k]
        
    # 종료 조건
    print(f'현재 로그 가능도 : {current_likelihood:.3f}')
    
    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(current_likelihood - next_likelihood)
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood
    
# 파라미터 저장
np.savez('gmm_params.npz', phis=phis, mus=mus, covs=covs)     
print("학습된 파라미터가 gmm_params.npz에 저장되었습니다.")

# 시각화 및 비교
true_labels = iris.target
predicted_labels = np.argmax(qs, axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.scatter(xs[:, 0], xs[:, 1], c=true_labels, cmap='viridis', s=10)
ax1.set_title('실제 분포')

ax2.scatter(xs[:, 0], xs[:, 1], c=predicted_labels, cmap='viridis', s=10)
ax2.scatter(mus[:, 0], mus[:, 1], c='red', marker='X', s=100, label='Means') # 평균값 시각화
ax2.set_title('GMM을 통한 군집화 예측 분포')

plt.show()