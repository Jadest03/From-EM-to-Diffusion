import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic') 
plt.rcParams['axes.unicode_minus'] = False

# 실제 데이터 가져오기
iris = load_iris()
xs_real = iris.data # 4차원 데이터 전체
xs_real = (xs_real - np.mean(xs_real, axis=0)) / np.std(xs_real, axis=0) # 표준화
true_labels = iris.target

# 학습된 파라미터 로드
data = np.load('gmm_params.npz')
phis = data['phis']
mus = data['mus']
covs = data['covs']
K, D = mus.shape
N_gen = 500 

# 데이터 생성 및 분류 (Sampling)
new_xs = np.zeros((N_gen, D)) # (500, 4)
new_labels = np.zeros(N_gen, dtype=int) # (500,)

for n in range(N_gen):
    k = np.random.choice(K, p=phis) # 가우시안 분포 선택
    new_labels[n] = k
    new_xs[n] = np.random.multivariate_normal(mus[k], covs[k]) # 선택된 가우시안에서 점 추출

# 시각화 및 비교
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(xs_real[:, 0], xs_real[:, 1], c=true_labels, cmap='viridis', alpha=0.5, s=20)
ax1.set_title('실제 붓꽃 데이터 분포 (True Labels)')
ax1.set_xlabel('Sepal Length (표준화)')
ax1.set_ylabel('Sepal Width (표준화)')

ax2.scatter(new_xs[:, 0], new_xs[:, 1], c=new_labels, cmap='viridis', alpha=0.5, s=20)
ax2.scatter(mus[:, 0], mus[:, 1], c='red', marker='X', s=150, label='Learned Centers')
ax2.set_title(f'GMM이 생성한 가상 데이터 (N={N_gen})')
ax2.set_xlabel('Sepal Length (표준화)')
ax2.set_ylabel('Sepal Width (표준화)')
ax2.legend()

plt.tight_layout()
plt.show()