import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

beta = np.linspace(0, 2, 500)

# # 특정 감쇠비 설정 하고 싶으면 아래 4줄과 줄23을 주석 제거 -> ctrl + /

# print('감쇠비 입력.')
# D = float(input())
# print('감쇠비: {}'.format(D))
# float(D)

D1 = 0.01
D2 = 0.2
D3 = 0.4
D4 = 0.707

M1 = np.sqrt(1 / ((1 - (beta ** 2)) ** 2 + (2 * D1 * beta) ** 2))
M2 = np.sqrt(1 / ((1 - (beta ** 2)) ** 2 + (2 * D2 * beta) ** 2))
M3 = np.sqrt(1 / ((1 - (beta ** 2)) ** 2 + (2 * D3 * beta) ** 2))
M4 = np.sqrt(1 / ((1 - (beta ** 2)) ** 2 + (2 * D4 * beta) ** 2))

# data = {"beta" : beta, "D" : D, "D1" : D1, "M1" : M1, "D2" : D2, "M2" : M2, "D3" : D3, "M3" : M3, "D4" : D4, "M4" : M4}

data = {"beta" : beta, "D1" : D1, "M1" : M1, "D2" : D2, "M2" : M2, "D3" : D3, "M3" : M3, "D4" : D4, "M4" : M4}
df = pd.DataFrame(data, columns=["beta", "D1", "M1", "D2", "M2", "D3", "M3", "D4", "M4"])
print(df)

phi_1 = np.arctan2((2 * D1 * beta), (1 - beta ** 2))
phi_2 = np.arctan2((2 * D2 * beta), (1 - beta ** 2))
phi_3 = np.arctan2((2 * D3 * beta), (1 - beta ** 2))
phi_4 = np.arctan2((2 * D4 * beta), (1 - beta ** 2))

phi = pd.DataFrame({"phi_1":phi_1, "phi_2":phi_2, "pi_3":phi_3, "phi_4":phi_4})
print(phi)

# 그래프
plt.title("Response curves for a viscously damped single-degree-of-freedom system.")

plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
plt.xlabel(r'$\frac{\omega}{\omega_n}$')
plt.ylabel(r"$M=\frac{A}{Q_0/k}$")

plt.plot(beta, M1, 'r--', label='D=0.01')
plt.plot(beta, M2, 'b--', label='D=0.2')
plt.plot(beta, M3, 'g--', label='D=0.4')
plt.plot(beta, M4, 'y--', label='D=0.707')
plt.legend()

plt.xlim([0, 2])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 3])     # Y축의 범위: [ymin, ymax]
plt.grid(True)

plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
plt.xlabel(r'$\frac{\omega}{\omega_n}$')
plt.ylabel(r"$\varphi$")

plt.plot(beta, phi_1, 'r--', label='D=0.01')
plt.plot(beta, phi_2, 'b--', label='D=0.2')
plt.plot(beta, phi_3, 'g--', label='D=0.4')
plt.plot(beta, phi_4, 'y--', label='D=0.707')
plt.legend()

plt.xlim([0, 2])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 3.141592654])     # Y축의 범위: [ymin, ymax]
plt.grid(True)

plt.savefig('Response_curves.png')
plt.show()
