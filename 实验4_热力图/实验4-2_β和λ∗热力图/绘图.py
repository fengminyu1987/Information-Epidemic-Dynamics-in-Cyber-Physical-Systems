import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# 从 CSV 文件加载数据
infection_path = "D:\第一篇论文\实验代码\最新代码（完整版）\\实验4_热力图\\实验4-2_β和λ∗热力图\\hypergraph_infection_density_matrix_beta_lambda_star.csv"
awareness_path = "D:\第一篇论文\实验代码\最新代码（完整版）\\实验4_热力图\\实验4-2_β和λ∗热力图\\hypergraph_awareness_density_matrix_beta_lambda_star.csv"

infection_density_matrix = np.loadtxt(infection_path, delimiter=",")
awareness_density_matrix = np.loadtxt(awareness_path, delimiter=",")

# 指定保存 PDF 的目标文件夹
pdf_output_dir = r"D:\第一篇论文\实验代码\最新代码（完整版）\实验4_热力图\实验4-2_β和λ∗热力图"
os.makedirs(pdf_output_dir, exist_ok=True)


# 设置默认字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['mathtext.rm'] = 'Times New Roman'
# mpl.rcParams['mathtext.default'] = 'rm'  # 默认使用罗马体渲染数学

# === 原始热力图 ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(awareness_density_matrix.T, origin='lower', aspect='auto', extent=[0, 1, 0, 1], cmap='hot')
cbar = plt.colorbar()
cbar.ax.set_title(r'$\rho^A$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda^*$')

plt.subplot(1, 2, 2)
plt.imshow(infection_density_matrix.T, origin='lower', aspect='auto', extent=[0, 1, 0, 1], cmap='hot')
cbar = plt.colorbar()
cbar.ax.set_title(r'$\rho^I$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda^*$')

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-2_β和λ∗原始热力图.pdf"), format='pdf')
plt.show()

# === 感知密度热力图 ===
plt.figure(figsize=(10, 6))
plt.imshow(
    awareness_density_matrix.T,
    cmap='jet', aspect='auto', origin='lower',
    extent=[0, 1, 0, 1],
    interpolation='gaussian'
)
cbar = plt.colorbar()
# 手动设置颜色条刻度为每次增加 0.2
ticks = np.arange(0, 0.9, 0.2)  # 设置从 0 到 0.9 的刻度，每次增加 0.2
cbar.set_ticks(ticks)
cbar.ax.set_title(r'$\rho^A$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda^*$')

ticks = np.arange(0, 1.01, 0.2)

plt.xticks(
    ticks,
    ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
)

plt.yticks(
    ticks,
    ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
)

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-2_感知β和λ∗热力图.pdf"), format='pdf')
plt.show()

# === 感染密度热力图 ===
plt.figure(figsize=(10, 6))
plt.imshow(
    infection_density_matrix.T,
    cmap='jet', aspect='auto', origin='lower',
    extent=[0, 1, 0, 1],
    interpolation='gaussian'
)
cbar = plt.colorbar()
cbar.ax.set_title(r'$\rho^I$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda^*$')

ticks = np.arange(0, 1.01, 0.2)

plt.xticks(
    ticks,
    ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
)

plt.yticks(
    ticks,
    ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
)

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-2_感染β和λ∗热力图.pdf"), format='pdf')
plt.show()
