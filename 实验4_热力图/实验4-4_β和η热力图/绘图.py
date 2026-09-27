import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import os
from matplotlib.ticker import FuncFormatter

# 从 CSV 文件加载数据
infection_path = "D:\第一篇论文\实验代码\最新代码（完整版）\实验4_热力图/实验4-4_β和η热力图/hypergraph_infection_density_matrix_beta_eta.csv"
awareness_path = "D:\第一篇论文\实验代码\最新代码（完整版）\实验4_热力图/实验4-4_β和η热力图/hypergraph_awareness_density_matrix_beta_eta.csv"

infection_density_matrix = np.loadtxt(infection_path, delimiter=",")
awareness_density_matrix = np.loadtxt(awareness_path, delimiter=",")

# 指定保存 PDF 的目标文件夹
pdf_output_dir = "D:\第一篇论文\实验代码\最新代码（完整版）\实验4_热力图\实验4-4_β和η热力图"
os.makedirs(pdf_output_dir, exist_ok=True)

# # 设置中文字体（如果需要显示中文标题）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置默认字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# === 原始热力图 ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(awareness_density_matrix.T, origin='lower', aspect='auto', 
           extent=[0, 1, 0, 2], cmap='hot')  # β范围0-1，η范围0-2
cbar = plt.colorbar()
cbar.ax.set_title(r'$\rho^A$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\eta$')

plt.subplot(1, 2, 2)
plt.imshow(infection_density_matrix.T, origin='lower', aspect='auto', 
           extent=[0, 1, 0, 2], cmap='hot')  # β范围0-1，η范围0-2
cbar = plt.colorbar()
cbar.ax.set_title(r'$\rho^I$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\eta$')

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-4_β和η原始热力图.pdf"), format='pdf')
plt.show()

# === 感知密度热力图 ===
plt.figure(figsize=(10, 6))
plt.imshow(
    awareness_density_matrix.T,
    cmap='jet', aspect='auto', origin='lower',
    extent=[0, 1, 0, 2],  # β范围0-1，η范围0-2
    interpolation='gaussian'
)

cbar = plt.colorbar()
cbar.set_ticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])  # 设置刻度，从0开始
cbar.ax.set_title(r'$\rho^A$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\eta$')

def x_formatter(x, pos):
    if np.isclose(x, 0):
        return '0'
    return f'{x:.1f}'

def y_formatter(y, pos):
    if np.isclose(y, 0):
        return '0'
    return f'{y:.2f}'

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-4_感知β和η热力图.pdf"), format='pdf')
plt.show()

# === 感染密度热力图 ===
plt.figure(figsize=(10, 6))
plt.imshow(
    infection_density_matrix.T,
    cmap='jet', aspect='auto', origin='lower',
    extent=[0, 1, 0, 2],  # β范围0-1，η范围0-2
    interpolation='gaussian'
)
cbar = plt.colorbar()
cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # 设置刻度，从0开始
cbar.ax.set_title(r'$\rho^I$', pad=10)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\eta$')

def x_formatter(x, pos):
    if np.isclose(x, 0):
        return '0'
    return f'{x:.1f}'

def y_formatter(y, pos):
    if np.isclose(y, 0):
        return '0'
    return f'{y:.2f}'

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

plt.tight_layout()
plt.savefig(os.path.join(pdf_output_dir, "实验4-4_感染β和η热力图.pdf"), format='pdf')
plt.show()

