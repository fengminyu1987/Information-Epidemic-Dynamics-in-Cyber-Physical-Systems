import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


# 设置默认字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'


# 读取实验数据
experiment_df = pd.read_csv(r'D:\第一篇论文\实验代码\最新代码（完整版）\实验1_MC与MMCA对比\MC0609优化并行版_QS方法.csv')
# 读取理论数据
theory_df = pd.read_csv(r'D:\第一篇论文\实验代码\最新代码（完整版）\实验1_MC与MMCA对比\MMCA0609.csv')

# 提取数据
beta_U_exp = experiment_df['beta_U']
rho_I_exp = experiment_df['infection_density']
rho_A_exp = experiment_df['awareness_density']

beta_U_theory = theory_df['beta_U']
rho_I_theory = theory_df['infection_density']
rho_A_theory = theory_df['awareness_density']

""" print('实际感知密度', rho_A_exp)
print('理论感知密度', rho_A_theory)
print('实际感知密度-理论感知密度',rho_A_exp-rho_A_theory )
print('(实际感知密度-理论感知密度)的绝对值',abs(rho_A_exp-rho_A_theory)) """
print('感知密度平均绝对误差', sum(abs(rho_A_exp-rho_A_theory)) / len(rho_A_theory))
print('感知密度平均相对误差', sum(abs(rho_A_exp-rho_A_theory)/abs(rho_A_theory))/len(rho_A_theory))
print("------------------------------------------")
# # 仅对分母大于零的元素进行计算
# valid_indices = rho_I_theory > 0  # 找到分母大于零的索引
# # 根据这些索引进行误差计算
# A = abs(rho_I_exp[valid_indices] - rho_I_theory[valid_indices])
# B = sum(A / abs(rho_I_theory[valid_indices]))
""" print('实际感染密度', rho_I_exp)
print('理论感染密度', rho_I_theory)
print('实际感染密度-理论感染密度', rho_I_exp-rho_I_theory )
print('(实际感染密度-理论感染密度)的绝对值', abs(rho_I_exp-rho_I_theory)) """
print('感染密度平均绝对误差', sum(abs(rho_I_exp-rho_I_theory))/len(rho_I_theory))
print('感染密度平均相对误差', sum(abs(rho_I_exp-rho_I_theory)/abs(rho_I_theory))/len(rho_I_theory))
print("------------------------------------------")
# 计算感染密度和感知密度的平均偏差
avg_abs_deviation_I = sum(abs(rho_I_exp - rho_I_theory)) / len(rho_I_theory)
avg_abs_deviation_A = sum(abs(rho_A_exp - rho_A_theory)) / len(rho_A_theory)

# 打印平均偏差
print(f'感染密度的平均偏差: {avg_abs_deviation_I:.6f}')
print(f'感知密度的平均偏差: {avg_abs_deviation_A:.6f}')

# 绘制密度曲线
plt.figure(figsize=(10, 6))

# MMCA感染密度：红色实线
plt.plot(beta_U_theory, rho_I_theory, color='red', label=r'$\mathrm{\rho^I(MMCA)}$', linewidth=2)
# MMCA感知密度：蓝色实线
plt.plot(beta_U_theory, rho_A_theory, color='blue', label=r'$\mathrm{\rho^A(MMCA)}$', linewidth=2)

# MC感染密度：红色空心向上三角形
plt.plot(beta_U_exp, rho_I_exp, color='red', marker='^', markerfacecolor='none',
         markeredgecolor='red', label=r'$\mathrm{\rho^I(MC)}$', linewidth=0, markersize=8, markeredgewidth=2)
# MC感知密度：蓝色空心圆
plt.plot(beta_U_exp, rho_A_exp, color='blue', marker='o', markerfacecolor='none',
         markeredgecolor='blue', label=r'$\mathrm{\rho^A(MC)}$', linewidth=0, markersize=8, markeredgewidth=2)

# 自定义格式化函数，将 0.0 显示为 0
def custom_formatter(x, pos):
    if x == 0:
        return '0'
    else:
        return '{:.1f}'.format(x)
# 使用自定义的格式化函数来格式化刻度标签
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

# 添加标题和标签
plt.xlim(0, 1)
plt.ylim(0, 0.85)
plt.title('')
plt.xlabel(r'$\beta$', fontsize=21)  # 增大x轴标签字体大小
plt.ylabel(r'$\rho$', fontsize=21)  # 增大y轴标签字体大小

# 设置刻度字体大小
plt.xticks(fontsize=19)  # 增大x轴刻度字体大小
plt.yticks(fontsize=19)  # 增大y轴刻度字体大小

# 添加网格
plt.grid(True)
# 添加图例
plt.legend(fontsize=22)

# 调整图形的边距以减少空白
plt.subplots_adjust(left=0.07+0.02, right=0.96, top=0.95, bottom=0.1+0.03)

# 保存图像为PDF
plt.savefig('D:\第一篇论文\实验代码\最新代码（完整版）\\实验1_MC与MMCA对比\\MC和MMCA对比图.pdf', format='pdf')
# 保存图像为PNG
plt.savefig('D:\第一篇论文\实验代码\最新代码（完整版）\\实验1_MC与MMCA对比\\MC和MMCA对比图.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()