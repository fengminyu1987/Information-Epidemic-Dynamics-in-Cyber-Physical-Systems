import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import random
import os
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

# # --- 调整后的参数设置 ---
# initial_infected_ratio = 0.01
# num_nodes = 200
# lambda_info = 0.01  #原来是0.03
# lambda_star = 0.01
# delta = 0.05        #原来是0.05
# mu = 0.05
# alpha = 1.0
# eta = 1
# seed_value = 211
# random.seed(seed_value)
# np.random.seed(seed_value)

# --- 调整后的参数设置 ---
initial_infected_ratio = 0.01
num_nodes = 200
lambda_info = 0.01  #原来是0.03
lambda_star = 0.01
delta = 0.05        #原来是0.05
mu = 0.05
alpha = 1.0
eta = 1
seed_value = 211
random.seed(seed_value)
np.random.seed(seed_value)



# --- 构建连通3-均匀超图 ---
def build_connected_3uniform_hypergraph(N, num_hyperedges):
    hyperedges = []
    nodes_included = set()

    # Step 1: 初始化一个三元组
    triplet = random.sample(range(N), 3)
    hyperedges.append(set(triplet))
    nodes_included.update(triplet)

    # Step 2: 每个新超边包含至少一个已在图中的节点
    while len(hyperedges) < num_hyperedges:
        existing_node = random.choice(list(nodes_included))
        other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
        new_triplet = set([existing_node] + other_nodes)
        if new_triplet not in hyperedges:
            hyperedges.append(new_triplet)
            nodes_included.update(new_triplet)

        # 若仍未包含所有节点，则强制加入未覆盖节点
        if len(nodes_included) < N and len(hyperedges) > N // 3:
            uncovered = list(set(range(N)) - nodes_included)
            for u in uncovered:
                partner_nodes = random.sample(list(nodes_included), 2)
                new_triplet = set([u] + partner_nodes)
                if new_triplet not in hyperedges:
                    hyperedges.append(new_triplet)
                    nodes_included.update(new_triplet)

    return hyperedges

num_hyperedges = int(num_nodes * 2.5)
hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)

# --- 节点到超边的映射 ---
node_to_hyperedges = {i: [] for i in range(num_nodes)}
for idx, he in enumerate(hyperedges):
    for node in he:
        node_to_hyperedges[node].append(idx)

node_simplex_neighbors = defaultdict(list)
for he in hyperedges:
    for i in he:
        others = list(set(he) - {i})
        if len(others) == 2:
            node_simplex_neighbors[i].append(tuple(sorted(others)))

G_info = nx.Graph()
G_info.add_nodes_from(range(num_nodes))
p_pairwise = 0.5
for he in hyperedges:
    for u, v in itertools.combinations(he, 2):
        if random.random() < p_pairwise:
            G_info.add_edge(u, v)
a_matrix = nx.to_numpy_array(G_info)

G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
b_matrix = nx.to_numpy_array(G_epidemic)

W_ij_matrix = np.zeros((num_nodes, num_nodes))
W_i_vector = np.zeros(num_nodes)
for i in range(num_nodes):
    E_i = set(node_to_hyperedges[i])
    for j in range(num_nodes):
        if i == j:
            continue
        E_j = set(node_to_hyperedges[j])
        intersection = len(E_i & E_j)
        union = len(E_i | E_j)
        if union > 0:
            W_ij = (intersection / union) ** alpha
            W_ij_matrix[i, j] = W_ij
            W_i_vector[i] += W_ij

def get_beta_Aij_matrix(beta_U, eta):
    beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        W_i = W_i_vector[i]
        for j in range(num_nodes):
            if i == j or b_matrix[i, j] == 0:
                beta_Aij_matrix[i, j] = 0
                continue
            
            if W_i == 0:
                beta_Aij_matrix[i, j] = beta_U
            else:
                gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
                beta_Aij_matrix[i, j] = gamma_ij * beta_U
    return beta_Aij_matrix

def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
    r_i_2 = np.ones(num_nodes)
    for i in range(num_nodes):
        product = 1
        for (j, k) in node_simplex_neighbors[i]:
            c_ijk = P_A[j] * P_A[k] 
            product *= (1 - c_ijk * lambda_star)
        r_i_2[i] = product
    return r_i_2

def get_r_i_1(a_matrix, P_A, W_ij_matrix):
    result = np.ones(len(P_A))
    for i in range(len(P_A)):
        temp = 1.0
        for j in range(len(P_A)):
            if a_matrix[i, j] == 1:
                temp *= (1 - lambda_info * P_A[j] * W_ij_matrix[i, j])
        result[i] = temp
    return result

def get_q_i_U(b_matrix, P_AI, beta_U):
    return np.prod(1 - b_matrix * P_AI[:, None] * beta_U, axis=0)

def get_q_i_A(b_matrix, P_AI, beta_Aij_matrix):
    q_i = np.ones(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if b_matrix[j, i] == 1:
                P_j_AI = P_AI[j]
                q_i[i] *= (1 - P_j_AI * beta_Aij_matrix[i, j])
    return q_i

# --- 多次运行MMCA并平均 ---
def run_mmca_multiple_times(beta_U, eta, num_runs=20000, max_iterations=200):
    """多次运行MMCA取平均,增加曲线平滑度"""
    all_us_histories = []
    all_as_histories = []
    all_ai_histories = []
    
    for run in range(num_runs):
        # 每次运行使用不同的初始感染节点
        np.random.seed(seed_value + run * 100)
        random.seed(seed_value + run * 100)
        
        P_AI = np.zeros(num_nodes)
        P_AS = np.zeros(num_nodes)
        P_US = np.ones(num_nodes)
        initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
        P_AI[initial_infected] = 1
        P_US[initial_infected] = 0
        
        history_US = [np.mean(P_US)]
        history_AS = [np.mean(P_AS)]
        history_AI = [np.mean(P_AI)]
        
        tolerance = 1e-10
        
        for iteration in range(max_iterations):
            P_A = P_AI + P_AS
            
            r_i_1 = get_r_i_1(a_matrix, P_A, W_ij_matrix)
            r_i_2 = get_r_i_2(node_simplex_neighbors, P_A, lambda_star)
            r_i = r_i_1 * r_i_2
            
            q_i_U = get_q_i_U(b_matrix, P_AI, beta_U)
            beta_Aij_matrix = get_beta_Aij_matrix(beta_U, eta)
            q_i_A = get_q_i_A(b_matrix, P_AI, beta_Aij_matrix)
            
            P_US_next = P_AI * delta * mu + P_US * r_i * q_i_U + P_AS * delta * q_i_U
            P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_i_A + P_AS * (1 - delta) * q_i_A
            P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)) + \
                               P_AS * (delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A))
            
            total_sum = P_US_next + P_AS_next + P_AI_next
            total_sum[total_sum == 0] = 1
            P_US_next /= total_sum
            P_AS_next /= total_sum
            P_AI_next /= total_sum
            
            if np.max(np.abs(P_US_next - P_US)) < tolerance and \
               np.max(np.abs(P_AS_next - P_AS)) < tolerance and \
               np.max(np.abs(P_AI_next - P_AI)) < tolerance:
                for _fill in range(max_iterations - iteration):
                    history_US.append(np.mean(P_US_next))
                    history_AS.append(np.mean(P_AS_next))
                    history_AI.append(np.mean(P_AI_next))
                break
            
            P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next
            history_US.append(np.mean(P_US))
            history_AS.append(np.mean(P_AS))
            history_AI.append(np.mean(P_AI))
        
        while len(history_US) < max_iterations + 1:
            history_US.append(np.mean(P_US))
            history_AS.append(np.mean(P_AS))
            history_AI.append(np.mean(P_AI))
        
        all_us_histories.append(history_US)
        all_as_histories.append(history_AS)
        all_ai_histories.append(history_AI)
    
    # 平均所有运行结果
    avg_us_history = np.mean(all_us_histories, axis=0)
    avg_as_history = np.mean(all_as_histories, axis=0)
    avg_ai_history = np.mean(all_ai_histories, axis=0)
    
    return avg_us_history, avg_as_history, avg_ai_history

class ImprovedMCSimulation:
    def __init__(self, beta_U, eta):
        self.beta_U = beta_U
        self.eta = eta
        self.reset_states()
        
    def reset_states(self):
        self.states = np.zeros(num_nodes, dtype=int)
        initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
        self.states[initial_infected] = 2
        
    def get_adaptive_beta(self, i, j):
        if b_matrix[i, j] == 0:
            return 0
        if W_i_vector[i] == 0:
            return self.beta_U
        gamma_ij = (W_ij_matrix[i, j] / W_i_vector[i]) ** self.eta
        return gamma_ij * self.beta_U
    
    def calculate_transition_probabilities(self, node):
        r_i_1 = 1.0 
        for j in range(num_nodes):
            if a_matrix[node, j] == 1:
                P_j_A = 1 if self.states[j] in [1, 2] else 0
                r_i_1 *= (1 - lambda_info * P_j_A * W_ij_matrix[node, j])
        
        r_i_2 = 1.0 
        for (j, k) in node_simplex_neighbors[node]: 
            P_j_A = 1 if self.states[j] in [1, 2] else 0
            P_k_A = 1 if self.states[k] in [1, 2] else 0
            if P_j_A == 1 and P_k_A == 1:
                r_i_2 *= (1 - lambda_star)
        
        r_i = r_i_1 * r_i_2
        
        q_i_U = 1.0
        for j in range(num_nodes):
            if b_matrix[j, node] == 1:
                P_j_AI = 1 if self.states[j] == 2 else 0
                q_i_U *= (1 - P_j_AI * self.beta_U)
            
        q_i_A = 1.0
        for j in range(num_nodes):
            if b_matrix[j, node] == 1:
                P_j_AI = 1 if self.states[j] == 2 else 0
                adaptive_beta = self.get_adaptive_beta(node, j)
                q_i_A *= (1 - P_j_AI * adaptive_beta)
        
        return r_i, q_i_U, q_i_A
    
    def step(self):
        new_states = self.states.copy()
        next_states_candidates = [None] * num_nodes

        for i in range(num_nodes):
            current_state = self.states[i]
            r_i, q_i_U, q_i_A = self.calculate_transition_probabilities(i)
            
            rand1 = random.random()
            rand2 = random.random()

            if current_state == 0:  # US状态
                if rand1 < (1 - r_i):
                    if rand2 < (1 - q_i_A):
                        next_states_candidates[i] = 2
                    else:
                        next_states_candidates[i] = 1
                else:
                    if rand2 < (1 - q_i_U):
                        next_states_candidates[i] = 2
                    else:
                        next_states_candidates[i] = 0
                        
            elif current_state == 1:  # AS状态
                if rand1 < delta:
                    if rand2 < (1 - q_i_U):
                        next_states_candidates[i] = 2
                    else:
                        next_states_candidates[i] = 0
                else:
                    if rand2 < (1 - q_i_A):
                        next_states_candidates[i] = 2
                    else:
                        next_states_candidates[i] = 1
                        
            elif current_state == 2:  # AI状态
                if rand1 < mu:
                    if rand2 < delta:
                        next_states_candidates[i] = 0
                    else:
                        next_states_candidates[i] = 1
                else:
                    next_states_candidates[i] = 2
        
        self.states = np.array(next_states_candidates)
    
    def run_simulation(self, max_steps=100):
        history_US = []
        history_AS = []
        history_AI = []
        
        for step in range(max_steps + 1):
            infected_count = np.sum(self.states == 2)
            aware_susceptible_count = np.sum(self.states == 1)
            unaware_susceptible_count = np.sum(self.states == 0)
            
            history_AI.append(infected_count / num_nodes)
            history_AS.append(aware_susceptible_count / num_nodes)
            history_US.append(unaware_susceptible_count / num_nodes)
            
            if step < max_steps:
                self.step()
        
        return history_US, history_AS, history_AI

def improved_mc_simulation_time_series(beta_U, eta, num_runs=2000):
    max_steps_mc = 200

    def run_single_mc(run_seed):
        random.seed(run_seed)
        np.random.seed(run_seed)
        sim = ImprovedMCSimulation(beta_U, eta)
        return sim.run_simulation(max_steps=max_steps_mc)

    results = Parallel(n_jobs=-1)(
        delayed(run_single_mc)(seed_value + run * 1000) for run in tqdm(range(num_runs), desc="MC模拟进行中")
    )
    
    all_us_histories = [res[0] for res in results]
    all_as_histories = [res[1] for res in results]
    all_ai_histories = [res[2] for res in results]
    
    avg_us_history = np.mean(all_us_histories, axis=0)
    avg_as_history = np.mean(all_as_histories, axis=0)
    avg_ai_history = np.mean(all_ai_histories, axis=0)
    
    return avg_us_history, avg_as_history, avg_ai_history

# --- 主要模拟 ---
selected_beta_U = 0.5

print(f"MMCA多次运行模拟 (beta_U={selected_beta_U}) 时间序列...")
mmca_us_history, mmca_as_history, mmca_ai_history = run_mmca_multiple_times(selected_beta_U, eta, num_runs=100, max_iterations=300)
time_steps_mmca = range(len(mmca_us_history))

print(f"MC模拟 (beta_U={selected_beta_U}) 时间序列...")
mc_us_history, mc_as_history, mc_ai_history = improved_mc_simulation_time_series(selected_beta_U, eta)
time_steps_mc = range(len(mc_us_history))


# 设置默认字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
# mpl.rcParams['mathtext.default'] = 'rm'  # 默认使用罗马体渲染数学


# --- 绘图 ---
plt.figure(figsize=(10, 6))

# # MMCA 图形（截取前51个点，即0到50步）
# plt.plot(time_steps_mmca[:51], mmca_ai_history[:51], color='red', linestyle='-', linewidth=2, label='MMCA: AI')
# plt.plot(time_steps_mmca[:51], mmca_as_history[:51], color='blue', linestyle='-', linewidth=2, label='MMCA: AS')
# plt.plot(time_steps_mmca[:51], mmca_us_history[:51], color='green', linestyle='-', linewidth=2, label='MMCA: US')

# # MC 图形（截取前51个点，即0到50步）
# plt.plot(time_steps_mc[:51], mc_ai_history[:51], color='red', linestyle='None', marker='D', fillstyle='none', markeredgecolor='red', markersize=10, label='MC: AI')
# plt.plot(time_steps_mc[:51], mc_as_history[:51], color='blue', linestyle='None', marker='o', fillstyle='none', markeredgecolor='blue', markersize=10, label='MC: AS')
# plt.plot(time_steps_mc[:51], mc_us_history[:51], color='green', linestyle='None', marker='s', fillstyle='none', markeredgecolor='green', markersize=10, label='MC: US')

# MMCA 图形（截取前51个点，即0到50步）- 使用更深的颜色和更粗的线条
plt.plot(time_steps_mmca[:51], mmca_ai_history[:51], color='red', linestyle='-', 
         linewidth=2, label='MMCA: AI')
plt.plot(time_steps_mmca[:51], mmca_as_history[:51], color='blue', linestyle='-', 
         linewidth=2, label='MMCA: AS')
plt.plot(time_steps_mmca[:51], mmca_us_history[:51], color='green', linestyle='-', 
         linewidth=2, label='MMCA: US')

# MC 图形（截取前51个点，即0到50步）- 增加标记边框宽度以匹配颜色深度
plt.plot(time_steps_mc[:51], mc_ai_history[:51], color='red', linestyle='None', 
         marker='D', fillstyle='none', markeredgecolor='red', markersize=10, 
         markeredgewidth=2, label='MC: AI')
plt.plot(time_steps_mc[:51], mc_as_history[:51], color='blue', linestyle='None', 
         marker='o', fillstyle='none', markeredgecolor='blue', markersize=10, 
         markeredgewidth=2, label='MC: AS')
plt.plot(time_steps_mc[:51], mc_us_history[:51], color='green', linestyle='None', 
         marker='s', fillstyle='none', markeredgecolor='green', markersize=10, 
         markeredgewidth=2, label='MC: US')


# # --- 绘图 ---
# plt.figure(figsize=(10, 6))

# # MMCA 图形（截取前51个点，即0到50步）
# plt.plot(time_steps_mmca[:51], mmca_ai_history[:51], color='darkred', linestyle='-', linewidth=2, label='MMCA: AI')
# plt.plot(time_steps_mmca[:51], mmca_as_history[:51], color='darkblue', linestyle='-', linewidth=2, label='MMCA: AS')
# plt.plot(time_steps_mmca[:51], mmca_us_history[:51], color='darkgreen', linestyle='-', linewidth=2, label='MMCA: US')

# # MC 图形（截取前51个点，即0到50步）
# plt.plot(time_steps_mc[:51], mc_ai_history[:51], color='darkred', linestyle='None', marker='D', fillstyle='none', markersize=10, label='MC: AI')
# plt.plot(time_steps_mc[:51], mc_as_history[:51], color='darkblue', linestyle='None', marker='o', fillstyle='none', markersize=10, label='MC: AS')
# plt.plot(time_steps_mc[:51], mc_us_history[:51], color='darkgreen', linestyle='None', marker='s', fillstyle='none', markersize=10, label='MC: US')


# plt.title(f'AI, AS, US 比例随时间变化')
plt.xlabel('Time Step', fontsize=21)
plt.ylabel('Proportion', fontsize=21)
plt.xlim(0, 50)  # 改为50步
plt.xticks(np.arange(0, 51, 10), fontsize=19)  # x轴刻度改为0, 10, 20, ..., 50
plt.yticks(fontsize=19)  # 增大y轴刻度字体大小
plt.grid(True, alpha=0.8)
plt.legend(fontsize=16)
plt.tight_layout()

output_dir = r'D:\第一篇论文\实验代码\最新代码（完整版）\实验2_比例随时间变化图'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, '比例随时间变化图.pdf'), format='pdf')
plt.savefig(os.path.join(output_dir, '比例随时间变化图.png'), dpi=300, bbox_inches='tight')
plt.show()

def calculate_steady_state_error(mmca_history, mc_history, stabilization_window=30):
    if len(mmca_history) < stabilization_window or len(mc_history) < stabilization_window:
        mmca_steady = mmca_history[-1] if mmca_history else 0
        mc_steady = mc_history[-1] if mc_history else 0
    else:
        mmca_steady = np.mean(mmca_history[-stabilization_window:])
        mc_steady = np.mean(mc_history[-stabilization_window:])
    
    error = np.abs(mmca_steady - mc_steady)
    percentage_error = (error / mmca_steady) * 100 if mmca_steady != 0 else 0
    return mmca_steady, mc_steady, error, percentage_error

print("\n--- 稳态误差分析 ---")


window_size = 30
mmca_ai_steady, mc_ai_steady, ai_error, ai_percentage_error = calculate_steady_state_error(mmca_ai_history, mc_ai_history, window_size)
mmca_us_steady, mc_us_steady, us_error, us_percentage_error = calculate_steady_state_error(mmca_us_history, mc_us_history, window_size)
mmca_as_steady, mc_as_steady, as_error, as_percentage_error = calculate_steady_state_error(mmca_as_history, mc_as_history, window_size)

print(f"AI: MMCA 稳态 = {mmca_ai_steady:.4f}, MC 稳态 = {mc_ai_steady:.4f}, 绝对误差 = {ai_error:.4f}, 百分比误差 = {ai_percentage_error:.2f}%")
print(f"US: MMCA 稳态 = {mmca_us_steady:.4f}, MC 稳态 = {mc_us_steady:.4f}, 绝对误差 = {us_error:.4f}, 百分比误差 = {us_percentage_error:.2f}%")
print(f"AS: MMCA 稳态 = {mmca_as_steady:.4f}, MC 稳态 = {mc_as_steady:.4f}, 绝对误差 = {as_error:.4f}, 百分比误差 = {as_percentage_error:.2f}%")