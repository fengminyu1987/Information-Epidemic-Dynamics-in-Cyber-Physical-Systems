# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import random
# import os
# import pandas as pd
# from collections import defaultdict
# import itertools
# from tqdm import tqdm
# from joblib import Parallel, delayed

# # --- 参数设置 ---
# initial_infected_ratio = 0.01
# num_nodes = 200
# lambda_info = 0.1
# lambda_star = 0.1
# delta = 0.8
# mu = 0.4
# alpha = 1.0
# eta = 1.0
# seed_value = 211
# random.seed(seed_value)
# np.random.seed(seed_value)

# # --- 构建连通3-均匀超图 ---
# def build_connected_3uniform_hypergraph(N, num_hyperedges):
#     hyperedges = []
#     nodes_included = set()

#     triplet = random.sample(range(N), 3)
#     hyperedges.append(set(triplet))
#     nodes_included.update(triplet)

#     while len(hyperedges) < num_hyperedges:
#         existing_node = random.choice(list(nodes_included))
#         other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
#         new_triplet = set([existing_node] + other_nodes)
#         if new_triplet not in hyperedges:
#             hyperedges.append(new_triplet)
#             nodes_included.update(new_triplet)

#         if len(nodes_included) < N and len(hyperedges) > N // 3:
#             uncovered = list(set(range(N)) - nodes_included)
#             for u in uncovered:
#                 partner_nodes = random.sample(list(nodes_included), 2)
#                 new_triplet = set([u] + partner_nodes)
#                 if new_triplet not in hyperedges:
#                     hyperedges.append(new_triplet)
#                     nodes_included.update(new_triplet)

#     return hyperedges

# num_hyperedges = int(num_nodes * 2.5)
# hyperedges = build_connected_3uniform_hypergraph(num_nodes, num_hyperedges)

# # --- 节点到超边的映射 ---
# node_to_hyperedges = {i: [] for i in range(num_nodes)}
# for idx, he in enumerate(hyperedges):
#     for node in he:
#         node_to_hyperedges[node].append(idx)

# # --- 构建高阶传播结构 ---
# node_simplex_neighbors = defaultdict(list)
# for he in hyperedges:
#     for i in he:
#         others = list(set(he) - {i})
#         if len(others) == 2:
#             node_simplex_neighbors[i].append((others[0], others[1]))

# # --- 普通信息图 ---
# G_info = nx.Graph()
# G_info.add_nodes_from(range(num_nodes))
# p_pairwise = 0.5
# for he in hyperedges:
#     for u, v in itertools.combinations(he, 2):
#         if random.random() < p_pairwise:
#             G_info.add_edge(u, v)
# a_matrix = nx.to_numpy_array(G_info)

# # --- 物理层传播图 ---
# G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
# b_matrix = nx.to_numpy_array(G_epidemic)

# # --- Jaccard 权重矩阵 ---
# W_ij_matrix = np.zeros((num_nodes, num_nodes))
# W_i_vector = np.zeros(num_nodes)
# for i in range(num_nodes):
#     E_i = set(node_to_hyperedges[i])
#     for j in range(num_nodes):
#         if i == j:
#             continue
#         E_j = set(node_to_hyperedges[j])
#         intersection = len(E_i & E_j)
#         union = len(E_i | E_j)
#         if union > 0:
#             W_ij = (intersection / union) ** alpha
#             W_ij_matrix[i, j] = W_ij
#             W_i_vector[i] += W_ij

# # --- 改进的Monte Carlo模拟类 ---
# class ImprovedMCSimulation:
#     def __init__(self, beta_U, eta):
#         self.beta_U = beta_U
#         self.eta = eta
#         self.reset_states()
        
#     def reset_states(self):
#         # 状态: 0=US, 1=AS, 2=AI
#         self.states = np.zeros(num_nodes, dtype=int)
#         # 初始感染
#         initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
#         self.states[initial_infected] = 2  # AI
        
#     def get_adaptive_beta(self, i, j):
#         """计算节点i在A状态下被节点j感染的自适应传播概率"""
#         if W_ij_matrix[i, j] or W_i_vector[i] == 0:
#             return self.beta_U
#         gamma_ij = (W_ij_matrix[i, j] / W_i_vector[i]) ** self.eta
#         return gamma_ij * self.beta_U
    
#     def calculate_transition_probabilities(self, node):
#         """计算节点的各种转换概率,与MMCA完全一致"""
#         current_state = self.states[node]
        
#         # 计算信息传播概率 (1 - r_i)
#         # r_i_1: 普通信息传播
#         r_i_1 = 1.0
#         for j in range(num_nodes):
#             if a_matrix[node, j] == 1:
#                 P_j_A = 1 if self.states[j] in [1, 2] else 0
#                 r_i_1 *= (1 - lambda_info * P_j_A * W_ij_matrix[node, j])
        
#         # r_i_2: 高阶信息传播
#         r_i_2 = 1.0
#         for (j, k) in node_simplex_neighbors[node]:
#             P_j_A = 1 if self.states[j] in [1, 2] else 0
#             P_k_A = 1 if self.states[k] in [1, 2] else 0
#             r_i_2 *= (1 - P_j_A * P_k_A * lambda_star)
        
#         r_i = r_i_1 * r_i_2
#         #info_transmission_prob = 1 - r_i
        
#         # 计算疫情传播概率
#         #if current_state == 0:  # US状态
#         q_i_U = 1.0
#         for j in range(num_nodes):
#             if b_matrix[j, node] == 1:
#                 P_j_AI = 1 if self.states[j] ==2 else 0
#                 q_i_U *= (1 - P_j_AI * self.beta_U)
#         #epidemic_prob = 1 - q_i_U
#         #elif current_state == 1:  # AS状态
         
#         q_i_A = 1.0
#         for j in range(num_nodes):
#             if b_matrix[j, node] == 1:
#                 P_j_AI = 1 if self.states[j] ==2 else 0
#                 adaptive_beta = self.get_adaptive_beta(node, j)
#                 q_i_A *= (1 - P_j_AI * adaptive_beta)
#         #epidemic_prob = 1 - q_i_A
        
#         """ else:  # AI状态
#         epidemic_prob = 0.0 """
        
#         return r_i, q_i_U, q_i_A
    
#     def step(self):
#         """改进的单步更新,严格按照MMCA状态转换逻辑"""
#         new_states = self.states.copy()
        
#         for i in range(num_nodes):
#             current_state = self.states[i]
#             r_i, q_i_U, q_i_A = self.calculate_transition_probabilities(i)
            
#             # 按照MMCA的状态转换概率进行更新
#             if current_state == 0:  # US状态
#                 # 计算转换到各状态的概率
#                 # P_US_next = r_i * q_i_U
#                 # P_AS_next = (1 - r_i) * q_i_A  
#                 # P_AI_next = (1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)
                
#                 #q_i_U = 1 - epidemic_prob  # 不被感染的概率
#                 #q_i_A = q_i_U  # US状态下，被感知后的感染抵抗能力与A状态相同
                
#                 # 使用随机数决定状态转换
#                 rand = random.random()
                
#                 # 首先决定是否被信息感知
#                 if random.random() < (1 - r_i):  # 被信息感知
#                     new_states[i] = 1  # AS
#                     # 再决定是否被感染（使用A状态的感染概率）
#                     if random.random() < (1 - q_i_A):
#                         new_states[i] = 2  # AI
#                     else:
#                         new_states[i] = 1  # AS
#                 else:  # 未被信息感知
#                     new_states[i] = 0  # US
#                     # 直接决定是否被感染（使用U状态的感染概率）
#                     if random.random() < (1 - q_i_U):
#                         new_states[i] = 2  # AI
#                     else:
#                         new_states[i] = 0  # US
                        
#             elif current_state == 1:  # AS状态
#                 # P_US_next = delta * q_i_U
#                 # P_AS_next = (1 - delta) * q_i_A
#                 # P_AI_next = delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A)
                
#                 """ q_i_A = 1 - epidemic_prob
#                 q_i_U = 1.0  # 如果变回US,假设感染抵抗能力为原始值
#                 for j in range(num_nodes):
#                     if b_matrix[j, i] == 1 and self.states[j] == 2:
#                         q_i_U *= (1 - self.beta_U) """
                
#                 # 首先决定是否失去感知
#                 if random.random() < delta:  # 失去感知
#                     new_states[i] = 0  # US
#                     if random.random() < (1 - q_i_U):
#                         new_states[i] = 2  # AI
#                     else:
#                         new_states[i] = 0  # US
#                 else:  # 保持感知
#                     new_states[i] = 1  # AS
#                     if random.random() < (1 - q_i_A):
#                         new_states[i] = 2  # AI
#                     else:
#                         new_states[i] = 1  # AS
                        
#             elif current_state == 2:  # AI状态
#                 # P_US_next = delta * mu
#                 # P_AS_next = (1 - delta) * mu
#                 # P_AI_next = 1 - mu
                
#                 if random.random() < delta:  # 恢复
#                     #new_states[i] = 'UI'  
#                     if random.random() < mu:
#                         new_states[i] = 0  # US
#                     else:
#                         new_states[i] = 2  # AI
#                 else:
#                     new_states[i] = 2  # AI
#                     if random.random() < mu:
#                         new_states[i] = 1  # AS
#                     else:
#                         new_states[i] = 2  # AI
        
#         self.states = new_states
    
#     def run_simulation(self, max_steps=3000, convergence_steps=100):
#         """运行模拟直到收敛，增加步数以确保稳定"""
#         infection_history = []
#         awareness_history = []
        
#         for step in range(max_steps):
#             # 记录当前状态
#             infected_count = np.sum(self.states == 2) # AI
#             aware_count = np.sum(self.states >= 1)  # AS + AI
            
#             infection_history.append(infected_count / num_nodes)
#             awareness_history.append(aware_count / num_nodes)
            
#             # 检查收敛
#             if step >= convergence_steps:
#                 recent_infection = infection_history[-convergence_steps:]
#                 recent_awareness = awareness_history[-convergence_steps:]
                
#                 if (np.std(recent_infection) < 0.001 and 
#                     np.std(recent_awareness) < 0.001):
#                     break
            
#             # 更新状态
#             self.step()
        
#         # 返回最后的稳态值
#         final_infection = np.mean(infection_history[-100:])
#         final_awareness = np.mean(awareness_history[-100:])
        
#         return final_infection, final_awareness

# # --- 改进的MC模拟主函数 ---
# def improved_mc_simulation(beta_U, eta, num_runs=200):
#     """增加模拟次数以提高精度"""
#     infection_results = []
#     awareness_results = []
    
#     for run in range(num_runs):
#         # 为每次运行设置不同的随机种子
#         random.seed(seed_value + run * 1000)
#         np.random.seed(seed_value + run * 1000)
        
#         sim = ImprovedMCSimulation(beta_U, eta)
#         infection_density, awareness_density = sim.run_simulation()
        
#         infection_results.append(infection_density)
#         awareness_results.append(awareness_density)
    
#     return np.mean(infection_results), np.mean(awareness_results)

# # --- 扫描 β_U 参数 ---
# beta_U_values = np.linspace(0, 1, 51)
# improved_infection_densities = []
# improved_awareness_densities = []

# print("开始改进的Monte Carlo模拟...")
# results = Parallel(n_jobs=-1, backend='loky')(
#     delayed(improved_mc_simulation)(beta_U, eta)
#     for beta_U in tqdm(beta_U_values, desc="Improved MC Simulating")
# )
# improved_infection_densities, improved_awareness_densities = zip(*results)

# """ # --- 保存结果 ---
# df_improved = pd.DataFrame({
#     'beta_U': beta_U_values,
#     'infection_density': improved_infection_densities,
#     'awareness_density': improved_awareness_densities
# })
# df_improved.to_csv('MC0609并行版.csv', index=False) """

# # --- 保存结果 ---
# df = pd.DataFrame({
#     'beta_U': beta_U_values,
#     'infection_density': improved_infection_densities,
#     'awareness_density': improved_awareness_densities
# })
# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, 'MC0609并行版.csv')

# # 保存文件
# df.to_csv(file_path, index=False)

# # --- 绘图 ---
# plt.figure(figsize=(10, 6))
# plt.plot(beta_U_values, improved_awareness_densities, color='blue', linestyle='--', linewidth=2, label='MC:Aware Nodes Ratio')
# plt.plot(beta_U_values, improved_infection_densities, color='red', marker='o', markersize=4, linewidth=2, label='MC:Infected Nodes Ratio')
# plt.title('Improved MC: Awareness & Infection Ratio vs. β')
# plt.xlabel('β')
# plt.ylabel('Ratio')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

# --- 参数设置 ---
initial_infected_ratio = 0.01
num_nodes = 200
lambda_info = 0.1
lambda_star = 0.1
delta = 0.8
mu = 0.4
alpha = 1.0
eta = 1.0
seed_value = 211
random.seed(seed_value)
np.random.seed(seed_value)

# --- 构建连通3-均匀超图 ---
def build_connected_3uniform_hypergraph(N, num_hyperedges):
    hyperedges = []
    nodes_included = set()

    triplet = random.sample(range(N), 3)
    hyperedges.append(set(triplet))
    nodes_included.update(triplet)

    while len(hyperedges) < num_hyperedges:
        existing_node = random.choice(list(nodes_included))
        other_nodes = random.sample([n for n in range(N) if n != existing_node], 2)
        new_triplet = set([existing_node] + other_nodes)
        if new_triplet not in hyperedges:
            hyperedges.append(new_triplet)
            nodes_included.update(new_triplet)

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

# --- 构建高阶传播结构 ---
node_simplex_neighbors = defaultdict(list)
for he in hyperedges:
    for i in he:
        others = list(set(he) - {i})
        if len(others) == 2:
            node_simplex_neighbors[i].append((others[0], others[1]))

# --- 普通信息图 ---
G_info = nx.Graph()
G_info.add_nodes_from(range(num_nodes))
p_pairwise = 0.5
for he in hyperedges:
    for u, v in itertools.combinations(he, 2):
        if random.random() < p_pairwise:
            G_info.add_edge(u, v)
a_matrix = nx.to_numpy_array(G_info)

# --- 物理层传播图 ---
G_epidemic = nx.watts_strogatz_graph(num_nodes, 4, 0.5, seed=2)
b_matrix = nx.to_numpy_array(G_epidemic)

# --- Jaccard 权重矩阵 ---
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

# --- QS方法的Monte Carlo模拟类 ---
class QSMCSimulation:
    def __init__(self, beta_U, eta):
        self.beta_U = beta_U
        self.eta = eta
        self.reset_states()
        
    def reset_states(self):
        # 状态: 0=US, 1=AS, 2=AI
        self.states = np.zeros(num_nodes, dtype=int)
        # 初始感染
        initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
        self.states[initial_infected] = 2  # AI
        
    def get_adaptive_beta(self, i, j):
        """计算节点i在A状态下被节点j感染的自适应传播概率"""
        if W_i_vector[i] == 0:
            return self.beta_U
        gamma_ij = (W_ij_matrix[i, j] / W_i_vector[i]) ** self.eta
        return gamma_ij * self.beta_U
    
    def calculate_info_transmission_prob(self, node):
        """计算信息传播概率 (1 - r_i)"""
        # r_i_1: 普通信息传播
        r_i_1 = 1.0
        for j in range(num_nodes):
            if a_matrix[node, j] == 1:
                P_j_A = 1 if self.states[j] in [1, 2] else 0
                r_i_1 *= (1 - lambda_info * P_j_A * W_ij_matrix[node, j])
        
        # r_i_2: 高阶信息传播
        r_i_2 = 1.0
        for (j, k) in node_simplex_neighbors[node]:
            P_j_A = 1 if self.states[j] in [1, 2] else 0
            P_k_A = 1 if self.states[k] in [1, 2] else 0
            r_i_2 *= (1 - P_j_A * P_k_A * lambda_star)
        
        r_i = r_i_1 * r_i_2
        return 1 - r_i
    
    def calculate_epidemic_transmission_prob(self, node, state_type='U'):
        """计算疫情传播概率"""
        q_i = 1.0
        for j in range(num_nodes):
            if b_matrix[j, node] == 1:
                P_j_AI = 1 if self.states[j] == 2 else 0
                if state_type == 'U':
                    q_i *= (1 - P_j_AI * self.beta_U)
                else:  # state_type == 'A'
                    adaptive_beta = self.get_adaptive_beta(node, j)
                    q_i *= (1 - P_j_AI * adaptive_beta)
        
        return 1 - q_i
    
    def qs_info_propagation_step(self):
        """QS方法：信息传播的准稳态步骤"""
        # 在疫情传播之前，先让信息传播达到准稳态
        max_info_steps = 50  # 限制信息传播的最大步数
        info_convergence_threshold = 0.001
        
        for info_step in range(max_info_steps):
            old_aware_count = np.sum(self.states >= 1)
            new_states = self.states.copy()
            
            # 只处理信息传播，不处理疫情传播和恢复
            for i in range(num_nodes):
                if self.states[i] == 0:  # US状态
                    info_prob = self.calculate_info_transmission_prob(i)
                    if random.random() < info_prob:
                        new_states[i] = 1  # US -> AS
                elif self.states[i] == 1:  # AS状态
                    if random.random() < delta:  # 失去感知
                        new_states[i] = 0  # AS -> US
            
            self.states = new_states
            
            # 检查信息传播是否收敛
            new_aware_count = np.sum(self.states >= 1)
            if abs(new_aware_count - old_aware_count) < info_convergence_threshold * num_nodes:
                break
    
    def epidemic_step(self):
        """疫情传播步骤"""
        new_states = self.states.copy()
        
        for i in range(num_nodes):
            current_state = self.states[i]
            
            if current_state == 0:  # US状态
                epidemic_prob = self.calculate_epidemic_transmission_prob(i, 'U')
                if random.random() < epidemic_prob:
                    new_states[i] = 2  # US -> AI
                    
            elif current_state == 1:  # AS状态
                epidemic_prob = self.calculate_epidemic_transmission_prob(i, 'A')
                if random.random() < epidemic_prob:
                    new_states[i] = 2  # AS -> AI
                    
            elif current_state == 2:  # AI状态
                if random.random() < mu:  # 恢复
                    if random.random() < delta:
                        new_states[i] = 0  # AI -> US
                    else:
                        new_states[i] = 1  # AI -> AS
        
        self.states = new_states
    
    def qs_step(self):
        """QS方法的完整步骤：先信息传播准稳态，再疫情传播"""
        # 步骤1：信息传播达到准稳态
        self.qs_info_propagation_step()
        
        # 步骤2：疫情传播
        self.epidemic_step()
    
    def run_simulation(self, max_steps=3000, convergence_steps=100):
        """运行QS方法的模拟直到收敛"""
        infection_history = []
        awareness_history = []
        
        for step in range(max_steps):
            # 记录当前状态
            infected_count = np.sum(self.states == 2)  # AI
            aware_count = np.sum(self.states >= 1)     # AS + AI
            
            infection_history.append(infected_count / num_nodes)
            awareness_history.append(aware_count / num_nodes)
            
            # 检查收敛
            if step >= convergence_steps:
                recent_infection = infection_history[-convergence_steps:]
                recent_awareness = awareness_history[-convergence_steps:]
                
                if (np.std(recent_infection) < 0.001 and 
                    np.std(recent_awareness) < 0.001):
                    break
            
            # QS方法的状态更新
            self.qs_step()
        
        # 返回最后的稳态值
        final_infection = np.mean(infection_history[-100:])
        final_awareness = np.mean(awareness_history[-100:])
        
        return final_infection, final_awareness

# --- QS方法的MC模拟主函数 ---
def qs_mc_simulation(beta_U, eta, num_runs=200):
    """QS方法的Monte Carlo模拟"""
    infection_results = []
    awareness_results = []
    
    for run in range(num_runs):
        # 为每次运行设置不同的随机种子
        random.seed(seed_value + run * 1000)
        np.random.seed(seed_value + run * 1000)
        
        sim = QSMCSimulation(beta_U, eta)
        infection_density, awareness_density = sim.run_simulation()
        
        infection_results.append(infection_density)
        awareness_results.append(awareness_density)
    
    return np.mean(infection_results), np.mean(awareness_results)

# --- 扫描 β_U 参数 ---
beta_U_values = np.linspace(0, 1, 51)
qs_infection_densities = []
qs_awareness_densities = []

print("开始QS方法的Monte Carlo模拟...")
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(qs_mc_simulation)(beta_U, eta)
    for beta_U in tqdm(beta_U_values, desc="QS-MC Simulating")
)
qs_infection_densities, qs_awareness_densities = zip(*results)

# --- 保存结果 ---
df = pd.DataFrame({
    'beta_U': beta_U_values,
    'infection_density': qs_infection_densities,
    'awareness_density': qs_awareness_densities
})

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'MC0609并行版_QS方法.csv')

# 保存文件
df.to_csv(file_path, index=False)
print(f"结果已保存到: {file_path}")

# # --- 绘图 ---
# plt.figure(figsize=(12, 8))

# # 子图1: 感染密度
# plt.subplot(2, 1, 1)
# plt.plot(beta_U_values, qs_infection_densities, color='red', marker='o', markersize=3, 
#          linewidth=2, label='QS-MC: Infected Nodes Ratio')
# plt.title('QS Method: Infection Density vs. β_U')
# plt.xlabel('β_U')
# plt.ylabel('Infection Density')
# plt.grid(True, alpha=0.3)
# plt.legend()

# # 子图2: 感知密度
# plt.subplot(2, 1, 2)
# plt.plot(beta_U_values, qs_awareness_densities, color='blue', linestyle='--', 
#          linewidth=2, label='QS-MC: Aware Nodes Ratio')
# plt.title('QS Method: Awareness Density vs. β_U')
# plt.xlabel('β_U')
# plt.ylabel('Awareness Density')
# plt.grid(True, alpha=0.3)
# plt.legend()

# plt.tight_layout()
# plt.show()

# --- 绘图 ---
plt.figure(figsize=(10, 6))
plt.plot(beta_U_values, qs_awareness_densities, color='blue', linestyle='--', linewidth=2, label='MC_QS:Aware Nodes Ratio')
plt.plot(beta_U_values, qs_infection_densities, color='red', marker='o', markersize=4, linewidth=2, label='MC_QS:Infected Nodes Ratio')
plt.xlabel('β')
plt.ylabel('Ratio')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()