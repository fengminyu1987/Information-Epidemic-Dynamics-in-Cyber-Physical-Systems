import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import os
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm

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

# --- 构建高阶传播结构 node_simplex_neighbors ---
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

""" # --- Jaccard 权重矩阵 ---
W_ij_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    E_i = set(node_to_hyperedges[i])
    for j in range(num_nodes):
        if i == j:
            continue
        E_j = set(node_to_hyperedges[j])
        intersection = len(E_i & E_j)
        union = len(E_i | E_j)
        if union > 0:
            W_ij = intersection / union
            W_ij_matrix[i, j] = W_ij
            
# --- 修改后的W_i_vector计算：只考虑同一超边内的邻居 ---
W_i_vector = np.zeros(num_nodes)
for i in range(num_nodes):
    # 获取与节点i处于同一超边的所有邻居节点
    hyperedge_neighbors = set()
    for he in hyperedges:
        if i in he:
            # 将该超边中除了i之外的所有节点加入邻居集合
            hyperedge_neighbors.update(he - {i})
    
    # 计算与这些超边邻居的Jaccard权重之和
    for neighbor in hyperedge_neighbors:
        W_i_vector[i] += W_ij_matrix[i, neighbor] """
        
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


# --- 自适应传播概率矩阵 ---
def get_beta_Aij_matrix(beta_U, eta):
    beta_Aij_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        W_i = W_i_vector[i]
        for j in range(num_nodes):
            if i == j or W_i == 0:
                continue
            gamma_ij = (W_ij_matrix[i, j] / W_i) ** eta
            beta_Aij_matrix[i, j] = gamma_ij * beta_U
    return beta_Aij_matrix

# --- 高阶传播 λ* ---
def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
    r_i_2 = np.ones(num_nodes)
    for i in range(num_nodes):
        product = 1
        for (j, k) in node_simplex_neighbors[i]:
            c_ijk = P_A[j] * P_A[k]
            product *= (1 - c_ijk * lambda_star)
        r_i_2[i] = product
    return r_i_2

""" def get_r_i_2(node_simplex_neighbors, P_A, lambda_star):
    r_i_2 = np.ones(num_nodes)
    for i in range(num_nodes):
        product = 1
        for (j, k) in node_simplex_neighbors[i]:
            c_ijk = 1 - (1 - P_A[j]) * (1 - P_A[k])  # 至少一个A节点的概率
            product *= (1 - c_ijk * lambda_star)
        r_i_2[i] = product
    return r_i_2 """


# --- 普通信息传播 ---
""" def get_r_i_1(a_matrix, P_A):
    return np.prod(1 - a_matrix * P_A[:, None] * lambda_info, axis=0) """

def get_r_i_1(a_matrix, P_A, W_ij_matrix):
    result = np.ones(len(P_A))
    for i in range(len(P_A)):
        temp = 1.0
        for j in range(len(P_A)):
            if a_matrix[i, j]:
                temp *= (1 - lambda_info * P_A[j] * W_ij_matrix[i, j])
        result[i] = temp
    return result


# --- 疫情传播（在U状态时不被其邻居感染的概率）---
def get_q_i_U(b_matrix, P_AI, beta):
    return np.prod(1 - b_matrix * P_AI[:, None] * beta, axis=0)

# --- 疫情传播（在A状态时不被其邻居感染的概率）---
def get_q_i_A(b_matrix, P_AI, beta_Aij_matrix):
    q_i = np.ones(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if b_matrix[j, i] == 1:
                q_i[i] *= (1 - P_AI[j] * beta_Aij_matrix[i, j])
    return q_i

# --- MMCA 迭代主体 ---
def iterate_probabilities_adaptive_beta(beta_U, eta):
    P_AI = np.zeros(num_nodes)
    P_AS = np.zeros(num_nodes)
    P_US = np.ones(num_nodes)
    initial_infected = np.random.choice(num_nodes, int(num_nodes * initial_infected_ratio), replace=False)
    P_AI[initial_infected] = 1
    P_US[initial_infected] = 0

    max_iterations = 10000
    tolerance = 1e-10

    for _ in range(max_iterations):
        P_A = P_AI + P_AS
        r_i = get_r_i_1(a_matrix, P_A, W_ij_matrix) * get_r_i_2(node_simplex_neighbors, P_A, lambda_star)


        q_i_U = get_q_i_U(b_matrix, P_AI, beta_U)
        beta_Aij_matrix = get_beta_Aij_matrix(beta_U, eta)
        q_i_A = get_q_i_A(b_matrix, P_AI, beta_Aij_matrix)

        P_US_next = P_AI * delta * mu + P_US * r_i * q_i_U + P_AS * delta * q_i_U
        P_AS_next = P_AI * (1 - delta) * mu + P_US * (1 - r_i) * q_i_A + P_AS * (1 - delta) * q_i_A
        P_AI_next = P_AI * (1 - mu) + P_US * ((1 - r_i) * (1 - q_i_A) + r_i * (1 - q_i_U)) + \
                    P_AS * (delta * (1 - q_i_U) + (1 - delta) * (1 - q_i_A))

        if np.max(np.abs(P_US_next - P_US)) < tolerance and \
           np.max(np.abs(P_AS_next - P_AS)) < tolerance and \
           np.max(np.abs(P_AI_next - P_AI)) < tolerance:
            break

        P_US, P_AS, P_AI = P_US_next, P_AS_next, P_AI_next

    infection_density = np.mean(P_AI)
    awareness_density = np.mean(P_A)
    return infection_density, awareness_density

# --- 扫描 β_U 参数 ---
beta_U_values = np.linspace(0, 1, 51)
infection_densities = []
awareness_densities = []

for beta_U in tqdm(beta_U_values, desc="Simulating"):
    infection_density, awareness_density = iterate_probabilities_adaptive_beta(beta_U, eta)
    infection_densities.append(infection_density)
    awareness_densities.append(awareness_density)

# --- 保存结果 ---
df = pd.DataFrame({
    'beta_U': beta_U_values,
    'infection_density': infection_densities,
    'awareness_density': awareness_densities
})
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'MMCA0609.csv')

# 保存文件
df.to_csv(file_path, index=False)

# --- 绘图 ---
plt.figure(figsize=(8, 6))
plt.plot(beta_U_values, awareness_densities, color='blue', marker='o', linewidth=1, label='MMCA:Aware Nodes Ratio')
plt.plot(beta_U_values, infection_densities, color='red', marker='o', linewidth=1, label='MMCA:Infected Nodes Ratio')
plt.title('Awareness & Infection Ratio vs. β')
plt.xlabel('β')
plt.ylabel('Ratio')
plt.yticks(np.arange(0.0, 1.2, 0.2))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
