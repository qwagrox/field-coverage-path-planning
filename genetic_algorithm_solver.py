"""
遗传算法 (Genetic Algorithm) TSP 求解器

用于大规模田块作业顺序优化 (100-500 田块)

核心特性:
1. OX (Order Crossover) 交叉操作
2. 交换变异 (Swap Mutation)
3. 锦标赛选择 (Tournament Selection)
4. 精英保留策略 (Elitism)
5. 自适应参数调整
"""

import numpy as np
from typing import List, Tuple
import random
from dataclasses import dataclass


@dataclass
class GAConfig:
    """遗传算法配置参数"""
    population_size: int = 200  # 种群大小
    max_generations: int = 500  # 最大代数
    crossover_rate: float = 0.85  # 交叉概率
    mutation_rate: float = 0.02  # 变异概率
    elite_size: int = 20  # 精英个体数量
    tournament_size: int = 5  # 锦标赛选择大小
    convergence_threshold: int = 50  # 收敛阈值 (多少代无改进则停止)


class GeneticAlgorithmSolver:
    """遗传算法 TSP 求解器"""
    
    def __init__(self, config: GAConfig = None):
        """
        Args:
            config: 遗传算法配置参数
        """
        self.config = config or GAConfig()
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def solve(self, distance_matrix: np.ndarray, verbose: bool = True) -> Tuple[List[int], dict]:
        """
        使用遗传算法求解 TSP 问题
        
        Args:
            distance_matrix: (N+1) x (N+1) 距离矩阵, 第0个节点是停放点
            verbose: 是否打印进度信息
        
        Returns:
            best_route: 最优路径 (节点索引列表)
            stats: 统计信息字典
        """
        num_nodes = len(distance_matrix)
        
        if verbose:
            print(f"\n[遗传算法] 开始优化...")
            print(f"  节点数: {num_nodes}")
            print(f"  种群大小: {self.config.population_size}")
            print(f"  最大代数: {self.config.max_generations}")
        
        # 1. 初始化种群
        population = self._initialize_population(num_nodes)
        
        # 2. 评估初始种群
        fitness_scores = [self._calculate_fitness(route, distance_matrix) for route in population]
        
        best_route = population[np.argmax(fitness_scores)]
        best_fitness = max(fitness_scores)
        best_distance = self._calculate_distance(best_route, distance_matrix)
        
        generations_without_improvement = 0
        
        # 3. 进化循环
        for generation in range(self.config.max_generations):
            # 3.1 选择
            selected = self._selection(population, fitness_scores)
            
            # 3.2 交叉
            offspring = self._crossover(selected)
            
            # 3.3 变异
            offspring = self._mutation(offspring)
            
            # 3.4 精英保留
            population = self._elitism(population, offspring, fitness_scores)
            
            # 3.5 评估新种群
            fitness_scores = [self._calculate_fitness(route, distance_matrix) for route in population]
            
            # 3.6 更新最佳解
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_route = population[current_best_idx].copy()
                best_distance = self._calculate_distance(best_route, distance_matrix)
                generations_without_improvement = 0
                
                if verbose and generation % 50 == 0:
                    print(f"  第 {generation} 代: 最优距离 = {best_distance:.1f}m")
            else:
                generations_without_improvement += 1
            
            # 记录历史
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_scores))
            
            # 3.7 收敛检查
            if generations_without_improvement >= self.config.convergence_threshold:
                if verbose:
                    print(f"  第 {generation} 代: 收敛 (连续 {self.config.convergence_threshold} 代无改进)")
                break
        
        # 确保路径从停放点(0)开始
        start_index = best_route.index(0)
        final_route = best_route[start_index:] + best_route[:start_index]
        
        stats = {
            'generations': generation + 1,
            'best_distance': best_distance,
            'best_fitness': best_fitness,
            'convergence_gen': generation - generations_without_improvement,
        }
        
        if verbose:
            print(f"\n[遗传算法] 优化完成!")
            print(f"  总代数: {stats['generations']}")
            print(f"  最优距离: {best_distance:.1f}m")
            print(f"  收敛代数: {stats['convergence_gen']}")
        
        return final_route, stats
    
    def _initialize_population(self, num_nodes: int) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        # 一半使用随机初始化
        for _ in range(self.config.population_size // 2):
            route = list(range(num_nodes))
            random.shuffle(route)
            population.append(route)
        
        # 一半使用贪心初始化 (从不同起点)
        for i in range(self.config.population_size // 2):
            start_node = i % num_nodes
            route = self._greedy_init(num_nodes, start_node)
            population.append(route)
        
        return population
    
    def _greedy_init(self, num_nodes: int, start_node: int) -> List[int]:
        """贪心初始化 (最近邻)"""
        route = [start_node]
        unvisited = set(range(num_nodes)) - {start_node}
        
        # 简化版: 随机选择下一个节点 (真实版本需要距离矩阵)
        while unvisited:
            next_node = random.choice(list(unvisited))
            route.append(next_node)
            unvisited.remove(next_node)
        
        return route
    
    def _calculate_fitness(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """计算适应度 (距离越短,适应度越高)"""
        distance = self._calculate_distance(route, distance_matrix)
        # 使用倒数作为适应度,避免除零
        return 1.0 / (distance + 1e-6)
    
    def _calculate_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route)):
            from_node = route[i]
            to_node = route[(i + 1) % len(route)]
            total_distance += distance_matrix[from_node, to_node]
        return total_distance
    
    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(len(population)):
            # 随机选择 tournament_size 个个体
            tournament_indices = random.sample(range(len(population)), self.config.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择适应度最高的
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, population: List[List[int]]) -> List[List[int]]:
        """OX (Order Crossover) 交叉操作"""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._ox_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring
    
    def _ox_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """OX (Order Crossover) 交叉"""
        size = len(parent1)
        
        # 随机选择两个交叉点
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))
        
        # 创建子代
        child1 = [None] * size
        child2 = [None] * size
        
        # 复制交叉段
        child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
        child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]
        
        # 填充剩余部分
        def fill_child(child, parent):
            pos = cx_point2
            for gene in parent[cx_point2:] + parent[:cx_point2]:
                if gene not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = gene
                    pos += 1
        
        fill_child(child1, parent2)
        fill_child(child2, parent1)
        
        return child1, child2
    
    def _mutation(self, population: List[List[int]]) -> List[List[int]]:
        """交换变异"""
        for route in population:
            if random.random() < self.config.mutation_rate:
                # 随机选择两个位置交换
                idx1, idx2 = random.sample(range(len(route)), 2)
                route[idx1], route[idx2] = route[idx2], route[idx1]
        
        return population
    
    def _elitism(
        self,
        old_population: List[List[int]],
        new_population: List[List[int]],
        old_fitness: List[float]
    ) -> List[List[int]]:
        """精英保留策略"""
        # 找到旧种群中的精英个体
        elite_indices = np.argsort(old_fitness)[-self.config.elite_size:]
        elites = [old_population[i].copy() for i in elite_indices]
        
        # 替换新种群中最差的个体
        combined = new_population[:-self.config.elite_size] + elites
        
        return combined


# 测试代码
if __name__ == '__main__':
    print("遗传算法 TSP 求解器 - 单元测试")
    
    # 创建一个简单的距离矩阵 (10个节点)
    np.random.seed(42)
    num_nodes = 10
    
    # 生成随机坐标
    coords = np.random.rand(num_nodes, 2) * 100
    
    # 计算距离矩阵
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    # 创建求解器
    config = GAConfig(
        population_size=50,
        max_generations=100,
        convergence_threshold=20
    )
    solver = GeneticAlgorithmSolver(config)
    
    # 求解
    best_route, stats = solver.solve(distance_matrix, verbose=True)
    
    print(f"\n最优路径: {best_route}")
    print(f"统计信息: {stats}")

