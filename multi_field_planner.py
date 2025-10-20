"""
多田块作业调度器 V3.8.0 - 大规模优化版本

新增功能:
1. 遗传算法 (支持 100-500 田块)
2. 多机协同 (VRP, 支持 2-10 台拖拉机)
3. 自动算法选择
4. 大规模场景可视化优化

版本历史:
- V3.7.0: 基础 TSP 优化 (2-opt)
- V3.8.0: 遗传算法 + 多机协同 + 自动选择
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
matplotlib.rcParams['axes.unicode_minus'] = False

from multi_layer_planner_v3 import TwoLayerPathPlannerV36, VehicleParams
from genetic_algorithm_solver import GeneticAlgorithmSolver, GAConfig
from multi_vehicle_planner import MultiVehiclePlanner, MultiVehicleRoute


@dataclass
class FieldData:
    """存储单个田块的所有相关信息"""
    id: str
    vertices: np.ndarray
    planner: TwoLayerPathPlannerV36 = None
    centroid: Tuple[float, float] = None
    area: float = None
    entry_points: List[Tuple[np.ndarray, np.ndarray]] = None
    exit_points: List[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class Connection:
    """存储两个田块之间的连接信息"""
    from_field: str
    to_field: str
    from_point: np.ndarray
    to_point: np.ndarray
    distance: float


@dataclass
class OptimizedRoute:
    """存储最终的优化结果 (单机版本)"""
    field_sequence: List[str]
    connections: List[Connection]
    total_transfer_distance: float
    total_work_distance: float
    total_distance: float
    optimization_method: str  # "2opt" or "genetic"
    optimization_stats: dict = None  # 优化统计信息


class MultiFieldPlannerV38:
    """多田块作业调度器 V3.8.0"""
    
    def __init__(
        self,
        fields_definitions: List[dict],
        depot_point: Tuple[float, float],
        vehicle_params: VehicleParams,
        num_vehicles: int = 1,
        optimization_method: str = "auto"
    ):
        """
        Args:
            fields_definitions: 字典列表, 每个字典包含 'id' 和 'vertices'
            depot_point: 停放点坐标
            vehicle_params: 车辆参数
            num_vehicles: 车辆数量 (1 = 单机, 2+ = 多机协同)
            optimization_method: 优化方法 ("auto", "2opt", "genetic")
        """
        self.depot = np.array(depot_point)
        self.vehicle_params = vehicle_params
        self.num_vehicles = num_vehicles
        self.optimization_method = optimization_method
        self.fields: Dict[str, FieldData] = {}
        
        print(f"\n{'='*70}")
        print(f"[V3.8.0] 初始化多田块作业调度器 (大规模优化版)")
        print(f"{'='*70}")
        print(f"停放点: ({depot_point[0]:.1f}, {depot_point[1]:.1f})")
        print(f"田块数量: {len(fields_definitions)}")
        print(f"车辆数量: {num_vehicles}")
        print(f"优化方法: {optimization_method}")
        
        self._prepare_fields(fields_definitions)
        
        # 自动选择优化方法
        if self.optimization_method == "auto":
            self.optimization_method = self._select_optimization_method()
            print(f"自动选择优化方法: {self.optimization_method}")
        
        print(f"{'='*70}")
    
    def _prepare_fields(self, fields_definitions: List[dict]):
        """初始化所有田块"""
        print(f"\n[准备田块]")
        for field_def in fields_definitions:
            field_id = field_def['id']
            vertices = field_def['vertices']
            
            planner = TwoLayerPathPlannerV36(
                vehicle_params=self.vehicle_params,
                field_vertices=vertices
            )
            
            centroid = planner.field_polygon.centroid.coords[0]
            area = planner.field_polygon.area
            
            entry_points = []
            exit_points = []
            
            for i, vertex in enumerate(planner.field_vertices):
                prev_vertex = planner.field_vertices[i - 1]
                next_vertex = planner.field_vertices[(i + 1) % len(planner.field_vertices)]
                
                v_in = np.array(vertex) - np.array(prev_vertex)
                v_in = v_in / np.linalg.norm(v_in)
                
                v_out = np.array(next_vertex) - np.array(vertex)
                v_out = v_out / np.linalg.norm(v_out)
                
                v_avg = (v_in + v_out) / 2
                if np.linalg.norm(v_avg) > 0.1:
                    v_avg = v_avg / np.linalg.norm(v_avg)
                else:
                    v_avg = v_in
                
                entry_points.append((np.array(vertex), v_avg))
                exit_points.append((np.array(vertex), v_avg))
            
            self.fields[field_id] = FieldData(
                id=field_id,
                vertices=np.array(vertices),
                planner=planner,
                centroid=centroid,
                area=area,
                entry_points=entry_points,
                exit_points=exit_points
            )
            print(f"  ✓ 田块 '{field_id}': 质心=({centroid[0]:.1f}, {centroid[1]:.1f}), 面积={area:.0f}m²")
    
    def _select_optimization_method(self) -> str:
        """根据田块数量自动选择优化方法"""
        num_fields = len(self.fields)
        
        if num_fields < 50:
            return "2opt"
        elif num_fields < 150:
            return "genetic"
        else:
            return "genetic"  # 大规模场景也使用遗传算法
    
    def optimize_sequence(self) -> OptimizedRoute:
        """执行优化 (单机版本)"""
        if self.num_vehicles > 1:
            raise ValueError("多机协同请使用 optimize_multi_vehicle() 方法")
        
        print(f"\n[单机优化] 使用 {self.optimization_method} 算法")
        
        # 1. 计算距离矩阵
        distance_matrix, node_ids = self._calculate_distance_matrix()
        
        # 2. 调用优化算法
        if self.optimization_method == "2opt":
            from multi_field_planner_v37 import TSPSolver
            optimal_route_indices = TSPSolver.solve(distance_matrix)
            stats = {'method': '2opt'}
        else:  # genetic
            config = GAConfig(
                population_size=min(200, len(self.fields) * 4),
                max_generations=500,
                convergence_threshold=50
            )
            solver = GeneticAlgorithmSolver(config)
            optimal_route_indices, stats = solver.solve(distance_matrix, verbose=True)
            stats['method'] = 'genetic'
        
        optimal_route_ids = [node_ids[i] for i in optimal_route_indices]
        field_sequence = [id for id in optimal_route_ids if id != "depot"]
        
        print(f"  优化后的田块顺序: {' -> '.join(field_sequence[:5])}{'...' if len(field_sequence) > 5 else ''}")
        
        # 3. 微观连接优化
        print(f"\n[微观连接优化]")
        connections = []
        total_transfer_distance = 0
        
        conn = self._find_best_connection("depot", field_sequence[0])
        connections.append(conn)
        total_transfer_distance += conn.distance
        
        for i in range(len(field_sequence) - 1):
            conn = self._find_best_connection(field_sequence[i], field_sequence[i + 1])
            connections.append(conn)
            total_transfer_distance += conn.distance
        
        conn = self._find_best_connection(field_sequence[-1], "depot")
        connections.append(conn)
        total_transfer_distance += conn.distance
        
        # 4. 计算总作业距离
        total_work_distance = sum(
            self.fields[field_id].area / self.vehicle_params.working_width
            for field_id in field_sequence
        )
        
        total_distance = total_transfer_distance + total_work_distance
        
        print(f"\n[优化结果]")
        print(f"  总转移距离: {total_transfer_distance:.1f}m")
        print(f"  总作业距离(估算): {total_work_distance:.1f}m")
        print(f"  总距离: {total_distance:.1f}m")
        
        return OptimizedRoute(
            field_sequence=field_sequence,
            connections=connections,
            total_transfer_distance=total_transfer_distance,
            total_work_distance=total_work_distance,
            total_distance=total_distance,
            optimization_method=self.optimization_method,
            optimization_stats=stats
        )
    
    def optimize_multi_vehicle(self) -> MultiVehicleRoute:
        """执行优化 (多机版本)"""
        if self.num_vehicles == 1:
            raise ValueError("单机优化请使用 optimize_sequence() 方法")
        
        print(f"\n[多机协同优化]")
        
        # 准备田块数据
        fields_data = {}
        for field_id, field_info in self.fields.items():
            fields_data[field_id] = {
                'centroid': field_info.centroid,
                'area': field_info.area,
                'vertices': field_info.vertices
            }
        
        # 创建多车辆规划器
        mvp = MultiVehiclePlanner(
            num_vehicles=self.num_vehicles,
            optimization_method=self.optimization_method
        )
        
        # 规划
        use_genetic = (self.optimization_method == "genetic")
        route = mvp.plan(fields_data, tuple(self.depot), self.vehicle_params, use_genetic)
        
        return route
    
    def _calculate_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """计算距离矩阵"""
        field_ids = list(self.fields.keys())
        num_fields = len(field_ids)
        node_ids = ["depot"] + field_ids
        
        distance_matrix = np.zeros((num_fields + 1, num_fields + 1))
        
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    if node_i == "depot":
                        pos_i = self.depot
                    else:
                        pos_i = np.array(self.fields[node_i].centroid)
                    
                    if node_j == "depot":
                        pos_j = self.depot
                    else:
                        pos_j = np.array(self.fields[node_j].centroid)
                    
                    distance_matrix[i, j] = np.linalg.norm(pos_i - pos_j)
        
        return distance_matrix, node_ids
    
    def _find_best_connection(self, from_id: str, to_id: str) -> Connection:
        """为两个连续节点找到最短的连接"""
        if from_id == "depot":
            from_candidates = [(self.depot, np.array([0, 0]))]
        else:
            from_candidates = self.fields[from_id].exit_points
        
        if to_id == "depot":
            to_candidates = [(self.depot, np.array([0, 0]))]
        else:
            to_candidates = self.fields[to_id].entry_points
        
        best_distance = float('inf')
        best_from_point = None
        best_to_point = None
        
        for from_point, _ in from_candidates:
            for to_point, _ in to_candidates:
                distance = np.linalg.norm(from_point - to_point)
                if distance < best_distance:
                    best_distance = distance
                    best_from_point = from_point
                    best_to_point = to_point
        
        return Connection(
            from_field=from_id,
            to_field=to_id,
            from_point=best_from_point,
            to_point=best_to_point,
            distance=best_distance
        )
    
    def visualize_single(self, route: OptimizedRoute, save_path: str = None):
        """可视化单机路径 (优化大规模场景)"""
        num_fields = len(self.fields)
        
        # 根据田块数量调整图形大小和字体
        if num_fields < 20:
            figsize = (14, 10)
            field_label_size = 10
            show_all_labels = True
        elif num_fields < 50:
            figsize = (16, 12)
            field_label_size = 8
            show_all_labels = True
        else:
            figsize = (20, 15)
            field_label_size = 6
            show_all_labels = False  # 大规模场景不显示所有标签
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 1. 绘制所有田块
        for field_id, field_data in self.fields.items():
            poly = MplPolygon(field_data.vertices, fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(poly)
            
            if show_all_labels:
                centroid = field_data.centroid
                ax.text(centroid[0], centroid[1], field_id, 
                       fontsize=field_label_size, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 2. 绘制停放点
        ax.plot(self.depot[0], self.depot[1], 'rs', markersize=15, label='停放点')
        ax.text(self.depot[0], self.depot[1] + 20, 'Depot', 
               fontsize=12, ha='center', color='red', weight='bold')
        
        # 3. 绘制优化后的连接路径
        for i, conn in enumerate(route.connections):
            ax.plot([conn.from_point[0], conn.to_point[0]], 
                   [conn.from_point[1], conn.to_point[1]], 
                   'b-', linewidth=2, alpha=0.6)
            
            mid_x = (conn.from_point[0] + conn.to_point[0]) / 2
            mid_y = (conn.from_point[1] + conn.to_point[1]) / 2
            dx = conn.to_point[0] - conn.from_point[0]
            dy = conn.to_point[1] - conn.from_point[1]
            
            if num_fields < 50:  # 只在小规模场景显示箭头和距离
                ax.arrow(mid_x, mid_y, dx*0.1, dy*0.1, 
                        head_width=15, head_length=10, fc='blue', ec='blue', alpha=0.6)
                
                if i % 3 == 0:  # 每3个连接显示一次距离
                    ax.text(mid_x, mid_y, f'{conn.distance:.0f}m', 
                           fontsize=8, ha='center', 
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 4. 标注访问顺序 (只标注前几个)
        num_to_label = min(10, len(route.field_sequence))
        for i in range(num_to_label):
            field_id = route.field_sequence[i]
            centroid = self.fields[field_id].centroid
            ax.text(centroid[0] - 20, centroid[1] - 20, f'#{i+1}', 
                   fontsize=12, ha='center', color='red', weight='bold')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        title = f'多田块作业顺序优化 (V3.8.0 - {route.optimization_method.upper()})\n'
        title += f'{num_fields}个田块, 总转移距离: {route.total_transfer_distance:.0f}m'
        if route.optimization_stats and 'generations' in route.optimization_stats:
            title += f', 迭代: {route.optimization_stats["generations"]}代'
        
        ax.set_title(title, fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可视化已保存: {save_path}")
        
        return fig, ax


if __name__ == '__main__':
    print("多田块调度器 V3.8.0 - 模块测试")
    print("请运行 test_multi_field_v38.py 进行完整测试")

