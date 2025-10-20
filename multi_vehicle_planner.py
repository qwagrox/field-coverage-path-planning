"""
多机协同路径规划器 (VRP - Vehicle Routing Problem)

用于大型农场的多台拖拉机协同作业调度

核心功能:
1. 田块分配 (K-means 聚类)
2. 负载均衡 (工作量均衡)
3. 路径优化 (每台机器的 TSP)
4. 全局协调
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from genetic_algorithm_solver import GeneticAlgorithmSolver, GAConfig


@dataclass
class VehicleRoute:
    """单台车辆的路径信息"""
    vehicle_id: int  # 车辆ID (0, 1, 2, ...)
    field_ids: List[str]  # 分配的田块ID列表
    field_sequence: List[str]  # 优化后的访问顺序
    total_transfer_distance: float  # 总转移距离
    total_work_distance: float  # 总作业距离 (估算)
    total_distance: float  # 总距离
    work_time: float  # 预估工作时间 (小时)


@dataclass
class MultiVehicleRoute:
    """多车辆的全局路径信息"""
    num_vehicles: int  # 车辆数量
    vehicle_routes: List[VehicleRoute]  # 每台车辆的路径
    total_transfer_distance: float  # 所有车辆的总转移距离
    total_work_distance: float  # 所有车辆的总作业距离
    total_distance: float  # 所有车辆的总距离
    max_work_time: float  # 最长工作时间 (瓶颈)
    load_balance_ratio: float  # 负载均衡比 (最大/平均)


class MultiVehiclePlanner:
    """多机协同路径规划器"""
    
    def __init__(
        self,
        num_vehicles: int,
        optimization_method: str = "genetic"  # "2opt" or "genetic"
    ):
        """
        Args:
            num_vehicles: 车辆数量
            optimization_method: 单车辆路径优化方法
        """
        self.num_vehicles = num_vehicles
        self.optimization_method = optimization_method
        
        print(f"\n[多机协同] 初始化 {num_vehicles} 台车辆")
    
    def plan(
        self,
        fields_data: Dict,  # {field_id: {'centroid': (x,y), 'area': float}}
        depot_point: Tuple[float, float],
        vehicle_params,
        use_genetic: bool = False
    ) -> MultiVehicleRoute:
        """
        规划多车辆的作业路径
        
        Args:
            fields_data: 田块数据字典
            depot_point: 停放点坐标
            vehicle_params: 车辆参数
            use_genetic: 是否使用遗传算法优化
        
        Returns:
            MultiVehicleRoute: 多车辆路径信息
        """
        field_ids = list(fields_data.keys())
        num_fields = len(field_ids)
        
        print(f"\n[第1步] 田块分配 (K-means 聚类)")
        print(f"  田块总数: {num_fields}")
        print(f"  车辆数量: {self.num_vehicles}")
        
        # 1. 田块聚类
        clusters = self._cluster_fields(fields_data, depot_point)
        
        # 2. 负载均衡调整
        clusters = self._balance_workload(clusters, fields_data)
        
        print(f"\n[第2步] 单车辆路径优化")
        
        # 3. 为每台车辆优化路径
        vehicle_routes = []
        
        for vehicle_id in range(self.num_vehicles):
            cluster_field_ids = clusters[vehicle_id]
            
            if len(cluster_field_ids) == 0:
                print(f"  车辆 {vehicle_id}: 无分配田块")
                continue
            
            print(f"  车辆 {vehicle_id}: {len(cluster_field_ids)} 个田块")
            
            # 构建距离矩阵
            distance_matrix = self._build_distance_matrix(
                cluster_field_ids,
                fields_data,
                depot_point
            )
            
            # 优化顺序
            if use_genetic and len(cluster_field_ids) > 20:
                # 使用遗传算法
                config = GAConfig(
                    population_size=min(100, len(cluster_field_ids) * 5),
                    max_generations=200,
                    convergence_threshold=30
                )
                solver = GeneticAlgorithmSolver(config)
                optimal_route, stats = solver.solve(distance_matrix, verbose=False)
            else:
                # 使用 2-opt
                from multi_field_planner_v37 import TSPSolver
                optimal_route = TSPSolver.solve(distance_matrix)
            
            # 转换为田块ID顺序
            node_ids = ["depot"] + cluster_field_ids
            field_sequence = [node_ids[i] for i in optimal_route if node_ids[i] != "depot"]
            
            # 计算距离
            transfer_distance = self._calculate_route_distance(optimal_route, distance_matrix)
            work_distance = sum(fields_data[fid]['area'] / vehicle_params.working_width 
                              for fid in field_sequence)
            total_distance = transfer_distance + work_distance
            
            # 估算工作时间 (假设作业速度 5 km/h, 转移速度 15 km/h)
            work_time = work_distance / 1000 / 5 + transfer_distance / 1000 / 15
            
            vehicle_route = VehicleRoute(
                vehicle_id=vehicle_id,
                field_ids=cluster_field_ids,
                field_sequence=field_sequence,
                total_transfer_distance=transfer_distance,
                total_work_distance=work_distance,
                total_distance=total_distance,
                work_time=work_time
            )
            
            vehicle_routes.append(vehicle_route)
            
            print(f"    - 转移距离: {transfer_distance:.0f}m")
            print(f"    - 作业距离: {work_distance:.0f}m")
            print(f"    - 预估时间: {work_time:.1f}h")
        
        # 4. 计算全局统计
        total_transfer = sum(vr.total_transfer_distance for vr in vehicle_routes)
        total_work = sum(vr.total_work_distance for vr in vehicle_routes)
        total_dist = sum(vr.total_distance for vr in vehicle_routes)
        max_time = max(vr.work_time for vr in vehicle_routes)
        avg_time = np.mean([vr.work_time for vr in vehicle_routes])
        load_balance = max_time / avg_time if avg_time > 0 else 1.0
        
        print(f"\n[全局统计]")
        print(f"  总转移距离: {total_transfer:.0f}m")
        print(f"  总作业距离: {total_work:.0f}m")
        print(f"  最长工作时间: {max_time:.1f}h (车辆 {vehicle_routes[np.argmax([vr.work_time for vr in vehicle_routes])].vehicle_id})")
        print(f"  负载均衡比: {load_balance:.2f} (越接近1越均衡)")
        
        return MultiVehicleRoute(
            num_vehicles=self.num_vehicles,
            vehicle_routes=vehicle_routes,
            total_transfer_distance=total_transfer,
            total_work_distance=total_work,
            total_distance=total_dist,
            max_work_time=max_time,
            load_balance_ratio=load_balance
        )
    
    def _cluster_fields(
        self,
        fields_data: Dict,
        depot_point: Tuple[float, float]
    ) -> List[List[str]]:
        """使用 K-means 聚类分配田块"""
        field_ids = list(fields_data.keys())
        
        # 提取质心坐标
        centroids = np.array([fields_data[fid]['centroid'] for fid in field_ids])
        
        # K-means 聚类
        kmeans = KMeans(n_clusters=self.num_vehicles, random_state=42)
        labels = kmeans.fit_predict(centroids)
        
        # 按簇分组
        clusters = [[] for _ in range(self.num_vehicles)]
        for i, field_id in enumerate(field_ids):
            cluster_id = labels[i]
            clusters[cluster_id].append(field_id)
        
        # 打印聚类结果
        for i, cluster in enumerate(clusters):
            print(f"  簇 {i}: {len(cluster)} 个田块")
        
        return clusters
    
    def _balance_workload(
        self,
        clusters: List[List[str]],
        fields_data: Dict
    ) -> List[List[str]]:
        """负载均衡调整 (简化版)"""
        # 计算每个簇的总面积
        cluster_areas = []
        for cluster in clusters:
            total_area = sum(fields_data[fid]['area'] for fid in cluster)
            cluster_areas.append(total_area)
        
        # 简化版: 不做调整,直接返回
        # 真实版本可以使用贪心算法或启发式算法调整
        return clusters
    
    def _build_distance_matrix(
        self,
        field_ids: List[str],
        fields_data: Dict,
        depot_point: Tuple[float, float]
    ) -> np.ndarray:
        """构建距离矩阵"""
        num_fields = len(field_ids)
        node_ids = ["depot"] + field_ids
        
        distance_matrix = np.zeros((num_fields + 1, num_fields + 1))
        
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # 获取坐标
                    if node_i == "depot":
                        pos_i = np.array(depot_point)
                    else:
                        pos_i = np.array(fields_data[node_i]['centroid'])
                    
                    if node_j == "depot":
                        pos_j = np.array(depot_point)
                    else:
                        pos_j = np.array(fields_data[node_j]['centroid'])
                    
                    distance_matrix[i, j] = np.linalg.norm(pos_i - pos_j)
        
        return distance_matrix
    
    def _calculate_route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route)):
            from_node = route[i]
            to_node = route[(i + 1) % len(route)]
            total_distance += distance_matrix[from_node, to_node]
        return total_distance
    
    def visualize(
        self,
        route: MultiVehicleRoute,
        fields_data: Dict,
        depot_point: Tuple[float, float],
        save_path: str = None
    ):
        """可视化多车辆路径"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # 颜色方案 (每台车辆一个颜色)
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_vehicles))
        
        # 1. 绘制所有田块
        for field_id, field_info in fields_data.items():
            if 'vertices' in field_info:
                vertices = field_info['vertices']
                poly = MplPolygon(vertices, fill=False, edgecolor='gray', linewidth=1, alpha=0.5)
                ax.add_patch(poly)
        
        # 2. 绘制停放点
        ax.plot(depot_point[0], depot_point[1], 'r*', markersize=20, label='停放点', zorder=10)
        ax.text(depot_point[0], depot_point[1] + 30, 'Depot', 
               fontsize=12, ha='center', color='red', weight='bold')
        
        # 3. 为每台车辆绘制路径
        for vehicle_route in route.vehicle_routes:
            vehicle_id = vehicle_route.vehicle_id
            color = colors[vehicle_id]
            
            # 绘制分配的田块 (填充颜色)
            for field_id in vehicle_route.field_ids:
                if 'vertices' in fields_data[field_id]:
                    vertices = fields_data[field_id]['vertices']
                    poly = MplPolygon(vertices, fill=True, facecolor=color, 
                                    edgecolor='black', linewidth=2, alpha=0.3)
                    ax.add_patch(poly)
                
                # 标注田块ID
                centroid = fields_data[field_id]['centroid']
                ax.text(centroid[0], centroid[1], field_id, 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 绘制路径
            if len(vehicle_route.field_sequence) > 0:
                # 停放点 -> 第一个田块
                first_field = vehicle_route.field_sequence[0]
                first_centroid = fields_data[first_field]['centroid']
                ax.plot([depot_point[0], first_centroid[0]], 
                       [depot_point[1], first_centroid[1]], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
                ax.arrow(depot_point[0], depot_point[1],
                        (first_centroid[0] - depot_point[0]) * 0.3,
                        (first_centroid[1] - depot_point[1]) * 0.3,
                        head_width=20, head_length=15, fc=color, ec=color, alpha=0.7)
                
                # 田块之间的路径
                for i in range(len(vehicle_route.field_sequence) - 1):
                    from_field = vehicle_route.field_sequence[i]
                    to_field = vehicle_route.field_sequence[i + 1]
                    from_centroid = fields_data[from_field]['centroid']
                    to_centroid = fields_data[to_field]['centroid']
                    
                    ax.plot([from_centroid[0], to_centroid[0]], 
                           [from_centroid[1], to_centroid[1]], 
                           color=color, linewidth=2, alpha=0.7)
                    
                    # 箭头
                    mid_x = (from_centroid[0] + to_centroid[0]) / 2
                    mid_y = (from_centroid[1] + to_centroid[1]) / 2
                    dx = to_centroid[0] - from_centroid[0]
                    dy = to_centroid[1] - from_centroid[1]
                    ax.arrow(mid_x, mid_y, dx*0.1, dy*0.1,
                            head_width=15, head_length=10, fc=color, ec=color, alpha=0.7)
                
                # 最后一个田块 -> 停放点
                last_field = vehicle_route.field_sequence[-1]
                last_centroid = fields_data[last_field]['centroid']
                ax.plot([last_centroid[0], depot_point[0]], 
                       [last_centroid[1], depot_point[1]], 
                       color=color, linewidth=2, alpha=0.7, linestyle='--')
        
        # 4. 图例
        legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=3, 
                                     label=f'车辆{i} ({len(route.vehicle_routes[i].field_ids)}田块, {route.vehicle_routes[i].work_time:.1f}h)')
                          for i in range(len(route.vehicle_routes))]
        legend_elements.insert(0, plt.Line2D([0], [0], marker='*', color='w', 
                                            markerfacecolor='r', markersize=15, label='停放点'))
        ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'多机协同作业路径 (V3.8.0)\n'
                    f'{self.num_vehicles}台车辆, {len(fields_data)}个田块, '
                    f'最长工作时间: {route.max_work_time:.1f}h, '
                    f'负载均衡比: {route.load_balance_ratio:.2f}',
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可视化已保存: {save_path}")
        
        return fig, ax


# 测试代码
if __name__ == '__main__':
    print("多机协同路径规划器 - 单元测试")
    print("请运行 test_multi_vehicle_v38.py 进行完整测试")

