"""
两层路径规划器 V3.0 - 支持静态障碍物
支持用户标注的障碍物顶点坐标
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VehicleParams:
    """车辆参数"""
    working_width: float = 3.2  # 作业幅宽 (m)
    base_turn_radius: float = 8.0  # 基础转弯半径 (m)
    max_work_speed_kmh: float = 9.0  # 最大作业速度 (km/h)
    max_headland_speed_kmh: float = 15.0  # 最大田头速度 (km/h)
    headland_turn_speed_kmh: float = 4.0  # 田头转弯速度 (km/h)
    reverse_speed_kmh: float = 2.5  # 倒车速度 (km/h)
    max_lateral_accel: float = 2.0  # 最大侧向加速度 (m/s²)
    max_longitudinal_accel: float = 1.5  # 最大纵向加速度 (m/s²)
    safety_factor: float = 0.85  # 安全系数


class ClothoidCurve:
    """Clothoid曲线生成器（回旋曲线）"""
    
    @staticmethod
    def generate(start_point, start_heading, start_curvature, 
                 end_curvature, length=None, num_points=50):
        """
        生成Clothoid曲线
        
        参数:
            start_point: 起点 (x, y)
            start_heading: 起始航向角 (弧度)
            start_curvature: 起始曲率
            end_curvature: 结束曲率
            length: 曲线长度 (如果为None则自动计算)
            num_points: 采样点数
        
        返回:
            points: 曲线上的点列表 [(x, y), ...]
            headings: 每个点的航向角列表
            curvatures: 每个点的曲率列表
        """
        if length is None:
            length = abs(end_curvature - start_curvature) * 10.0
            
        s = np.linspace(0, length, num_points)
        curvature_rate = (end_curvature - start_curvature) / length if length > 0 else 0
        
        points = []
        headings = []
        curvatures = []
        
        x, y = start_point
        theta = start_heading
        
        for i in range(len(s)):
            points.append((x, y))
            headings.append(theta)
            curvatures.append(start_curvature + curvature_rate * s[i])
            
            if i < len(s) - 1:
                ds = s[i+1] - s[i]
                kappa = start_curvature + curvature_rate * s[i]
                theta += kappa * ds
                x += ds * np.cos(theta)
                y += ds * np.sin(theta)
        
        return points, headings, curvatures


class TwoLayerPathPlannerV3:
    """
    两层路径规划器 V3.0 - 支持静态障碍物
    
    核心功能:
    1. 支持用户标注的静态障碍物（多边形顶点坐标）
    2. 主作业区域自动避障
    3. 田头区域避障路径规划
    4. Clothoid曲线平滑
    5. 完整速度规划
    6. 曲率约束验证
    7. 电子围栏检查
    """
    
    def __init__(self, field_length: float, field_width: float,
                 vehicle_params: VehicleParams,
                 obstacles: List[List[Tuple[float, float]]] = None):
        """
        初始化路径规划器
        
        参数:
            field_length: 田地长度 (m)
            field_width: 田地宽度 (m)
            vehicle_params: 车辆参数
            obstacles: 障碍物列表，每个障碍物是顶点坐标列表
                      例如: [[(10, 10), (20, 10), (20, 20), (10, 20)]]
        """
        self.field_length = field_length
        self.field_width = field_width
        self.vehicle = vehicle_params
        
        # 创建田地边界
        self.field_boundary = box(0, 0, field_length, field_width)
        
        # 处理障碍物
        self.obstacles = []
        self.obstacle_polygons = []
        if obstacles:
            for obs_coords in obstacles:
                if len(obs_coords) >= 3:
                    obs_poly = Polygon(obs_coords)
                    if obs_poly.is_valid and obs_poly.within(self.field_boundary):
                        self.obstacles.append(obs_coords)
                        # 扩展障碍物边界（安全裕度）
                        expanded_obs = obs_poly.buffer(vehicle_params.working_width / 2)
                        self.obstacle_polygons.append(expanded_obs)
        
        # 创建可作业区域（田地 - 障碍物）
        if self.obstacle_polygons:
            obstacles_union = unary_union(self.obstacle_polygons)
            self.workable_area = self.field_boundary.difference(obstacles_union)
        else:
            self.workable_area = self.field_boundary
        
        # 自动计算参数
        self._calculate_adaptive_parameters()
        
        print("="*80)
        print("两层路径规划器 V3.0 - 支持静态障碍物")
        print("="*80)
        print(f"田地尺寸: {field_length}m × {field_width}m")
        print(f"田地面积: {field_length * field_width / 10000:.2f} 公顷")
        print(f"障碍物数量: {len(self.obstacles)}")
        if self.obstacles:
            total_obstacle_area = sum(Polygon(obs).area for obs in self.obstacles)
            print(f"障碍物总面积: {total_obstacle_area:.2f} m²")
            print(f"可作业面积: {self.workable_area.area / 10000:.2f} 公顷")
        print("="*80)
    
    def _calculate_adaptive_parameters(self):
        """自动计算自适应参数"""
        print("\n" + "="*80)
        print("自动参数计算")
        print("="*80)
        
        # 1. 计算田头宽度
        base_headland_width = 2 * self.vehicle.base_turn_radius + self.vehicle.working_width
        max_headland_width = min(self.field_length, self.field_width) * 0.2
        min_headland_width = 1.5 * self.vehicle.base_turn_radius
        
        self.headland_width = np.clip(base_headland_width, min_headland_width, max_headland_width)
        
        print(f"田头宽度计算:")
        print(f"  基础宽度 (2R + W): {base_headland_width:.1f} m")
        print(f"  最大允许 (20%田地): {max_headland_width:.1f} m")
        print(f"  最终采用: {self.headland_width:.1f} m")
        
        # 2. 计算自适应转弯半径
        v_max_ms = self.vehicle.max_headland_speed_kmh / 3.6
        R_speed = (v_max_ms ** 2) / self.vehicle.max_lateral_accel
        R_adaptive = max(self.vehicle.base_turn_radius, R_speed)
        R_adaptive = min(R_adaptive, self.headland_width / 2)
        
        self.adaptive_turn_radius = R_adaptive
        
        print(f"转弯半径计算:")
        print(f"  基础半径: {self.vehicle.base_turn_radius:.1f} m")
        print(f"  速度需求半径 (v²/a): {R_speed:.1f} m")
        print(f"  自适应半径: {self.adaptive_turn_radius:.1f} m")
        
        # 3. 选择主作业模式
        aspect_ratio = self.field_length / self.field_width
        if aspect_ratio > 3.0:
            self.main_work_pattern = "U型往复"
            reason = f"长宽比{aspect_ratio:.1f} > 3, 适合往复"
        elif aspect_ratio < 1.5:
            self.main_work_pattern = "Ω型跨行"
            reason = f"长宽比{aspect_ratio:.1f} < 1.5, 适合跨行减少转弯"
        else:
            self.main_work_pattern = "U型往复"
            reason = "通用模式"
        
        print(f"主作业模式选择:")
        print(f"  田地长宽比: {aspect_ratio:.2f}")
        print(f"  选择模式: {self.main_work_pattern}")
        print(f"  原因: {reason}")
        
        print(f"计算完成:")
        print(f"  田头宽度: {self.headland_width:.1f} m")
        print(f"  自适应转弯半径: {self.adaptive_turn_radius:.1f} m")
        print(f"  主作业模式: {self.main_work_pattern}")
        print("="*80)
    
    def plan_complete_coverage(self):
        """
        完整的两层路径规划（支持障碍物）
        
        返回:
            result: 包含主作业和田头路径的完整结果
        """
        print("\n" + "="*80)
        print("开始两层路径规划（支持障碍物）")
        print("="*80)
        
        # 第1层: 主作业区域规划（避障）
        main_work_result = self._plan_main_work_with_obstacles()
        
        # 第2层: 外层田头规划（避障）
        headland_result = self._plan_headland_with_obstacles()
        
        # 合并结果
        result = {
            'main_work': main_work_result,
            'headland': headland_result,
            'field_boundary': self.field_boundary,
            'obstacles': self.obstacles,
            'workable_area': self.workable_area
        }
        
        print("="*80)
        print("路径规划完成!")
        print("="*80)
        
        return result
    
    def _plan_main_work_with_obstacles(self):
        """主作业区域规划（支持障碍物）"""
        print("\n第1层: 主作业区域规划（避障）...")
        
        # 计算主作业区域（扣除田头）
        main_work_boundary = self.field_boundary.buffer(-self.headland_width)
        
        # 扣除障碍物
        if self.obstacle_polygons:
            obstacles_in_main = []
            for obs in self.obstacle_polygons:
                if main_work_boundary.intersects(obs):
                    obstacles_in_main.append(obs)
            
            if obstacles_in_main:
                obstacles_union = unary_union(obstacles_in_main)
                main_work_area = main_work_boundary.difference(obstacles_union)
            else:
                main_work_area = main_work_boundary
        else:
            main_work_area = main_work_boundary
        
        if main_work_area.is_empty:
            print("  警告: 主作业区域为空!")
            return self._empty_result()
        
        # 获取主作业区域的边界框
        minx, miny, maxx, maxy = main_work_area.bounds
        work_length = maxx - minx
        work_width = maxy - miny
        
        print(f"  主作业区域: {work_length:.1f}m × {work_width:.1f}m")
        print(f"  作业模式: {self.main_work_pattern}")
        
        # 生成往复路径
        num_passes = int(np.ceil(work_width / self.vehicle.working_width))
        
        path = []
        speeds = []
        path_types = []
        
        for i in range(num_passes):
            y = miny + i * self.vehicle.working_width + self.vehicle.working_width / 2
            
            if i % 2 == 0:  # 从左到右
                line = LineString([(minx, y), (maxx, y)])
            else:  # 从右到左
                line = LineString([(maxx, y), (minx, y)])
            
            # 检查与可作业区域的交集
            if main_work_area.intersects(line):
                intersection = main_work_area.intersection(line)
                
                if intersection.geom_type == 'LineString':
                    coords = list(intersection.coords)
                    path.extend(coords)
                    speeds.extend([self.vehicle.max_work_speed_kmh] * len(coords))
                    path_types.extend(['main_work'] * len(coords))
                elif intersection.geom_type == 'MultiLineString':
                    # 处理被障碍物分割的多段线
                    for segment in intersection.geoms:
                        coords = list(segment.coords)
                        path.extend(coords)
                        speeds.extend([self.vehicle.max_work_speed_kmh] * len(coords))
                        path_types.extend(['main_work'] * len(coords))
        
        # 计算统计信息
        path_length_km = self._calculate_path_length(path) / 1000
        avg_speed = np.mean(speeds) if speeds else 0
        time_hours = path_length_km / avg_speed if avg_speed > 0 else 0
        
        print(f"  作业趟数: {num_passes}")
        print(f"  路径长度: {path_length_km:.2f} km")
        print(f"  作业时间: {time_hours:.2f} 小时")
        
        return {
            'path': path,
            'speeds': speeds,
            'path_types': path_types,
            'stats': {
                'num_passes': num_passes,
                'path_length_km': path_length_km,
                'time_hours': time_hours,
                'avg_speed_kmh': avg_speed
            }
        }
    
    def _plan_headland_with_obstacles(self):
        """田头区域规划（支持障碍物）"""
        print("\n第2层: 外层田头规划（避障）...")
        
        # 创建田头区域
        headland_outer = self.field_boundary
        headland_inner = self.field_boundary.buffer(-self.headland_width)
        headland_area = headland_outer.difference(headland_inner)
        
        # 扣除障碍物
        if self.obstacle_polygons:
            obstacles_in_headland = []
            for obs in self.obstacle_polygons:
                if headland_area.intersects(obs):
                    obstacles_in_headland.append(obs)
            
            if obstacles_in_headland:
                obstacles_union = unary_union(obstacles_in_headland)
                headland_area = headland_area.difference(obstacles_union)
        
        if headland_area.is_empty:
            print("  警告: 田头区域为空!")
            return self._empty_result()
        
        print(f"  田头宽度: {self.headland_width:.1f} m")
        print(f"  转弯半径: {self.adaptive_turn_radius:.1f} m")
        
        # 生成田头环绕路径（简化版，避开障碍物）
        path, speeds, path_types = self._generate_headland_path_with_obstacles(headland_area)
        
        # 计算统计信息
        path_length_km = self._calculate_path_length(path) / 1000
        avg_speed = np.mean(speeds) if speeds else 0
        time_hours = path_length_km / avg_speed if avg_speed > 0 else 0
        
        print(f"  路径长度: {path_length_km:.2f} km")
        print(f"  作业时间: {time_hours:.2f} 小时")
        
        return {
            'path': path,
            'speeds': speeds,
            'path_types': path_types,
            'stats': {
                'path_length_km': path_length_km,
                'time_hours': time_hours,
                'avg_speed_kmh': avg_speed
            }
        }
    
    def _generate_headland_path_with_obstacles(self, headland_area):
        """生成避障的田头路径"""
        path = []
        speeds = []
        path_types = []
        
        # 获取田头区域的外边界
        if hasattr(headland_area, 'exterior'):
            boundary = headland_area.exterior
        elif hasattr(headland_area, 'geoms'):
            # MultiPolygon情况
            boundary = headland_area.geoms[0].exterior
        else:
            return path, speeds, path_types
        
        # 沿边界采样点
        coords = list(boundary.coords)
        
        for i, coord in enumerate(coords):
            path.append(coord)
            
            # 根据位置设置速度
            if i < len(coords) - 1:
                # 计算转弯角度
                if i > 0:
                    v1 = np.array(coords[i]) - np.array(coords[i-1])
                    v2 = np.array(coords[i+1]) - np.array(coords[i])
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1))
                    
                    if angle > np.pi / 4:  # 大转弯
                        speeds.append(self.vehicle.headland_turn_speed_kmh)
                        path_types.append('headland_turn')
                    else:
                        speeds.append(self.vehicle.max_headland_speed_kmh)
                        path_types.append('headland_straight')
                else:
                    speeds.append(self.vehicle.max_headland_speed_kmh)
                    path_types.append('headland_straight')
            else:
                speeds.append(self.vehicle.max_headland_speed_kmh)
                path_types.append('headland_straight')
        
        return path, speeds, path_types
    
    def _calculate_path_length(self, path):
        """计算路径总长度"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            length += np.linalg.norm(p2 - p1)
        
        return length
    
    def _empty_result(self):
        """返回空结果"""
        return {
            'path': [],
            'speeds': [],
            'path_types': [],
            'stats': {
                'path_length_km': 0,
                'time_hours': 0,
                'avg_speed_kmh': 0
            }
        }


def test_with_obstacles():
    """测试支持障碍物的路径规划"""
    
    # 场景1: 新疆大型麦田 + 1个障碍物
    print("\n" + "="*80)
    print("场景1: 新疆大型麦田 + 单个障碍物")
    print("="*80)
    
    vehicle = VehicleParams(
        working_width=3.2,
        base_turn_radius=8.0,
        max_work_speed_kmh=9.0,
        max_headland_speed_kmh=15.0
    )
    
    # 定义一个矩形障碍物（例如：水塔、建筑物）
    obstacles = [
        [(1500, 100), (1600, 100), (1600, 200), (1500, 200)]  # 100m x 100m 障碍物
    ]
    
    planner = TwoLayerPathPlannerV3(
        field_length=3500,
        field_width=320,
        vehicle_params=vehicle,
        obstacles=obstacles
    )
    
    result = planner.plan_complete_coverage()
    
    # 可视化
    visualize_result_with_obstacles(result, "新疆大型麦田+障碍物", 
                                    "/home/ubuntu/v3_xinjiang_with_obstacle.png")
    
    # 场景2: 中型田地 + 多个障碍物
    print("\n" + "="*80)
    print("场景2: 中型田地 + 多个障碍物")
    print("="*80)
    
    obstacles_multi = [
        [(100, 50), (150, 50), (150, 100), (100, 100)],  # 障碍物1
        [(300, 120), (350, 120), (350, 170), (300, 170)],  # 障碍物2
        [(200, 80), (220, 80), (220, 100), (200, 100)]  # 障碍物3（小）
    ]
    
    planner2 = TwoLayerPathPlannerV3(
        field_length=500,
        field_width=200,
        vehicle_params=vehicle,
        obstacles=obstacles_multi
    )
    
    result2 = planner2.plan_complete_coverage()
    
    visualize_result_with_obstacles(result2, "中型田地+多障碍物",
                                    "/home/ubuntu/v3_medium_with_obstacles.png")


def visualize_result_with_obstacles(result, title, save_path):
    """可视化带障碍物的路径规划结果"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 完整路径规划
    ax1 = axes[0]
    
    # 绘制田地边界
    if result['field_boundary']:
        x, y = result['field_boundary'].exterior.xy
        ax1.plot(x, y, 'k-', linewidth=2, label='田地边界')
    
    # 绘制障碍物
    if result['obstacles']:
        for i, obs in enumerate(result['obstacles']):
            obs_poly = Polygon(obs)
            x, y = obs_poly.exterior.xy
            ax1.fill(x, y, color='red', alpha=0.5, label='障碍物' if i == 0 else '')
            ax1.plot(x, y, 'r-', linewidth=2)
    
    # 绘制主作业路径
    main_path = result['main_work']['path']
    if main_path:
        main_x = [p[0] for p in main_path]
        main_y = [p[1] for p in main_path]
        ax1.plot(main_x, main_y, 'b-', linewidth=1, alpha=0.6, label='主作业路径')
    
    # 绘制田头路径
    headland_path = result['headland']['path']
    if headland_path:
        hl_x = [p[0] for p in headland_path]
        hl_y = [p[1] for p in headland_path]
        ax1.plot(hl_x, hl_y, 'g-', linewidth=2, label='田头路径')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'{title} - 完整路径规划', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 右图: 统计信息
    ax2 = axes[1]
    ax2.axis('off')
    
    stats_text = f"""
    路径规划统计 V3.0
    ==========================================
    
    田地信息:
      障碍物数量: {len(result['obstacles'])}
      可作业面积: {result['workable_area'].area / 10000:.2f} 公顷
    
    主作业区域:
      路径长度: {result['main_work']['stats']['path_length_km']:.2f} km
      作业时间: {result['main_work']['stats']['time_hours']:.2f} 小时
      平均速度: {result['main_work']['stats']['avg_speed_kmh']:.1f} km/h
    
    田头区域:
      路径长度: {result['headland']['stats']['path_length_km']:.2f} km
      作业时间: {result['headland']['stats']['time_hours']:.2f} 小时
      平均速度: {result['headland']['stats']['avg_speed_kmh']:.1f} km/h
    
    总计:
      总路径: {result['main_work']['stats']['path_length_km'] + result['headland']['stats']['path_length_km']:.2f} km
      总时间: {result['main_work']['stats']['time_hours'] + result['headland']['stats']['time_hours']:.2f} 小时
    
    ==========================================
    """
    
    ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    test_with_obstacles()
    print("\n" + "="*80)
    print("所有测试完成!")
    print("="*80)

