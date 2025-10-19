"""
两层路径规划器 V3.3 - 强制降速版

新增功能:
1. 转弯处强制降速（基于曲率自适应）
2. 完整的速度规划验证
3. 确保满足曲率约束
4. 多场景测试支持

版本历史:
- V3.0: 支持静态障碍物
- V3.1: 集成OptimizedClothoid（有问题）
- V3.2: 修复田头覆盖算法（99.3%覆盖率）
- V3.3: 强制降速+多场景测试（本版本）
"""

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import time


@dataclass
class VehicleParams:
    """车辆参数"""
    working_width: float = 3.2
    min_turn_radius: float = 8.0
    max_work_speed_kmh: float = 9.0
    max_headland_speed_kmh: float = 15.0
    headland_turn_speed_kmh: float = 4.0
    max_lateral_accel: float = 2.0
    max_longitudinal_accel: float = 1.5
    safety_factor: float = 0.85


class TwoLayerPathPlannerV33:
    """
    两层路径规划器 V3.3 - 强制降速版
    
    核心改进:
    1. 基于曲率的自适应降速
    2. 确保满足曲率约束
    3. 完整的速度规划验证
    """
    
    def __init__(
        self,
        field_length: float,
        field_width: float,
        vehicle_params: VehicleParams,
        obstacles: List[List[Tuple[float, float]]] = None
    ):
        self.field_length = field_length
        self.field_width = field_width
        self.vehicle = vehicle_params
        self.obstacles = obstacles or []
        
        # 自动计算田头宽度
        self.headland_width = self._calculate_headland_width()
        
        # 自动选择主作业模式
        self.main_work_pattern = self._select_main_work_pattern()
        
        print(f"[V3.3] 初始化完成:")
        print(f"  田地尺寸: {field_length}m × {field_width}m")
        print(f"  田头宽度: {self.headland_width:.1f}m (自动计算)")
        print(f"  主作业模式: {self.main_work_pattern}")
        print(f"  障碍物数量: {len(self.obstacles)}")
        print(f"  新特性: 基于曲率的强制降速")
    
    def _calculate_headland_width(self) -> float:
        """自动计算合理的田头宽度"""
        min_width = 2.0 * self.vehicle.min_turn_radius
        recommended_width = min_width + 2.0 * self.vehicle.working_width
        max_width = min(self.field_length, self.field_width) * 0.15
        return min(recommended_width, max_width)
    
    def _select_main_work_pattern(self) -> str:
        """自动选择主作业模式"""
        aspect_ratio = self.field_length / self.field_width
        if aspect_ratio > 3.0:
            return "U型往复"
        elif aspect_ratio < 1.5:
            return "Ω型跨行"
        else:
            return "U型往复"
    
    def plan_complete_coverage(self) -> Dict:
        """完整的两层路径规划"""
        print("\n" + "="*70)
        print("开始两层路径规划 (V3.3 强制降速版)")
        print("="*70)
        
        start_time = time.time()
        
        # 第1层：主作业区域
        print("\n[第1层] 主作业区域规划...")
        main_work_result = self._plan_main_work_area()
        
        # 第2层：田头覆盖
        print("\n[第2层] 田头区域规划...")
        headland_result = self._plan_headland_coverage()
        
        # 合并路径并应用强制降速
        print("\n[速度规划] 应用基于曲率的强制降速...")
        all_path = np.vstack([main_work_result['path'], headland_result['path']])
        all_speeds = np.concatenate([main_work_result['speeds'], headland_result['speeds']])
        
        # 计算曲率并调整速度
        adjusted_speeds = self._apply_curvature_based_speed_limit(all_path, all_speeds)
        
        # 更新速度
        main_len = len(main_work_result['path'])
        main_work_result['speeds'] = adjusted_speeds[:main_len]
        headland_result['speeds'] = adjusted_speeds[main_len:]
        
        # 重新计算时间
        main_work_result['stats']['time_hours'] = self._calculate_work_time(
            main_work_result['path'], 
            main_work_result['speeds']
        ) / 3600
        
        headland_result['stats']['time_hours'] = self._calculate_work_time(
            headland_result['path'],
            headland_result['speeds']
        ) / 3600
        
        total_time = time.time() - start_time
        
        result = {
            'main_work': main_work_result,
            'headland': headland_result,
            'total_time': total_time,
            'version': 'V3.3',
            'features': ['强制降速', '曲率约束满足']
        }
        
        print(f"\n{'='*70}")
        print(f"路径规划完成! 总耗时: {total_time:.3f}秒")
        print(f"{'='*70}")
        
        return result
    
    def _apply_curvature_based_speed_limit(
        self,
        path: np.ndarray,
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        基于曲率的强制降速
        
        核心算法:
        v_max = sqrt(a_lat_max / kappa) * safety_factor
        
        确保: a_lat = v^2 * kappa <= a_lat_max
        """
        if len(path) < 3:
            return speeds
        
        adjusted_speeds = speeds.copy()
        max_lateral_accel = self.vehicle.max_lateral_accel
        safety_factor = self.vehicle.safety_factor
        
        speed_adjustments = 0
        
        # 计算每个点的曲率
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # 计算曲率
            kappa = self._calculate_curvature(p1, p2, p3)
            
            if kappa > 1e-6:  # 有曲率
                # 计算最大允许速度（m/s）
                v_max_ms = np.sqrt(max_lateral_accel / kappa) * safety_factor
                v_max_kmh = v_max_ms * 3.6
                
                # 强制降速
                if adjusted_speeds[i] > v_max_kmh:
                    adjusted_speeds[i] = v_max_kmh
                    speed_adjustments += 1
        
        print(f"  速度调整点数: {speed_adjustments}/{len(path)} ({speed_adjustments/len(path)*100:.1f}%)")
        
        # 平滑速度（考虑加速度约束）
        adjusted_speeds = self._smooth_speed_profile(path, adjusted_speeds)
        
        return adjusted_speeds
    
    def _calculate_curvature(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> float:
        """计算三点的曲率"""
        dx1, dy1 = p2 - p1
        dx2, dy2 = p3 - p2
        
        ds1 = np.sqrt(dx1**2 + dy1**2)
        ds2 = np.sqrt(dx2**2 + dy2**2)
        
        if ds1 < 1e-6 or ds2 < 1e-6:
            return 0.0
        
        theta1 = np.arctan2(dy1, dx1)
        theta2 = np.arctan2(dy2, dx2)
        dtheta = theta2 - theta1
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        kappa = abs(2 * dtheta / (ds1 + ds2))
        
        return kappa
    
    def _smooth_speed_profile(
        self,
        path: np.ndarray,
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        平滑速度曲线，考虑加速度约束
        
        三遍算法:
        1. 前向传播（加速度约束）
        2. 后向传播（减速度约束）
        3. 最终平滑
        """
        if len(path) < 2:
            return speeds
        
        smoothed = speeds.copy()
        max_accel = self.vehicle.max_longitudinal_accel  # m/s²
        
        # 第1遍：前向传播（加速度约束）
        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i] - path[i-1])
            if dist < 1e-6:
                continue
            
            v1_ms = smoothed[i-1] / 3.6
            v2_ms = smoothed[i] / 3.6
            
            # 最大可达速度
            v_max_ms = np.sqrt(v1_ms**2 + 2 * max_accel * dist)
            v_max_kmh = v_max_ms * 3.6
            
            if smoothed[i] > v_max_kmh:
                smoothed[i] = v_max_kmh
        
        # 第2遍：后向传播（减速度约束）
        for i in range(len(path) - 2, -1, -1):
            dist = np.linalg.norm(path[i+1] - path[i])
            if dist < 1e-6:
                continue
            
            v2_ms = smoothed[i+1] / 3.6
            v1_ms = smoothed[i] / 3.6
            
            # 最大可达速度
            v_max_ms = np.sqrt(v2_ms**2 + 2 * max_accel * dist)
            v_max_kmh = v_max_ms * 3.6
            
            if smoothed[i] > v_max_kmh:
                smoothed[i] = v_max_kmh
        
        return smoothed
    
    def _plan_main_work_area(self) -> Dict:
        """规划主作业区域（第1层）"""
        # 定义主作业区域边界
        main_boundary = Polygon([
            (self.headland_width, self.headland_width),
            (self.field_length - self.headland_width, self.headland_width),
            (self.field_length - self.headland_width, self.field_width - self.headland_width),
            (self.headland_width, self.field_width - self.headland_width)
        ])
        
        # 处理障碍物
        if self.obstacles:
            obstacles_polygons = []
            for obs in self.obstacles:
                obs_poly = Polygon(obs)
                expanded = obs_poly.buffer(self.vehicle.working_width / 2)
                obstacles_polygons.append(expanded)
            
            obstacles_union = unary_union(obstacles_polygons)
            main_work_area = main_boundary.difference(obstacles_union)
        else:
            main_work_area = main_boundary
        
        # 生成U型往复路径
        path, speeds = self._generate_u_pattern_path(main_work_area)
        
        path_length = self._calculate_path_length(path)
        work_time = self._calculate_work_time(path, speeds)
        
        return {
            'path': path,
            'speeds': speeds,
            'pattern': self.main_work_pattern,
            'area': main_work_area,
            'stats': {
                'path_length_km': path_length / 1000,
                'time_hours': work_time / 3600,
                'avg_speed_kmh': (path_length / 1000) / (work_time / 3600) if work_time > 0 else 0
            }
        }
    
    def _generate_u_pattern_path(self, work_area: Polygon) -> Tuple[np.ndarray, np.ndarray]:
        """生成U型往复路径"""
        bounds = work_area.bounds
        min_x, min_y, max_x, max_y = bounds
        
        num_passes = int((max_y - min_y) / self.vehicle.working_width) + 1
        
        path_segments = []
        speeds = []
        
        for i in range(num_passes):
            y = min_y + i * self.vehicle.working_width
            
            if i % 2 == 0:
                line = LineString([(min_x, y), (max_x, y)])
            else:
                line = LineString([(max_x, y), (min_x, y)])
            
            intersection = work_area.intersection(line)
            
            if intersection.is_empty:
                continue
            
            if isinstance(intersection, LineString):
                coords = np.array(intersection.coords)
                path_segments.append(coords)
                speeds.extend([self.vehicle.max_work_speed_kmh] * len(coords))
            
            # 添加转弯
            if i < num_passes - 1:
                if len(path_segments) > 0:
                    turn_path, turn_speeds = self._generate_simple_arc_turn(
                        path_segments[-1][-1],
                        y,
                        y + self.vehicle.working_width,
                        i % 2 == 0
                    )
                    path_segments.append(turn_path)
                    speeds.extend(turn_speeds)
        
        if path_segments:
            path = np.vstack(path_segments)
            speeds = np.array(speeds)
        else:
            path = np.array([[0, 0]])
            speeds = np.array([0])
        
        return path, speeds
    
    def _generate_simple_arc_turn(
        self,
        start_point: np.ndarray,
        current_y: float,
        next_y: float,
        turn_right: bool
    ) -> Tuple[np.ndarray, List[float]]:
        """简单圆弧转弯"""
        x, y = start_point
        R = self.vehicle.min_turn_radius
        
        num_points = 20
        angles = np.linspace(0, np.pi, num_points)
        
        if turn_right:
            arc_x = x + R * np.sin(angles)
            arc_y = y - R * (1 - np.cos(angles))
        else:
            arc_x = x - R * np.sin(angles)
            arc_y = y - R * (1 - np.cos(angles))
        
        turn_path = np.column_stack([arc_x, arc_y])
        
        # 初始速度（后续会被强制降速调整）
        turn_speeds = [self.vehicle.headland_turn_speed_kmh] * num_points
        
        return turn_path, turn_speeds
    
    def _plan_headland_coverage(self) -> Dict:
        """规划田头覆盖区域"""
        # 田头区域
        field_boundary = Polygon([
            (0, 0),
            (self.field_length, 0),
            (self.field_length, self.field_width),
            (0, self.field_width)
        ])
        
        main_boundary = Polygon([
            (self.headland_width, self.headland_width),
            (self.field_length - self.headland_width, self.headland_width),
            (self.field_length - self.headland_width, self.field_width - self.headland_width),
            (self.headland_width, self.field_width - self.headland_width)
        ])
        
        headland_area = field_boundary.difference(main_boundary)
        
        # 生成多层环绕路径
        path, speeds = self._generate_multilayer_headland(headland_area)
        
        path_length = self._calculate_path_length(path)
        work_time = self._calculate_work_time(path, speeds)
        coverage_rate = self._calculate_coverage_rate(path, headland_area)
        
        return {
            'path': path,
            'speeds': speeds,
            'area': headland_area,
            'stats': {
                'path_length_km': path_length / 1000,
                'time_hours': work_time / 3600,
                'avg_speed_kmh': (path_length / 1000) / (work_time / 3600) if work_time > 0 else 0,
                'coverage_rate': coverage_rate
            }
        }
    
    def _generate_multilayer_headland(
        self,
        headland_area: Polygon
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成多层田头环绕路径"""
        layer_spacing = self.vehicle.working_width * 0.85
        num_layers = int(self.headland_width / layer_spacing) + 1
        num_layers = max(1, min(num_layers, 10))
        
        all_path_segments = []
        all_speeds = []
        
        for layer in range(num_layers):
            offset = layer * layer_spacing + self.vehicle.working_width / 2
            
            if offset >= self.headland_width:
                break
            
            layer_path, layer_speeds = self._generate_single_layer_loop(offset, layer)
            
            all_path_segments.append(layer_path)
            all_speeds.extend(layer_speeds)
        
        if all_path_segments:
            path = np.vstack(all_path_segments)
            speeds = np.array(all_speeds)
        else:
            path = np.array([[0, 0]])
            speeds = np.array([0])
        
        return path, speeds
    
    def _generate_single_layer_loop(
        self,
        offset: float,
        layer_index: int
    ) -> Tuple[np.ndarray, List[float]]:
        """生成单层田头环绕路径"""
        corners = [
            (offset, offset),
            (self.field_length - offset, offset),
            (self.field_length - offset, self.field_width - offset),
            (offset, self.field_width - offset)
        ]
        
        path_segments = []
        speeds = []
        
        start_point = corners[0]
        path_segments.append(np.array([start_point]))
        speeds.append(self.vehicle.max_headland_speed_kmh)
        
        for i in range(4):
            current_corner = corners[i]
            next_corner = corners[(i + 1) % 4]
            
            # 直线段
            straight_path = self._generate_straight_segment(current_corner, next_corner, 20)
            path_segments.append(straight_path)
            speeds.extend([self.vehicle.max_headland_speed_kmh] * len(straight_path))
            
            # 转弯段
            if i < 3:
                turn_path, turn_speeds = self._generate_corner_turn_with_reverse(
                    next_corner, i, layer_index
                )
                path_segments.append(turn_path)
                speeds.extend(turn_speeds)
        
        path = np.vstack(path_segments)
        
        return path, speeds
    
    def _generate_straight_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 20
    ) -> np.ndarray:
        """生成直线段"""
        x = np.linspace(start[0], end[0], num_points)
        y = np.linspace(start[1], end[1], num_points)
        return np.column_stack([x, y])
    
    def _generate_corner_turn_with_reverse(
        self,
        corner: Tuple[float, float],
        corner_index: int,
        layer_index: int
    ) -> Tuple[np.ndarray, List[float]]:
        """在角落处生成转弯 + 倒车填补"""
        R = self.vehicle.min_turn_radius
        x, y = corner
        
        add_reverse = (layer_index == 0)
        
        num_points = 15
        angles = np.linspace(0, np.pi/2, num_points)
        
        if corner_index == 0:
            arc_x = x + R * (1 - np.cos(angles))
            arc_y = y + R * np.sin(angles)
        elif corner_index == 1:
            arc_x = x - R * np.sin(angles)
            arc_y = y + R * (1 - np.cos(angles))
        elif corner_index == 2:
            arc_x = x - R * (1 - np.cos(angles))
            arc_y = y - R * np.sin(angles)
        else:
            arc_x = x + R * np.sin(angles)
            arc_y = y - R * (1 - np.cos(angles))
        
        turn_path = np.column_stack([arc_x, arc_y])
        turn_speeds = [self.vehicle.headland_turn_speed_kmh] * num_points
        
        if add_reverse:
            reverse_length = R * 0.8
            reverse_points = 10
            
            end_x, end_y = turn_path[-1]
            start_x, start_y = turn_path[-2]
            
            dx = end_x - start_x
            dy = end_y - start_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                dx /= dist
                dy /= dist
                
                reverse_x = end_x - dx * np.linspace(0, reverse_length, reverse_points)
                reverse_y = end_y - dy * np.linspace(0, reverse_length, reverse_points)
                reverse_path = np.column_stack([reverse_x, reverse_y])
                
                turn_path = np.vstack([turn_path, reverse_path])
                turn_speeds.extend([2.5] * reverse_points)
        
        return turn_path, turn_speeds
    
    def _calculate_path_length(self, path: np.ndarray) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0
        diff = np.diff(path, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        return np.sum(distances)
    
    def _calculate_work_time(self, path: np.ndarray, speeds: np.ndarray) -> float:
        """计算作业时间"""
        if len(path) < 2 or len(speeds) == 0:
            return 0.0
        
        diff = np.diff(path, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        avg_speeds = (speeds[:-1] + speeds[1:]) / 2
        avg_speeds_ms = avg_speeds / 3.6
        avg_speeds_ms = np.maximum(avg_speeds_ms, 0.1)
        
        times = distances / avg_speeds_ms
        return np.sum(times)
    
    def _calculate_coverage_rate(self, path: np.ndarray, area: Polygon) -> float:
        """计算覆盖率"""
        if len(path) < 2:
            return 0.0
        
        path_line = LineString(path)
        path_buffer = path_line.buffer(self.vehicle.working_width / 2)
        
        covered_area = path_buffer.intersection(area).area
        total_area = area.area
        
        if total_area > 0:
            return (covered_area / total_area) * 100
        else:
            return 0.0
    
    def verify_curvature_constraints(self, path: np.ndarray, speeds: np.ndarray) -> Dict:
        """验证曲率约束（考虑降速后的速度）"""
        print("\n[曲率约束验证（降速后）]")
        
        if len(path) < 3:
            return {'max_curvature': 0, 'violations': 0, 'pass': True}
        
        curvatures = []
        lateral_accels = []
        
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            kappa = self._calculate_curvature(p1, p2, p3)
            curvatures.append(kappa)
            
            # 计算实际侧向加速度
            v_ms = speeds[i] / 3.6
            a_lat = v_ms**2 * kappa
            lateral_accels.append(a_lat)
        
        curvatures = np.array(curvatures)
        lateral_accels = np.array(lateral_accels)
        
        max_curvature = np.max(curvatures) if len(curvatures) > 0 else 0
        max_lateral_accel = np.max(lateral_accels) if len(lateral_accels) > 0 else 0
        max_allowed_accel = self.vehicle.max_lateral_accel
        
        # 检查侧向加速度是否超出
        accel_violations = np.sum(lateral_accels > max_allowed_accel)
        accel_violation_rate = (accel_violations / len(lateral_accels) * 100) if len(lateral_accels) > 0 else 0
        
        if len(curvatures) > 1:
            kappa_jumps = np.abs(np.diff(curvatures))
            max_jump = np.max(kappa_jumps)
        else:
            max_jump = 0
        
        print(f"  最大曲率: {max_curvature:.4f}")
        print(f"  最大侧向加速度: {max_lateral_accel:.4f} m/s² (允许: {max_allowed_accel:.4f} m/s²)")
        print(f"  侧向加速度超出: {accel_violations}/{len(lateral_accels)} ({accel_violation_rate:.1f}%)")
        print(f"  最大曲率跳变: {max_jump:.6f}")
        print(f"  状态: {'✅ 通过' if accel_violation_rate < 5 else '⚠️ 部分超出'}")
        
        return {
            'max_curvature': max_curvature,
            'max_lateral_accel': max_lateral_accel,
            'max_allowed_accel': max_allowed_accel,
            'accel_violations': accel_violations,
            'accel_violation_rate': accel_violation_rate,
            'max_jump': max_jump,
            'pass': accel_violation_rate < 5
        }


def run_multi_scenario_tests():
    """运行多场景测试"""
    
    print("="*70)
    print("V3.3 多场景测试")
    print("="*70)
    
    scenarios = [
        {
            'name': '新疆大型麦田',
            'length': 3500,
            'width': 320,
            'obstacles': [[(1500, 100), (1600, 100), (1600, 200), (1500, 200)]]
        },
        {
            'name': '中型田地',
            'length': 500,
            'width': 200,
            'obstacles': [
                [(200, 80), (250, 80), (250, 120), (200, 120)],
                [(350, 140), (380, 140), (380, 170), (350, 170)]
            ]
        },
        {
            'name': '小型田地',
            'length': 100,
            'width': 80,
            'obstacles': []
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"测试场景: {scenario['name']}")
        print(f"{'='*70}")
        
        vehicle = VehicleParams(
            working_width=3.2,
            min_turn_radius=8.0,
            max_work_speed_kmh=9.0,
            max_headland_speed_kmh=15.0
        )
        
        planner = TwoLayerPathPlannerV33(
            field_length=scenario['length'],
            field_width=scenario['width'],
            vehicle_params=vehicle,
            obstacles=scenario['obstacles']
        )
        
        result = planner.plan_complete_coverage()
        
        # 验证曲率约束
        all_path = np.vstack([result['main_work']['path'], result['headland']['path']])
        all_speeds = np.concatenate([result['main_work']['speeds'], result['headland']['speeds']])
        curvature_check = planner.verify_curvature_constraints(all_path, all_speeds)
        
        results.append({
            'scenario': scenario['name'],
            'result': result,
            'curvature_check': curvature_check,
            'planner': planner
        })
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"规划结果 - {scenario['name']}")
        print(f"{'='*70}")
        
        print(f"\n主作业区域:")
        print(f"  路径长度: {result['main_work']['stats']['path_length_km']:.2f} km")
        print(f"  作业时间: {result['main_work']['stats']['time_hours']:.2f} h")
        
        print(f"\n田头区域:")
        print(f"  路径长度: {result['headland']['stats']['path_length_km']:.2f} km")
        print(f"  作业时间: {result['headland']['stats']['time_hours']:.2f} h")
        print(f"  覆盖率: {result['headland']['stats']['coverage_rate']:.2f}%")
        
        total_length = (result['main_work']['stats']['path_length_km'] + 
                       result['headland']['stats']['path_length_km'])
        total_time = (result['main_work']['stats']['time_hours'] + 
                     result['headland']['stats']['time_hours'])
        print(f"\n总计:")
        print(f"  总路径长度: {total_length:.2f} km")
        print(f"  总作业时间: {total_time:.2f} h")
        print(f"  计算时间: {result['total_time']:.3f} s")
    
    # 可视化所有场景
    visualize_multi_scenarios(results)
    
    return results


def visualize_multi_scenarios(results):
    """可视化多场景测试结果"""
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, res in enumerate(results):
        scenario_name = res['scenario']
        result = res['result']
        curvature_check = res['curvature_check']
        planner = res['planner']
        
        # 每个场景2个子图
        # 子图1：完整路径
        ax1 = plt.subplot(3, 3, idx*3 + 1)
        
        field_rect = Rectangle((0, 0), planner.field_length, planner.field_width,
                               fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(field_rect)
        
        main_path = result['main_work']['path']
        if len(main_path) > 0:
            ax1.plot(main_path[:, 0], main_path[:, 1], 'b-', linewidth=1, alpha=0.7, label='主作业')
        
        headland_path = result['headland']['path']
        if len(headland_path) > 0:
            ax1.plot(headland_path[:, 0], headland_path[:, 1], 'r-', linewidth=2, label='田头')
        
        if planner.obstacles:
            for obs in planner.obstacles:
                obs_poly = MplPolygon(obs, fill=True, facecolor='gray', 
                                     edgecolor='black', alpha=0.5)
                ax1.add_patch(obs_poly)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{scenario_name}\n覆盖率: {result["headland"]["stats"]["coverage_rate"]:.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 子图2：速度分布
        ax2 = plt.subplot(3, 3, idx*3 + 2)
        
        all_speeds = np.concatenate([result['main_work']['speeds'], result['headland']['speeds']])
        ax2.hist(all_speeds, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=np.mean(all_speeds), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(all_speeds):.1f} km/h')
        ax2.set_xlabel('速度 (km/h)')
        ax2.set_ylabel('频数')
        ax2.set_title(f'速度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：性能指标
        ax3 = plt.subplot(3, 3, idx*3 + 3)
        ax3.axis('off')
        
        total_length = (result['main_work']['stats']['path_length_km'] + 
                       result['headland']['stats']['path_length_km'])
        total_time = (result['main_work']['stats']['time_hours'] + 
                     result['headland']['stats']['time_hours'])
        
        info_text = f"""
{scenario_name}

田地尺寸:
  {planner.field_length}m × {planner.field_width}m
  {planner.field_length * planner.field_width / 10000:.2f}公顷

路径规划:
  总长度: {total_length:.2f} km
  总时间: {total_time:.2f} h
  田头覆盖率: {result['headland']['stats']['coverage_rate']:.1f}%
  计算时间: {result['total_time']:.3f}s

曲率验证:
  最大侧向加速度: {curvature_check['max_lateral_accel']:.3f} m/s²
  允许值: {curvature_check['max_allowed_accel']:.3f} m/s²
  超出率: {curvature_check['accel_violation_rate']:.1f}%
  状态: {'✅ 通过' if curvature_check['pass'] else '⚠️ 部分超出'}
        """
        
        ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/v33_multi_scenario_results.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: /home/ubuntu/v33_multi_scenario_results.png")


if __name__ == "__main__":
    results = run_multi_scenario_tests()

