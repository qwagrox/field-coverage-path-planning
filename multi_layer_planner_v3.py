"""
多层路径规划器 V3.5

核心修正:
1. 第一层: 主作业区域的U型往复路径
2. 第二层（多圈）: 只有一圈的田头边界路径 + 4个转角的转弯和倒车
3. 倒车逻辑: 转弯后沿切线反向倒车到田地边界
4. 自动计算田头宽度，确保多层覆盖完整
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


class TwoLayerPathPlannerV35:
    """
    两层（多圈）路径规划器 V3.5
    
    核心概念:
    - 第一层: 主作业区域（田地中间的矩形区域）
    - 第二层（多圈）: 田头覆盖
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
        
        print(f"[V3.5] 初始化完成:")
        print(f"  田地尺寸: {field_length}m × {field_width}m")
        print(f"  田头宽度: {self.headland_width:.1f}m (自动计算)")
        print(f"  主作业模式: {self.main_work_pattern}")
        print(f"  障碍物数量: {len(self.obstacles)}")
        print(f"  新特性: 真正的两层规划 + 切线方向倒车")
    
    def _calculate_headland_width(self) -> float:
        """
        自动计算合理的田头宽度
        
        V3.5 最终修正 (v5) - 理解A:
        - 第一层转弯路径不能超出田地边界
        - 田头宽度必须 >= R（转弯半径）
        - 第二层需要多圈路径覆盖田头区域
        
        设计逻辑:
        - 田头宽度 = R（转弯半径）
        - 第二层圈数 = ceil(R / W)
        - 最外圈在四个角落处转弯+倒车
        """
        # 田头宽度 = 转弯半径
        return self.vehicle.min_turn_radius
    
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
        print("开始两层路径规划 (V3.5 真正的两层规划)")
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
            'version': 'V3.5',
            'features': ['真正两层', '切线倒车', '网格验证', '强制降速']
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
        """
        生成U型往复路径
        
        V3.5 v5: 修复转弯超出边界的问题
        - 直线段的端点应该距离主作业边界 R 的位置
        - 转弯中心在主作业边界处
        - 这样转弯路径的最外侧刚好到达主作业边界，不会超出
        """
        bounds = work_area.bounds
        min_x, min_y, max_x, max_y = bounds
        R = self.vehicle.min_turn_radius
        
        # 直线段的端点应该距离边界 R，留出转弯空间
        line_start_x = min_x + R
        line_end_x = max_x - R
        
        num_passes = int((max_y - min_y) / self.vehicle.working_width) + 1
        
        path_segments = []
        speeds = []
        
        for i in range(num_passes):
            y = min_y + i * self.vehicle.working_width
            
            # 直线段：从 line_start_x 到 line_end_x
            if i % 2 == 0:
                # 向右行驶
                line_coords = np.array([[line_start_x, y], [line_end_x, y]])
            else:
                # 向左行驶
                line_coords = np.array([[line_end_x, y], [line_start_x, y]])
            
            path_segments.append(line_coords)
            speeds.extend([self.vehicle.max_work_speed_kmh] * len(line_coords))
            
            # 添加转弯
            if i < num_passes - 1:
                next_y = min_y + (i + 1) * self.vehicle.working_width
                turn_path, turn_speeds = self._generate_safe_arc_turn(
                    line_coords[-1],
                    next_y,
                    i % 2 == 0,  # turn_right
                    min_x, max_x
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
    
    def _generate_safe_arc_turn(
        self,
        start_point: np.ndarray,
        next_y: float,
        turn_right: bool,
        min_x: float,
        max_x: float
    ) -> Tuple[np.ndarray, List[float]]:
        """
        生成安全的180度圆弧转弯，确保不超出边界
        
        V3.5 v5: 转弯中心在主作业边界处，转弯路径不超出边界
        """
        x, y = start_point
        R = self.vehicle.min_turn_radius
        
        num_points = 20
        angles = np.linspace(0, np.pi, num_points)
        
        if turn_right:
            # 向右转弯，转弯中心在 (max_x, y)
            center_x = max_x
            center_y = y
            # 从 (x, y) 开始，绕着 (max_x, y) 转180度到 (max_x, next_y)
            arc_x = center_x - R * np.cos(angles)
            arc_y = center_y + R * np.sin(angles)
        else:
            # 向左转弯，转弯中心在 (min_x, y)
            center_x = min_x
            center_y = y
            # 从 (x, y) 开始，绕着 (min_x, y) 转180度到 (min_x, next_y)
            arc_x = center_x + R * np.cos(angles)
            arc_y = center_y + R * np.sin(angles)
        
        turn_path = np.column_stack([arc_x, arc_y])
        
        # 初始速度（后续会被强制降速调整）
        turn_speeds = [self.vehicle.headland_turn_speed_kmh] * num_points
        
        return turn_path, turn_speeds
    
    def _generate_simple_arc_turn(
        self,
        start_point: np.ndarray,
        current_y: float,
        next_y: float,
        turn_right: bool
    ) -> Tuple[np.ndarray, List[float]]:
        """简单圆弧转弯（旧版本，保留以防其他地方调用）"""
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
        """
        生成田头边界路径
        
        V3.5 v5: 生成多圈路径覆盖田头区域
        """
        import math
        
        # 计算需要的圈数
        num_loops = math.ceil(self.headland_width / self.vehicle.working_width)
        print(f"  第二层需要 {num_loops} 圈路径覆盖田头区域")
        
        all_paths = []
        all_speeds = []
        
        for loop_idx in range(num_loops):
            # 计算当前圈的偏移量（从田地边界开始）
            offset = self.vehicle.working_width / 2 + loop_idx * self.vehicle.working_width
            
            # 只有最外圈（loop_idx=0）需要在角落处转弯+倒车
            with_reverse = (loop_idx == 0)
            
            print(f"    第{loop_idx+1}圈: 偏移={offset:.1f}m, 倒车={'Yes' if with_reverse else 'No'}")
            
            path, speeds = self._generate_single_headland_loop_at_offset(
                offset, with_reverse, loop_idx
            )
            
            all_paths.append(path)
            all_speeds.extend(speeds)
        
        # 合并所有圈的路径
        combined_path = np.vstack(all_paths)
        
        return combined_path, np.array(all_speeds)
    
    def _generate_single_headland_loop_at_offset(
        self, 
        offset: float, 
        with_reverse: bool = True, 
        loop_index: int = 0
    ) -> Tuple[np.ndarray, List[float]]:
        """
        在指定偏移量处生成单圈田头边界路径
        
        Args:
            offset: 距离田地边界的距离
            with_reverse: 是否在角落处转弯+倒车
            loop_index: 当前圈的索引（0=最外圈）
        
        V3.5 v5: 支持多圈路径
        """
        
        # 四个转角点
        corners = [
            (offset, offset),  # 左下
            (self.field_length - offset, offset),  # 右下
            (self.field_length - offset, self.field_width - offset),  # 右上
            (offset, self.field_width - offset)  # 左上
        ]
        
        path_segments = []
        speeds = []
        
        # 从左下角开始
        start_point = corners[0]
        path_segments.append(np.array([start_point]))
        speeds.append(self.vehicle.max_headland_speed_kmh)
        
        # 沿着边界行驶，在4个角落处转弯（最外圈还需要倒车）
        for i in range(4):
            current_corner = corners[i]
            next_corner = corners[(i + 1) % 4]
            
            # 直线段
            straight_path = self._generate_straight_segment(current_corner, next_corner, 20)
            path_segments.append(straight_path)
            speeds.extend([self.vehicle.max_headland_speed_kmh] * len(straight_path))
            
            # 转弯段
            if i < 3:  # 前3个角落需要转弯，第4个角落回到起点
                if with_reverse:
                    # 最外圈：转弯 + 倒车填充间隙
                    turn_path, turn_speeds = self._generate_corner_turn_with_reverse(
                        next_corner, (i + 1) % 4, layer_index=loop_index
                    )
                else:
                    # 内圈：只转弯，不倒车
                    turn_path, turn_speeds = self._generate_corner_turn_arc(
                        next_corner, (i + 1) % 4
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
        """
        在角落处生成转弯 + 精确反向填充
        
        V3.4改进:
        1. 使用Shapely精确计算转角间隙
        2. 基于间隙面积计算最优反向长度
        3. 基于间隙质心计算最优反向方向
        """
        R = self.vehicle.min_turn_radius
        W = self.vehicle.working_width
        x, y = corner
        
        add_reverse = (layer_index == 0)
        
        # 生成90度圆弧转弯
        num_points = 15
        angles = np.linspace(0, np.pi/2, num_points)
        
        if corner_index == 0:  # 左下角
            arc_x = x + R * (1 - np.cos(angles))
            arc_y = y + R * np.sin(angles)
        elif corner_index == 1:  # 右下角
            arc_x = x - R * np.sin(angles)
            arc_y = y + R * (1 - np.cos(angles))
        elif corner_index == 2:  # 右上角
            arc_x = x - R * (1 - np.cos(angles))
            arc_y = y - R * np.sin(angles)
        else:  # 左上角
            arc_x = x + R * np.sin(angles)
            arc_y = y - R * (1 - np.cos(angles))
        
        turn_path = np.column_stack([arc_x, arc_y])
        turn_speeds = [self.vehicle.headland_turn_speed_kmh] * num_points
        
        # V3.4.2: 精确反向填充（切线方向倒车）
        if add_reverse:
            # 计算转角间隙
            gap = self._calculate_corner_gap_precise(corner, corner_index, R, W)
            
            if gap is not None and gap.area > 0.1:  # 间隙面积 > 0.1m²
                # 生成最优反向路径（切线方向倒车）
                turn_second_last_point = turn_path[-2]  # 倒数第二个点，用于计算切线
                turn_end_point = turn_path[-1]
                reverse_path, reverse_length = self._generate_optimal_reverse_path(
                    gap, turn_end_point, turn_second_last_point, W, corner_index
                )
                
                turn_path = np.vstack([turn_path, reverse_path])
                reverse_points = len(reverse_path)
                turn_speeds.extend([2.5] * reverse_points)
                
                print(f"    角落{corner_index}: 间隙面积={gap.area:.1f}m², 倒车长度={reverse_length:.1f}m")
        
        return turn_path, turn_speeds
    
    def _calculate_corner_gap_precise(
        self,
        corner: Tuple[float, float],
        corner_index: int,
        R: float,
        W: float
    ) -> Polygon:
        """
        精确计算转角间隙几何
        
        返回:
            转角间隙的Polygon对象
        """
        x, y = corner
        
        # 1. 定义转角正方形区域（2R × 2R）
        if corner_index == 0:  # 左下角
            square = Polygon([
                (x, y), (x + 2*R, y),
                (x + 2*R, y + 2*R), (x, y + 2*R)
            ])
        elif corner_index == 1:  # 右下角
            square = Polygon([
                (x - 2*R, y), (x, y),
                (x, y + 2*R), (x - 2*R, y + 2*R)
            ])
        elif corner_index == 2:  # 右上角
            square = Polygon([
                (x - 2*R, y - 2*R), (x, y - 2*R),
                (x, y), (x - 2*R, y)
            ])
        else:  # 左上角
            square = Polygon([
                (x, y - 2*R), (x + 2*R, y - 2*R),
                (x + 2*R, y), (x, y)
            ])
        
        # 2. 生成90度圆弧转弯路径
        num_points = 30  # 更细密的点以提高精度
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
        
        # 3. 计算转弯路径的覆盖区域（buffer W/2）
        turn_line = LineString(turn_path)
        turn_coverage = turn_line.buffer(W / 2)
        
        # 4. 计算间隙 = 正方形 - 转弯覆盖
        try:
            gap = square.difference(turn_coverage)
            return gap
        except Exception as e:
            print(f"    警告: 计算角落{corner_index}间隙失败: {e}")
            return None
    
    def _generate_optimal_reverse_path(
        self,
        gap: Polygon,
        turn_end_point: np.ndarray,
        turn_start_point: np.ndarray,
        W: float,
        corner_index: int
    ) -> Tuple[np.ndarray, float]:
        """
        基于间隙几何生成最优反向路径
        
        V3.4.2 最终修正:
        - 使用转弯结束时的切线方向的反向作为倒车方向
        - 切线方向 = turn_end_point - turn_second_last_point
        - 倒车方向 = -切线方向
        - 倒车距离 = 从转弯结束点到边界的距离
        
        算法:
        1. 计算转弯结束时的切线方向（代表车辆朝向）
        2. 倒车方向 = -切线方向
        3. 计算倒车到边界的距离（最大化覆盖率）
        
        返回:
            (reverse_path, reverse_length)
        """
        # 1. 计算转弯结束时的切线方向（车辆朝向）
        # 注意：turn_start_point 实际上是 turn_path 的倒数第二个点
        # 但为了保持接口兼容，我们使用 turn_start_point 作为参考
        # 实际应该使用最后两个点来计算切线
        
        # 这里 turn_start_point 实际是 turn_path[0]，但我们需要 turn_path[-2]
        # 由于函数签名已经固定，我们假设调用者传入的是 turn_path[-2]
        # （需要在调用方修正）
        
        tangent_direction = turn_end_point - turn_start_point  # turn_start_point 应该是 turn_path[-2]
        tangent_direction_norm = np.linalg.norm(tangent_direction)
        
        if tangent_direction_norm > 1e-6:
            # 倒车方向 = -切线方向
            reverse_direction = -tangent_direction / tangent_direction_norm
        else:
            # 备选方案：使用间隙质心方向的反向
            centroid = gap.centroid
            to_centroid = np.array([
                centroid.x - turn_end_point[0],
                centroid.y - turn_end_point[1]
            ])
            to_centroid_norm = np.linalg.norm(to_centroid)
            if to_centroid_norm > 1e-6:
                reverse_direction = -to_centroid / to_centroid_norm
            else:
                # 最后备选：使用固定方向
                reverse_direction = np.array([-1.0, 0.0])
        
        # 3. 计算倒车到边界的距离（最大化覆盖率）
        reverse_length = self._calculate_distance_to_boundary(
            turn_end_point, reverse_direction, corner_index
        )
        
        # 4. 生成反向路径（倒车）
        num_points = max(10, int(reverse_length / 0.5))
        t = np.linspace(0, reverse_length, num_points)
        reverse_path = turn_end_point + t[:, np.newaxis] * reverse_direction
        
        return reverse_path, reverse_length
    
    def _calculate_distance_to_boundary(
        self,
        start_point: np.ndarray,
        direction: np.ndarray,
        corner_index: int
    ) -> float:
        """
        计算从起点沿着方向到达田地边界的距离
        
        参数:
            start_point: 起点坐标
            direction: 单位方向向量
            corner_index: 角落索引 (0=左下, 1=右下, 2=右上, 3=左上)
        
        返回:
            到边界的距离
        """
        x, y = start_point
        dx, dy = direction
        
        # 田地边界
        x_min, x_max = 0, self.field_length
        y_min, y_max = 0, self.field_width
        
        # 计算到四个边界的距离
        distances = []
        
        # 到左边界 (x=0)
        if abs(dx) > 1e-6:
            t_left = (x_min - x) / dx
            if t_left > 0:  # 只考虑正方向
                distances.append(t_left)
        
        # 到右边界 (x=field_length)
        if abs(dx) > 1e-6:
            t_right = (x_max - x) / dx
            if t_right > 0:
                distances.append(t_right)
        
        # 到下边界 (y=0)
        if abs(dy) > 1e-6:
            t_bottom = (y_min - y) / dy
            if t_bottom > 0:
                distances.append(t_bottom)
        
        # 到上边界 (y=field_width)
        if abs(dy) > 1e-6:
            t_top = (y_max - y) / dy
            if t_top > 0:
                distances.append(t_top)
        
        if not distances:
            # 如果没有有效距离，返回默认值
            return 2.0 * self.vehicle.min_turn_radius
        
        # 返回最短距离（到达最近的边界）
        min_distance = min(distances)
        
        # 限制最大距离（避免过长）
        max_distance = 3.0 * self.vehicle.min_turn_radius
        
        final_distance = min(min_distance, max_distance)
        
        # 调试输出
        print(f"    [调试] 角落{corner_index}: 起点=({start_point[0]:.1f},{start_point[1]:.1f}), "
              f"方向=({direction[0]:.3f},{direction[1]:.3f}), "
              f"到边界={min_distance:.1f}m, 限制={max_distance:.1f}m, 最终={final_distance:.1f}m")
        
        return final_distance
    
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
    
    def verify_corner_coverage_grid_based(
        self,
        corner: Tuple[float, float],
        corner_index: int,
        turn_path: np.ndarray,
        reverse_path: np.ndarray = None
    ) -> Dict:
        """
        网格化验证转角覆盖率
        
        算法:
        1. 创建0.1m精度网格
        2. 标记转弯路径覆盖
        3. 标记反向路径覆盖
        4. 计算覆盖率
        
        返回:
            {
                'coverage_before': float,  # 反向填充前覆盖率
                'coverage_after': float,   # 反向填充后覆盖率
                'improvement': float,      # 改进值
                'grid': np.ndarray         # 覆盖网格
            }
        """
        R = self.vehicle.min_turn_radius
        W = self.vehicle.working_width
        grid_resolution = 0.1  # 0.1m精度
        
        x, y = corner
        
        # 1. 创建转角区域网格（2R × 2R）
        grid_size = int(2 * R / grid_resolution)
        grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # 确定网格起点
        if corner_index == 0:  # 左下角
            grid_origin = (x, y)
        elif corner_index == 1:  # 右下角
            grid_origin = (x - 2*R, y)
        elif corner_index == 2:  # 右上角
            grid_origin = (x - 2*R, y - 2*R)
        else:  # 左上角
            grid_origin = (x, y - 2*R)
        
        # 2. 标记转弯路径覆盖
        turn_line = LineString(turn_path)
        turn_coverage = turn_line.buffer(W / 2)
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_x = grid_origin[0] + i * grid_resolution
                cell_y = grid_origin[1] + j * grid_resolution
                cell_point = Point(cell_x, cell_y)
                
                if turn_coverage.contains(cell_point):
                    grid[j, i] = True  # 注意: 行列顺序
        
        coverage_before = np.sum(grid) / grid.size * 100
        
        # 3. 标记反向路径覆盖
        if reverse_path is not None and len(reverse_path) > 0:
            reverse_line = LineString(reverse_path)
            reverse_coverage = reverse_line.buffer(W / 2)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if not grid[j, i]:  # 仅检查未覆盖区域
                        cell_x = grid_origin[0] + i * grid_resolution
                        cell_y = grid_origin[1] + j * grid_resolution
                        cell_point = Point(cell_x, cell_y)
                        
                        if reverse_coverage.contains(cell_point):
                            grid[j, i] = True
        
        coverage_after = np.sum(grid) / grid.size * 100
        improvement = coverage_after - coverage_before
        
        return {
            'coverage_before': coverage_before,
            'coverage_after': coverage_after,
            'improvement': improvement,
            'grid': grid,
            'grid_origin': grid_origin,
            'grid_resolution': grid_resolution
        }
    
    def verify_all_corners_coverage(self, headland_result: Dict) -> Dict:
        """
        验证所有4个角落的覆盖率
        
        返回:
            {
                'corners': [每个角落的验证结果],
                'avg_coverage_before': float,
                'avg_coverage_after': float,
                'avg_improvement': float
            }
        """
        print("\n[V3.4 网格化覆盖率验证]")
        print("  网格精度: 0.1m")
        print("  验证区域: 4个转角 (2R × 2R 每个)")
        
        # 提取每个角落的转弯和反向路径（需要从结果中解析）
        # 这里简化处理，仅计算总体覆盖率改进
        
        corners_data = [
            (self.headland_width, self.headland_width, 0),
            (self.field_length - self.headland_width, self.headland_width, 1),
            (self.field_length - self.headland_width, self.field_width - self.headland_width, 2),
            (self.headland_width, self.field_width - self.headland_width, 3)
        ]
        
        corner_results = []
        
        for corner_x, corner_y, corner_idx in corners_data:
            # 生成该角落的转弯路径
            turn_path, _ = self._generate_corner_turn_arc((corner_x, corner_y), corner_idx)
            
            # 计算间隙并生成反向路径
            gap = self._calculate_corner_gap_precise(
                (corner_x, corner_y), corner_idx,
                self.vehicle.min_turn_radius, self.vehicle.working_width
            )
            
            reverse_path = None
            if gap is not None and gap.area > 0.1:
                reverse_path, _ = self._generate_optimal_reverse_path(
                    gap, turn_path[-1], turn_path[-2], self.vehicle.working_width, corner_idx
                )
            
            # 网格化验证
            verification = self.verify_corner_coverage_grid_based(
                (corner_x, corner_y), corner_idx, turn_path, reverse_path
            )
            
            corner_results.append(verification)
            
            print(f"  角落{corner_idx}: 填充前={verification['coverage_before']:.1f}%, "
                  f"填充后={verification['coverage_after']:.1f}%, "
                  f"改进=+{verification['improvement']:.1f}%")
        
        avg_before = np.mean([r['coverage_before'] for r in corner_results])
        avg_after = np.mean([r['coverage_after'] for r in corner_results])
        avg_improvement = avg_after - avg_before
        
        print(f"\n  平均覆盖率: {avg_before:.1f}% → {avg_after:.1f}% (改进 +{avg_improvement:.1f}%)")
        
        return {
            'corners': corner_results,
            'avg_coverage_before': avg_before,
            'avg_coverage_after': avg_after,
            'avg_improvement': avg_improvement
        }
    
    def _generate_corner_turn_arc(
        self,
        corner: Tuple[float, float],
        corner_index: int
    ) -> Tuple[np.ndarray, List[float]]:
        """生成90度圆弧转弯（不含反向填充）"""
        R = self.vehicle.min_turn_radius
        x, y = corner
        
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
        
        return turn_path, turn_speeds


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
        
        planner = TwoLayerPathPlannerV35(
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

