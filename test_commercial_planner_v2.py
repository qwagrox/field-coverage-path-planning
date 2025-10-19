"""
商业化两层路径规划器 V2.0 - 完整测试和可视化

测试所有功能:
1. Clothoid曲线
2. 速度规划
3. 曲率约束
4. 电子围栏
5. 自适应参数
6. 多场景测试
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from two_layer_planner_v2 import TwoLayerPathPlannerV2, VehicleParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_complete_plan(result, planner, save_path):
    """完整可视化路径规划结果"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 完整路径图
    ax1 = plt.subplot(2, 3, 1)
    plot_complete_path(ax1, result, planner)
    
    # 2. 速度剖面图
    ax2 = plt.subplot(2, 3, 2)
    plot_speed_profile(ax2, result)
    
    # 3. 曲率剖面图
    ax3 = plt.subplot(2, 3, 3)
    plot_curvature_profile(ax3, result, planner)
    
    # 4. 角落细节图
    ax4 = plt.subplot(2, 3, 4)
    plot_corner_detail(ax4, result, planner)
    
    # 5. 覆盖率分析
    ax5 = plt.subplot(2, 3, 5)
    plot_coverage_analysis(ax5, result)
    
    # 6. 统计信息
    ax6 = plt.subplot(2, 3, 6)
    plot_statistics(ax6, result, planner)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存: {save_path}")
    plt.close()


def plot_complete_path(ax, result, planner):
    """绘制完整路径"""
    
    # 田地边界
    boundary = planner.field_boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='田地边界')
    ax.fill(x, y, color='lightgreen', alpha=0.2)
    
    # 主作业区域边界
    main_area = planner.main_work_area
    x, y = main_area.exterior.xy
    ax.plot(x, y, 'b--', linewidth=1, label='主作业区域')
    
    # 主作业路径（采样显示）
    main_points = result['main_work']['points']
    sample_rate = max(len(main_points) // 500, 1)
    ax.plot(main_points[::sample_rate, 0], main_points[::sample_rate, 1], 
            'b-', linewidth=0.5, alpha=0.5, label='主作业路径')
    
    # 田头路径
    headland_points = result['headland']['points']
    headland_types = result['headland']['types']
    
    # 分类绘制
    forward_mask = np.array([t == 'forward' for t in headland_types])
    turn_mask = np.array([t == 'turn' for t in headland_types])
    reverse_mask = np.array([t == 'reverse' for t in headland_types])
    
    if np.any(forward_mask):
        ax.plot(headland_points[forward_mask, 0], headland_points[forward_mask, 1],
                'g-', linewidth=2, label='田头前进', alpha=0.8)
    
    if np.any(turn_mask):
        ax.plot(headland_points[turn_mask, 0], headland_points[turn_mask, 1],
                'orange', linewidth=2, label='田头转弯', alpha=0.8)
    
    if np.any(reverse_mask):
        ax.plot(headland_points[reverse_mask, 0], headland_points[reverse_mask, 1],
                'r-', linewidth=2, label='倒车填补', alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('完整路径规划 (两层设计)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_speed_profile(ax, result):
    """绘制速度剖面"""
    
    # 主作业速度
    main_points = result['main_work']['points']
    main_speeds = result['main_work']['speeds']
    
    main_distances = np.zeros(len(main_points))
    for i in range(1, len(main_points)):
        main_distances[i] = main_distances[i-1] + np.linalg.norm(main_points[i] - main_points[i-1])
    main_distances /= 1000  # 转换为km
    
    # 田头速度
    headland_points = result['headland']['points']
    headland_speeds = result['headland']['speeds']
    
    headland_distances = np.zeros(len(headland_points))
    for i in range(1, len(headland_points)):
        headland_distances[i] = headland_distances[i-1] + np.linalg.norm(headland_points[i] - headland_points[i-1])
    headland_distances = headland_distances / 1000 + main_distances[-1]  # 接续主作业
    
    # 绘制
    sample_rate = max(len(main_distances) // 1000, 1)
    ax.plot(main_distances[::sample_rate], main_speeds[::sample_rate], 
            'b-', linewidth=1, label='主作业', alpha=0.7)
    ax.plot(headland_distances, headland_speeds, 
            'g-', linewidth=1.5, label='田头', alpha=0.8)
    
    ax.axhline(y=9.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='主作业速度限制')
    ax.axhline(y=15.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='田头速度限制')
    
    ax.set_xlabel('累计距离 (km)')
    ax.set_ylabel('速度 (km/h)')
    ax.set_title('速度剖面 (自适应调整)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 16)


def plot_curvature_profile(ax, result, planner):
    """绘制曲率剖面"""
    
    # 田头曲率（主要关注）
    headland_points = result['headland']['points']
    headland_curvatures = result['headland']['curvatures']
    
    distances = np.zeros(len(headland_points))
    for i in range(1, len(headland_points)):
        distances[i] = distances[i-1] + np.linalg.norm(headland_points[i] - headland_points[i-1])
    
    # 绘制
    ax.plot(distances, headland_curvatures, 'b-', linewidth=1.5, label='实际曲率', alpha=0.8)
    
    # 最大允许曲率
    max_curvature = 1.0 / planner.vehicle.adaptive_turn_radius
    ax.axhline(y=max_curvature, color='r', linestyle='--', linewidth=2, 
               label=f'最大允许 (R={planner.vehicle.adaptive_turn_radius:.1f}m)', alpha=0.7)
    
    ax.set_xlabel('累计距离 (m)')
    ax.set_ylabel('曲率 (1/m)')
    ax.set_title('曲率剖面 (Clothoid平滑)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 标注超出区域
    violations = headland_curvatures > max_curvature * 1.1
    if np.any(violations):
        ax.fill_between(distances, 0, headland_curvatures, 
                        where=violations, color='red', alpha=0.2, label='超出区域')


def plot_corner_detail(ax, result, planner):
    """绘制角落细节（倒车填补）"""
    
    # 选择右下角进行详细展示
    corner_x = planner.field_length - 50
    corner_y = 50
    
    ax.set_xlim(corner_x - 30, corner_x + 30)
    ax.set_ylim(corner_y - 30, corner_y + 30)
    
    # 田地边界
    boundary = planner.field_boundary
    x, y = boundary.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2)
    
    # 田头路径
    headland_points = result['headland']['points']
    headland_types = result['headland']['types']
    
    # 筛选角落附近的点
    mask = (np.abs(headland_points[:, 0] - corner_x) < 30) & \
           (np.abs(headland_points[:, 1] - corner_y) < 30)
    
    corner_points = headland_points[mask]
    corner_types = np.array(headland_types)[mask]
    
    # 分类绘制
    for i, (point, ptype) in enumerate(zip(corner_points, corner_types)):
        if ptype == 'forward':
            color = 'green'
            marker = 'o'
            label = '前进' if i == 0 else ''
        elif ptype == 'turn':
            color = 'orange'
            marker = 's'
            label = '转弯' if i == 0 else ''
        else:  # reverse
            color = 'red'
            marker = '^'
            label = '倒车' if i == 0 else ''
        
        ax.plot(point[0], point[1], marker=marker, color=color, 
                markersize=4, alpha=0.6, label=label if label else None)
    
    # 连线
    if len(corner_points) > 0:
        ax.plot(corner_points[:, 0], corner_points[:, 1], 
                'b-', linewidth=1.5, alpha=0.5)
    
    # 标注
    ax.text(corner_x + 20, corner_y + 20, '右下角\n倒车填补示意', 
            fontsize=10, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('角落细节 (转弯+倒车)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def plot_coverage_analysis(ax, result):
    """绘制覆盖率分析"""
    
    coverage = result['coverage']
    
    categories = ['总面积', '覆盖面积', '角落空隙\n(原始)', '角落空隙\n(填补后)']
    values = [
        coverage['total_area_ha'] * 10000,  # 转换为m²
        coverage['covered_area_ha'] * 10000,
        coverage['corner_gap_m2'],
        coverage['remaining_gap_m2']
    ]
    colors = ['lightblue', 'lightgreen', 'orange', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if value > 1000:
            label = f'{value/10000:.2f} ha'
        else:
            label = f'{value:.1f} m²'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('面积 (m²)')
    ax.set_title(f'覆盖率分析 ({coverage["coverage_rate"]:.2f}%)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加覆盖率文本
    ax.text(0.5, 0.95, f'空隙减少: {coverage["gap_reduction"]:.1f}%',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))


def plot_statistics(ax, result, planner):
    """绘制统计信息"""
    
    ax.axis('off')
    
    # 准备统计文本
    stats_text = f"""
商业化两层路径规划器 V2.0
{'='*50}

【田地参数】
  尺寸: {planner.field_length}m × {planner.field_width}m
  面积: {result['coverage']['total_area_ha']:.2f} 公顷
  长宽比: {planner.field_length/planner.field_width:.2f}

【车辆参数】
  作业幅宽: {planner.vehicle.working_width} m
  基础转弯半径: {planner.vehicle.base_turn_radius} m
  自适应转弯半径: {planner.vehicle.adaptive_turn_radius:.1f} m
  最大作业速度: {planner.vehicle.max_work_speed_kmh} km/h
  最大田头速度: {planner.vehicle.max_headland_speed_kmh} km/h

【自动计算参数】
  田头宽度: {planner.vehicle.headland_width:.1f} m
  主作业模式: {planner.main_work_pattern}

【第1层: 主作业区域】
  作业趟数: {result['main_work']['stats']['num_passes']}
  路径长度: {result['main_work']['stats']['distance_km']:.2f} km
  作业时间: {result['main_work']['stats']['time_hours']:.2f} 小时

【第2层: 外层田头】
  路径长度: {result['headland']['stats']['distance_km']:.2f} km
  作业时间: {result['headland']['stats']['time_hours']:.2f} 小时
  倒车次数: {result['headland']['stats']['num_reverse']}

【总体统计】
  总路径长度: {result['main_work']['stats']['distance_km'] + result['headland']['stats']['distance_km']:.2f} km
  总作业时间: {result['main_work']['stats']['time_hours'] + result['headland']['stats']['time_hours']:.2f} 小时
  作业效率: {result['coverage']['total_area_ha'] / (result['main_work']['stats']['time_hours'] + result['headland']['stats']['time_hours']):.2f} ha/h
  覆盖率: {result['coverage']['coverage_rate']:.2f}%

【约束验证】
  曲率约束: {'✓ 通过' if result['validation']['curvature']['passed'] else '⚠ 部分超出'}
  边界约束: {'✓ 通过' if result['validation']['boundary']['passed'] else '⚠ 部分超出'}
  速度约束: {'✓ 通过' if result['validation']['speed']['passed'] else '⚠ 部分超出'}

【关键特性】
  ✓ Clothoid曲线 (曲率连续)
  ✓ 自适应速度规划
  ✓ 动态转弯半径
  ✓ 电子围栏检查
  ✓ 倒车填补策略
  ✓ 自动参数计算
  ✓ 智能模式选择
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def test_multiple_scenarios():
    """测试多个场景"""
    
    scenarios = [
        {
            'name': '新疆大型麦田',
            'length': 3500,
            'width': 320,
            'working_width': 3.2,
            'turn_radius': 8.0
        },
        {
            'name': '中型田地',
            'length': 500,
            'width': 200,
            'working_width': 2.5,
            'turn_radius': 6.0
        },
        {
            'name': '小型正方形田地',
            'length': 100,
            'width': 100,
            'working_width': 2.0,
            'turn_radius': 5.0
        }
    ]
    
    print("\n" + "="*80)
    print("多场景测试")
    print("="*80)
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"场景: {scenario['name']}")
        print(f"{'='*80}")
        
        vehicle = VehicleParams(
            working_width=scenario['working_width'],
            base_turn_radius=scenario['turn_radius']
        )
        
        planner = TwoLayerPathPlannerV2(
            field_length=scenario['length'],
            field_width=scenario['width'],
            vehicle_params=vehicle
        )
        
        result = planner.plan_complete_coverage()
        
        # 可视化
        save_path = f"/home/ubuntu/commercial_v2_{scenario['name']}.png"
        visualize_complete_plan(result, planner, save_path)
        
        print(f"\n场景 '{scenario['name']}' 测试完成!")


if __name__ == "__main__":
    print("="*80)
    print("商业化两层路径规划器 V2.0 - 完整测试")
    print("="*80)
    
    test_multiple_scenarios()
    
    print("\n" + "="*80)
    print("所有测试完成!")
    print("="*80)

