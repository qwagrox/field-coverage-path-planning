"""
测试V3.5.1版本的智能起点/终点选择功能
"""

from multi_layer_planner_v3 import TwoLayerPathPlannerV35, VehicleParams
import matplotlib.pyplot as plt
import numpy as np

def test_with_start_end_points():
    """测试带起点/终点约束的路径规划"""
    
    # 车辆参数
    vehicle = VehicleParams(
        working_width=3.2,
        min_turn_radius=8.0,
        max_work_speed_kmh=9.0,
        max_headland_speed_kmh=14.0
    )
    
    # 场景：中型田地
    field_length = 500
    field_width = 200
    
    # 定义起点和终点（在田地边界内）
    start_point = (10, 10)  # 左下角附近
    end_point = (490, 190)  # 右上角附近
    
    print("="*80)
    print("V3.5.1 智能起点/终点选择测试")
    print("="*80)
    print(f"田地尺寸: {field_length}m × {field_width}m")
    print(f"起点位置: {start_point}")
    print(f"终点位置: {end_point}")
    print()
    
    # 创建规划器
    planner = TwoLayerPathPlannerV35(
        field_length=field_length,
        field_width=field_width,
        vehicle_params=vehicle,
        start_point=start_point,
        end_point=end_point
    )
    
    # 规划路径
    result = planner.plan_complete_coverage()
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：完整路径
    ax1.set_title('V3.5.1 完整路径（带起点/终点连接）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 绘制田地边界
    ax1.plot([0, field_length, field_length, 0, 0], 
             [0, 0, field_width, field_width, 0], 
             'k-', linewidth=2, label='田地边界')
    
    # 绘制主作业区域边界
    hw = planner.headland_width
    ax1.plot([hw, field_length-hw, field_length-hw, hw, hw],
             [hw, hw, field_width-hw, field_width-hw, hw],
             'b--', linewidth=1, alpha=0.5, label='主作业区域边界')
    
    # 绘制起点连接路径（如果存在）
    if result['approach_path'] is not None:
        approach = result['approach_path']
        ax1.plot(approach[:, 0], approach[:, 1], 'g-', linewidth=2, label='起点连接路径', alpha=0.7)
        ax1.plot(approach[0, 0], approach[0, 1], 'go', markersize=12, label='停放位置（起点）')
    
    # 绘制主作业路径
    main_path = result['main_work']['path']
    ax1.plot(main_path[:, 0], main_path[:, 1], 'b-', linewidth=0.5, alpha=0.6, label='主作业路径')
    
    # 绘制田头路径
    headland_path = result['headland']['path']
    ax1.plot(headland_path[:, 0], headland_path[:, 1], 'r-', linewidth=1.5, alpha=0.8, label='田头路径')
    
    # 绘制终点连接路径（如果存在）
    if result['departure_path'] is not None:
        departure = result['departure_path']
        ax1.plot(departure[:, 0], departure[:, 1], 'm-', linewidth=2, label='终点连接路径', alpha=0.7)
        ax1.plot(departure[-1, 0], departure[-1, 1], 'mo', markersize=12, label='停放位置（终点）')
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # 右图：起点区域放大
    ax2.set_title('起点区域放大（智能起点选择）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 放大起点区域
    zoom_size = 100
    ax2.set_xlim(0, zoom_size)
    ax2.set_ylim(0, zoom_size)
    
    # 绘制田地边界
    ax2.plot([0, field_length, field_length, 0, 0], 
             [0, 0, field_width, field_width, 0], 
             'k-', linewidth=2, label='田地边界')
    
    # 绘制起点连接路径
    if result['approach_path'] is not None:
        approach = result['approach_path']
        ax2.plot(approach[:, 0], approach[:, 1], 'g-', linewidth=3, label='起点连接路径')
        ax2.plot(approach[0, 0], approach[0, 1], 'go', markersize=15, label='停放位置')
        ax2.plot(approach[-1, 0], approach[-1, 1], 'rs', markersize=15, label='作业起点')
    
    # 绘制田头路径起始部分
    headland_path = result['headland']['path']
    mask = (headland_path[:, 0] < zoom_size) & (headland_path[:, 1] < zoom_size)
    if np.any(mask):
        ax2.plot(headland_path[mask, 0], headland_path[mask, 1], 'r-', linewidth=2, alpha=0.8, label='田头路径')
    
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/github_repo/v351_start_end_test.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: v351_start_end_test.png")
    
    # 打印统计信息
    print("\n" + "="*80)
    print("路径规划统计")
    print("="*80)
    
    if result['approach_path'] is not None:
        approach_len = planner._calculate_path_length(result['approach_path'])
        print(f"起点连接路径长度: {approach_len:.1f}m")
    
    print(f"主作业路径长度: {result['main_work']['stats']['path_length_km']*1000:.1f}m")
    print(f"田头路径长度: {result['headland']['stats']['path_length_km']*1000:.1f}m")
    
    if result['departure_path'] is not None:
        departure_len = planner._calculate_path_length(result['departure_path'])
        print(f"终点连接路径长度: {departure_len:.1f}m")
    
    total_len = result['main_work']['stats']['path_length_km'] + result['headland']['stats']['path_length_km']
    if result['approach_path'] is not None:
        total_len += approach_len / 1000
    if result['departure_path'] is not None:
        total_len += departure_len / 1000
    
    print(f"总路径长度: {total_len:.2f}km")
    print(f"田头覆盖率: {result['headland']['stats']['coverage_rate']*100:.1f}%")
    print(f"计算耗时: {result['total_time']:.3f}秒")
    print(f"版本: {result['version']}")
    print(f"特性: {', '.join(result['features'])}")
    print("="*80)

if __name__ == "__main__":
    test_with_start_end_points()

