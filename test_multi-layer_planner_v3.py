"""
V3.5 测试 - 真正的两层路径规划
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_layer_planner_v3 import (
    TwoLayerPathPlannerV35, VehicleParams
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def test_v35():
    """测试V3.5真正的两层规划"""
    print("="*70)
    print("V3.5 测试 - 真正的两层路径规划")
    print("="*70)
    
    # 使用中型田地测试
    vehicle = VehicleParams(
        working_width=3.2,
        min_turn_radius=8.0,
        max_work_speed_kmh=9.0,
        max_headland_speed_kmh=15.0
    )
    
    planner = TwoLayerPathPlannerV35(
        field_length=500,
        field_width=200,
        vehicle_params=vehicle,
        obstacles=[]
    )
    
    # 规划路径
    result = planner.plan_complete_coverage()
    
    # 验证曲率约束
    all_path = np.vstack([result['main_work']['path'], result['headland']['path']])
    all_speeds = np.concatenate([result['main_work']['speeds'], result['headland']['speeds']])
    curvature_check = planner.verify_curvature_constraints(all_path, all_speeds)
    
    # 验证转角覆盖率
    corner_coverage = planner.verify_all_corners_coverage(result['headland'])
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：完整路径
    ax1 = axes[0]
    
    # 绘制田地边界
    field_boundary = plt.Rectangle(
        (0, 0), planner.field_length, planner.field_width,
        fill=False, edgecolor='black', linewidth=2, label='田地边界'
    )
    ax1.add_patch(field_boundary)
    
    # 绘制主作业区域边界
    main_boundary = plt.Rectangle(
        (planner.headland_width, planner.headland_width),
        planner.field_length - 2*planner.headland_width,
        planner.field_width - 2*planner.headland_width,
        fill=False, edgecolor='blue', linewidth=1.5, 
        linestyle='--', label='主作业区域'
    )
    ax1.add_patch(main_boundary)
    
    # 绘制主作业路径
    main_path = result['main_work']['path']
    ax1.plot(main_path[:, 0], main_path[:, 1], 'b-', linewidth=1, 
            alpha=0.6, label='第一层：主作业路径')
    
    # 绘制田头路径
    headland_path = result['headland']['path']
    ax1.plot(headland_path[:, 0], headland_path[:, 1], 'r-', linewidth=2, 
            label='第二层：田头路径（一圈）')
    
    # 标注4个角落
    corners = [
        (planner.headland_width, planner.headland_width, '角0'),
        (planner.field_length - planner.headland_width, planner.headland_width, '角1'),
        (planner.field_length - planner.headland_width, 
         planner.field_width - planner.headland_width, '角2'),
        (planner.headland_width, planner.field_width - planner.headland_width, '角3')
    ]
    
    for x, y, label in corners:
        ax1.plot(x, y, 'ro', markersize=12)
        ax1.text(x, y, f'  {label}', fontsize=10, color='red', fontweight='bold')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'V3.5 真正的两层路径规划\n'
                 f'田头覆盖率: {result["headland"]["stats"]["coverage_rate"]:.1f}%',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：左下角放大图
    ax2 = axes[1]
    
    # 绘制左下角区域
    corner_x, corner_y = planner.headland_width, planner.headland_width
    R = planner.vehicle.min_turn_radius
    
    # 2R×2R区域
    square = plt.Rectangle((corner_x, corner_y), 2*R, 2*R,
                           fill=False, edgecolor='gray', linewidth=2, 
                           linestyle=':', label='2R×2R区域')
    ax2.add_patch(square)
    
    # 绘制田地边界
    ax2.plot([0, 0], [0, corner_y + 2*R], 'k-', linewidth=2, label='田地边界')
    ax2.plot([0, corner_x + 2*R], [0, 0], 'k-', linewidth=2)
    
    # 绘制主作业区域边界
    ax2.plot([corner_x, corner_x], [0, corner_y + 2*R], 'b--', linewidth=1.5, label='主作业边界')
    ax2.plot([0, corner_x + 2*R], [corner_y, corner_y], 'b--', linewidth=1.5)
    
    # 绘制田头路径（左下角部分）
    headland_corner = headland_path[
        (headland_path[:, 0] < corner_x + 2*R) & 
        (headland_path[:, 1] < corner_y + 2*R)
    ]
    ax2.plot(headland_corner[:, 0], headland_corner[:, 1], 'r-', 
            linewidth=3, label='田头路径（含倒车）')
    
    # 标注角落
    ax2.plot(corner_x, corner_y, 'ro', markersize=15)
    ax2.text(corner_x, corner_y, '  角0\n  (转弯+倒车)', fontsize=12, 
            color='red', fontweight='bold')
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title(f'左下角放大图\n'
                 f'转角覆盖率改进: +{corner_coverage["avg_improvement"]:.1f}%',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, corner_x + 2*R + 2)
    ax2.set_ylim(-2, corner_y + 2*R + 2)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/v35_true_two_layer_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存可视化: /home/ubuntu/v35_true_two_layer_test.png")
    plt.close()
    
    # 输出统计
    print(f"\n[统计摘要]")
    print(f"  田头覆盖率: {result['headland']['stats']['coverage_rate']:.1f}%")
    print(f"  转角平均覆盖率改进: +{corner_coverage['avg_improvement']:.1f}%")
    print(f"  侧向加速度违反率: {curvature_check['accel_violation_rate']:.1f}%")
    
    print(f"\n[路径统计]")
    print(f"  主作业路径点数: {len(result['main_work']['path'])}")
    print(f"  田头路径点数: {len(result['headland']['path'])}")
    
    return result, corner_coverage


if __name__ == '__main__':
    result, corner_coverage = test_v35()
    
    print("\n" + "="*70)
    print("V3.5 测试完成!")
    print("="*70)
    print("\n关键改进:")
    print("  1. 真正的两层规划（不是多层田头）")
    print("  2. 第二层只有一圈路径")
    print("  3. 转弯后切线方向倒车到田地边界")

