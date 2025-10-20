"""
测试V3.6版本的平行四边形田块支持
"""

from multi_layer_planner_v3 import TwoLayerPathPlannerV36, VehicleParams
import matplotlib.pyplot as plt
import numpy as np

def test_parallelogram_fields():
    """测试不同倾斜角度的平行四边形田块"""
    
    # 车辆参数
    vehicle = VehicleParams(
        working_width=3.2,
        min_turn_radius=8.0,
        max_work_speed_kmh=9.0,
        max_headland_speed_kmh=14.0
    )
    
    # 测试场景
    scenarios = [
        {
            'name': '矩形（向后兼容测试）',
            'field_length': 500,
            'field_width': 200,
            'field_vertices': None
        },
        {
            'name': '平行四边形（小倾斜30m）',
            'field_length': None,
            'field_width': None,
            'field_vertices': [
                (0, 0),
                (500, 30),
                (500, 230),
                (0, 200)
            ]
        },
        {
            'name': '平行四边形（大倾斜100m）',
            'field_length': None,
            'field_width': None,
            'field_vertices': [
                (0, 0),
                (500, 100),
                (500, 300),
                (0, 200)
            ]
        },
        {
            'name': '平行四边形（极端倾斜200m）',
            'field_length': None,
            'field_width': None,
            'field_vertices': [
                (0, 0),
                (500, 200),
                (500, 400),
                (0, 200)
            ]
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print("\n" + "="*80)
        print(f"测试场景 {i+1}: {scenario['name']}")
        print("="*80)
        
        try:
            # 创建规划器
            if scenario['field_vertices'] is not None:
                planner = TwoLayerPathPlannerV36(
                    vehicle_params=vehicle,
                    field_vertices=scenario['field_vertices']
                )
            else:
                planner = TwoLayerPathPlannerV36(
                    vehicle_params=vehicle,
                    field_length=scenario['field_length'],
                    field_width=scenario['field_width']
                )
            
            # 规划路径
            result = planner.plan_complete_coverage()
            
            # 保存结果
            results.append({
                'name': scenario['name'],
                'planner': planner,
                'result': result
            })
            
            # 打印统计
            print(f"\n路径规划统计:")
            print(f"  主作业路径长度: {result['main_work']['stats']['path_length_km']:.2f}km")
            print(f"  田头路径长度: {result['headland']['stats']['path_length_km']:.2f}km")
            print(f"  田头覆盖率: {result['headland']['stats']['coverage_rate']*100:.1f}%")
            print(f"  计算耗时: {result['total_time']:.3f}秒")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 可视化所有场景
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, res in enumerate(results):
        ax = axes[i]
        planner = res['planner']
        result = res['result']
        
        ax.set_title(f"{res['name']}\n覆盖率: {result['headland']['stats']['coverage_rate']*100:.1f}%", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 绘制田块边界
        vertices = planner.field_vertices + [planner.field_vertices[0]]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        ax.plot(xs, ys, 'k-', linewidth=2, label='田块边界')
        
        # 绘制主作业路径
        main_path = result['main_work']['path']
        ax.plot(main_path[:, 0], main_path[:, 1], 'b-', linewidth=0.5, alpha=0.6, label='主作业路径')
        
        # 绘制田头路径
        headland_path = result['headland']['path']
        ax.plot(headland_path[:, 0], headland_path[:, 1], 'r-', linewidth=1.5, alpha=0.8, label='田头路径')
        
        # 标注角落角度（如果是平行四边形）
        if planner.field_shape == 'parallelogram':
            for j, (vertex, angle) in enumerate(zip(planner.field_vertices, planner.corner_angles)):
                ax.plot(vertex[0], vertex[1], 'go', markersize=8)
                ax.text(vertex[0], vertex[1], f'  {angle:.1f}°', fontsize=9, 
                       verticalalignment='bottom')
        
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('v36_parallelogram_test.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存到: v36_parallelogram_test.png")
    
    # 打印汇总
    print("\n" + "="*80)
    print("测试汇总")
    print("="*80)
    for res in results:
        result = res['result']
        print(f"{res['name']}:")
        print(f"  田头覆盖率: {result['headland']['stats']['coverage_rate']*100:.1f}%")
        print(f"  总路径长度: {result['main_work']['stats']['path_length_km'] + result['headland']['stats']['path_length_km']:.2f}km")
    print("="*80)

if __name__ == "__main__":
    test_parallelogram_fields()

