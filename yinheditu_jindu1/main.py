#!/usr/bin/env python3
""" Pioneer 宇宙地图生成器 - 完整合集版 """

import numpy as np
import matplotlib.pyplot as plt

# 导入两个模块
import weitushengcheng
import xingxingshengcheng

def run_all_steps():
    """运行所有生成步骤"""
    print("=" * 70)
    print(" Pioneer 宇宙地图生成器 - 完整合集版")
    print("=" * 70)
    
    # 步骤1-3：生成星系地图
    print("\n" + "=" * 70)
    print("【步骤 1-3】星系密度地图与恒星分布生成")
    print("=" * 70)
    
    weitushengcheng.plot_density_map()
    weitushengcheng.plot_density_sampled()
    weitushengcheng.plot_star_map(scan_range=60)
    
    # 步骤4：生成恒星系统内部
    print("\n" + "=" * 70)
    print("【步骤 4】恒星系统内部生成（行星、卫星、轨道）")
    print("=" * 70)
    
    # 创建示例系统
    examples = []
    
    # 太阳型系统
    sol_system = xingxingshengcheng.StarSystem(0, 0, 0, 0)
    sol_system.name = "Sol"
    sol_system.star_type = "STAR_G"
    sol_system.explored = True
    sol_system.seed = weitushengcheng.UNIVERSE_SEED
    
    sol_rng = weitushengcheng.RNG(0, 0, 0, weitushengcheng.UNIVERSE_SEED)
    xingxingshengcheng.generate_detailed_star_system(sol_system, sol_rng)
    
    examples.append((sol_system, "Sol (太阳型)"))
    
    # 红矮星系统
    proxima_system = xingxingshengcheng.StarSystem(-1, 0, 0, 0)
    proxima_system.name = "Proxima"
    proxima_system.star_type = "STAR_M"
    proxima_system.explored = True
    proxima_system.seed = weitushengcheng.UNIVERSE_SEED + 1
    
    proxima_rng = weitushengcheng.RNG(-1, 0, 0, weitushengcheng.UNIVERSE_SEED + 1)
    xingxingshengcheng.generate_detailed_star_system(proxima_system, proxima_rng)
    
    examples.append((proxima_system, "Proxima (红矮星)"))
    
    # 蓝巨星系统
    rigel_system = xingxingshengcheng.StarSystem(50, 20, 0, 0)
    rigel_system.name = "Rigel-Prime"
    rigel_system.star_type = "STAR_B"
    rigel_system.explored = False
    rigel_system.seed = weitushengcheng.UNIVERSE_SEED + 2
    
    rigel_rng = weitushengcheng.RNG(50, 20, 0, weitushengcheng.UNIVERSE_SEED + 2)
    xingxingshengcheng.generate_detailed_star_system(rigel_system, rigel_rng)
    
    examples.append((rigel_system, "Rigel-Prime (蓝巨星)"))
    
    # 打印统计
    print("\n📊 恒星系统统计:")
    for system, title in examples:
        bodies = system.get_all_bodies()
        print(f"\n{title}:")
        print(f"  总天体数: {len(bodies)}")
        
        body_types = {}
        for b in bodies:
            body_types[b.type] = body_types.get(b.type, 0) + 1
        
        for btype, count in sorted(body_types.items()):
            print(f"    {btype}: {count}")
    
    # 绘制对比图
    xingxingshengcheng.plot_multiple_star_systems(examples)
    xingxingshengcheng.plot_star_system_orbit_diagram(sol_system)
    
    print("\n✅ 所有步骤完成！")
    print("\n生成的文件：")
    print("  - galaxy_density_map.png (星系密度图)")
    print("  - galaxy_density_sampled.png (采样后的密度图)")
    print("  - galaxy_star_map.png (星系恒星地图)")
    print("  - multiple_star_systems_comparison.png (多系统对比)")
    print("  - star_system_Sol_orbits.png (太阳系轨道图)")

if __name__ == "__main__":
    run_all_steps()
