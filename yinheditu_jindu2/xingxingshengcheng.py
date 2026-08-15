#!/usr/bin/env python3
""" 步骤4：恒星系统内部生成（行星、卫星、轨道） - 3D版本 """
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import hashlib
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from weitushengcheng import (
    SECTOR_SIZE, UNIVERSE_SEED, GALAXY_RADIUS, SOL_OFFSET_X, SOL_OFFSET_Y,
    MAP_SIZE, GALAXY_EXPLORED_MAX, GALAXY_EXPLORED_MIN, GALAXY_EXPLORED_MIX,
    RNG, STAR_TYPES, generate_galaxy_density_map, sample, get_sector_density,
    pick_star_type, is_explored, gen_system_name
)

AU = 1.496e11
EARTH_RADIUS = 6.371e6
EARTH_MASS = 5.972e24
SUN_RADIUS = 6.96e8
SUN_MASS = 1.989e30
G = 6.674e-11
STEFAN_BOLTZMANN = 5.67e-8

class SystemBody:
    def __init__(self, body_type, parent=None):
        self.type = body_type
        self.parent = parent
        self.children = []
        self.mass = 0.0
        self.radius = 0.0
        self.temperature = 0.0
        self.semi_major_axis = 0.0
        self.eccentricity = 0.0
        self.inclination = 0.0
        self.longitude_of_ascending_node = 0.0
        self.argument_of_periapsis = 0.0
        self.true_anomaly = 0.0
        self.atmosphere_density = 0.0
        self.atmosphere_composition = {}
        self.has_life = False
        self.metallicity = 0.0
        self.rotation_period = 0.0
        self.axial_tilt = 0.0
        self.has_rings = False
        self.ring_inner_radius = 0.0
        self.ring_outer_radius = 0.0
        self.ring_color = (1.0, 0.8, 0.6, 0.6)
        self.name = ""
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    def get_distance_to_parent(self):
        if not self.parent or self.semi_major_axis == 0:
            return 0.0
        return self.semi_major_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(self.true_anomaly))
    def get_orbit_period(self):
        if not self.parent or self.semi_major_axis == 0 or self.parent.mass == 0:
            return 0.0
        m_solar = self.parent.mass * (EARTH_MASS / SUN_MASS)
        if m_solar <= 0:
            return 0.0
        period_years = np.sqrt(self.semi_major_axis**3 / m_solar)
        return period_years

class StarSystem:
    def __init__(self, sx, sy, sz, sys_idx):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.sys_idx = sys_idx
        self.name = ""
        self.star_type = "STAR_M"
        self.explored = False
        self.seed = 0
        self.root_bodies = []
    def get_all_bodies(self):
        bodies = []
        def traverse(body):
            bodies.append(body)
            for child in body.children:
                traverse(child)
        for root in self.root_bodies:
            traverse(root)
        return bodies

def calc_hill_radius(primary, satellite):
    if primary.mass <= 0:
        return 0.0
    distance = satellite.get_distance_to_parent()
    if distance <= 0:
        return 0.0
    m_primary = primary.mass * (EARTH_MASS / SUN_MASS)
    m_satellite = satellite.mass * (EARTH_MASS / SUN_MASS)
    hill_radius = distance * (m_satellite / (3 * m_primary))**(1.0/3.0)
    return hill_radius

def calc_disk_density(primary, disc_min, disc_max):
    if disc_max <= disc_min:
        return 0.0
    total_area = np.pi * (disc_max**2 - disc_min**2)
    density = primary.mass / total_area
    return density

def mass_from_disk_area(min_slice, max_slice, disc_max):
    if max_slice <= min_slice:
        return 0.0
    a = min_slice / disc_max
    b = max_slice / disc_max
    mass = (b**2 - a**2) * (1.0 - (a + b) / 2.0)
    return max(0.0, mass)

def test_overlap(x1, x2, y1, y2):
    return (x1 >= y1 and x1 <= y2) or (x2 >= y1 and x2 <= y2) or \
           (y1 >= x1 and y1 <= x2) or (y2 >= x1 and y2 <= x2)

def calc_body_satellite_shell_density(primary, rng):
    if primary.type.startswith("STAR_"):
        disc_min = 4.0 * (primary.radius * EARTH_RADIUS / AU)
        stellar_mass_solar = primary.mass * (EARTH_MASS / SUN_MASS)
        disc_max = 100.0 * np.sqrt(stellar_mass_solar)
        disc_min = max(disc_min, 0.1)
        density = calc_disk_density(primary, disc_min, disc_max)
    else:
        disc_min = 4.0 * (primary.radius * EARTH_RADIUS / AU) * 0.1
        disc_max = calc_hill_radius(primary, primary) * 0.3
        density = calc_disk_density(primary, disc_min, disc_max)
    return disc_min, disc_max, density

def make_body_in_orbit_slice(system, primary, min_slice, max_slice, disc_max, density, rng):
    if max_slice <= min_slice or density <= 0:
        return None
    periapsis = min_slice + (max_slice - min_slice) * rng.double(0.4, 0.6)
    apoapsis = max_slice - (max_slice - min_slice) * rng.double(0.4, 0.6)
    semi_major_axis = (periapsis + apoapsis) / 2.0
    eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis)
    eccentricity = min(eccentricity, 0.7)
    if semi_major_axis <= 0:
        return None
    effective_max_slice = min_slice + (max_slice - min_slice) * (1 - eccentricity)**2
    mass = mass_from_disk_area(min_slice, effective_max_slice, disc_max)
    mass *= density * rng.double(0.5, 2.0)
    mass = max(0.001, min(mass, 317.0 * 13.0))
    body_type = determine_body_type(mass, parent=primary)
    body = SystemBody(body_type)
    body.mass = mass
    body.semi_major_axis = semi_major_axis
    body.eccentricity = eccentricity
    body.inclination = rng.double(0, np.pi / 12)
    body.longitude_of_ascending_node = rng.double(0, 2 * np.pi)
    body.argument_of_periapsis = rng.double(0, 2 * np.pi)
    body.true_anomaly = rng.double(0, 2 * np.pi)
    set_body_properties(body, rng, primary)
    return body

def determine_body_type(mass, parent):
    distance_to_star = 0.0
    if parent:
        star = parent
        while star.parent and not star.type.startswith("STAR_"):
            star = star.parent
        if star.type.startswith("STAR_"):
            distance_to_star = parent.semi_major_axis if parent.semi_major_axis > 0 else 0.0
    if mass < 0.01:
        return "ASTEROID"
    elif mass < 0.1:
        return "DWARF_PLANET"
    elif mass < 1.0:
        return "TERRESTRIAL"
    elif mass < 10.0:
        if distance_to_star > 3.0:
            return "ICE_GIANT"
        return "OCEAN_WORLD"
    elif mass < 317.0:
        return "GAS_GIANT"
    else:
        return "ICE_GIANT"

def set_body_properties(body, rng, parent):
    if body.type == "ASTEROID":
        body.radius = (body.mass * 0.1)**0.33
    elif body.type == "DWARF_PLANET":
        body.radius = body.mass**0.3
    elif body.type == "TERRESTRIAL":
        body.radius = body.mass**0.27
    elif body.type == "OCEAN_WORLD":
        body.radius = body.mass**0.4
    elif body.type == "GAS_GIANT":
        body.radius = (body.mass * 0.1)**0.5
    elif body.type == "ICE_GIANT":
        body.radius = (body.mass * 0.15)**0.5
    body.rotation_period = rng.double(6, 24)
    body.axial_tilt = rng.double(0, np.pi / 4)
    if body.type in ["TERRESTRIAL", "OCEAN_WORLD"]:
        body.atmosphere_density = rng.double(0.5, 2.0)
        if body.type == "TERRESTRIAL":
            body.atmosphere_composition = {
                'N2': rng.double(0.7, 0.8),
                'O2': rng.double(0.1, 0.2),
                'CO2': rng.double(0.0, 0.1)
            }
        else:
            body.atmosphere_composition = {
                'H2O': rng.double(0.8, 0.95),
                'N2': rng.double(0.05, 0.2)
            }
    if body.type in ["GAS_GIANT", "ICE_GIANT"]:
        body.has_rings = rng.double() < 0.5
        if body.has_rings:
            body.ring_inner_radius = 1.5 * body.radius
            body.ring_outer_radius = 2.5 * body.radius
            ring_hue = rng.double(0.0, 1.0)
            body.ring_color = (ring_hue, ring_hue, ring_hue, 0.6)

def make_planets_around(system, primary, rng):
    disc_min, disc_max, density = calc_body_satellite_shell_density(primary, rng)
    if disc_max <= disc_min or density <= 0:
        return
    num_tries = 0
    max_planets = 8
    planets = []
    current_pos = disc_min + rng.double(0.1, 0.3) * (disc_max - disc_min)
    while len(planets) < max_planets and num_tries < max_planets * 2:
        slice_size = rng.double(0.3, 0.8) * current_pos
        next_pos = current_pos + slice_size
        if next_pos >= disc_max:
            break
        planet = make_body_in_orbit_slice(
            system, primary, current_pos, next_pos, disc_max, density, rng
        )
        if planet:
            overlap = False
            for existing in planets:
                if test_overlap(
                    existing.semi_major_axis * (1 - existing.eccentricity),
                    existing.semi_major_axis * (1 + existing.eccentricity),
                    planet.semi_major_axis * (1 - planet.eccentricity),
                    planet.semi_major_axis * (1 + planet.eccentricity)
                ):
                    overlap = True
                    break
            if not overlap:
                planets.append(planet)
                primary.add_child(planet)
                planet.name = f"{primary.name} {chr(ord('a') + len(planets) - 1)}"
                if planet.type in ["GAS_GIANT", "ICE_GIANT"]:
                    make_moons_around(system, planet, rng)
            current_pos = next_pos + rng.double(0.1, 0.3) * slice_size
        else:
            current_pos = next_pos + rng.double(0.1, 0.3) * slice_size
        num_tries += 1

def make_moons_around(system, planet, rng):
    if planet.mass < 0.1:
        return
    disc_min, disc_max, density = calc_body_satellite_shell_density(planet, rng)
    if disc_max <= disc_min or density <= 0:
        return
    num_tries = 0
    max_moons = 4
    moons = []
    current_pos = disc_min + rng.double(0.2, 0.5) * (disc_max - disc_min)
    while len(moons) < max_moons and num_tries < max_moons * 3:
        slice_size = rng.double(0.3, 0.7) * current_pos
        next_pos = current_pos + slice_size
        if next_pos >= disc_max:
            break
        moon_density = density * 0.1
        moon = make_body_in_orbit_slice(
            system, planet, current_pos, next_pos, disc_max, moon_density, rng
        )
        if moon and moon.mass < planet.mass * 0.01:
            moons.append(moon)
            planet.add_child(moon)
            moon.name = f"{planet.name}-{len(moons)}"
        current_pos = next_pos + rng.double(0.2, 0.5) * slice_size
        num_tries += 1

def calculate_surface_temperature(body, star):
    if body.semi_major_axis <= 0:
        return body.temperature
    luminosity_factor = star.mass
    energy_density = luminosity_factor / (16 * np.pi**2 * body.semi_major_axis**2)
    albedo = 0.3
    greenhouse = 0.1
    effective_temp = (energy_density * (1 - albedo) / (4 * STEFAN_BOLTZMANN * (1 - greenhouse)))**0.25
    return effective_temp

def generate_detailed_star_system(star_system, rng):
    star = SystemBody(star_system.star_type)
    star_info = STAR_TYPES.get(star_system.star_type, STAR_TYPES["STAR_M"])
    mass_range = star_info.get("mass_range", (0.08, 0.5))
    star.mass_solar = rng.double(*mass_range)  # 0.105 M☉（太阳质量）
    star.mass = star.mass_solar * (SUN_MASS / EARTH_MASS)  # → 34957 M⊕
    star.radius = star.mass**0.8
    star.temperature = 3000 * (star.mass / 0.5)**0.5
    star.name = star_system.name
    star_system.root_bodies.append(star)
    make_planets_around(star_system, star, rng)
    if not star_system.get_all_bodies():
        print(f"警告: {star_system.name} 没有生成天体，强制创建地球...")
        planet = SystemBody("TERRESTRIAL")
        planet.name = "Earth"
        planet.mass = 1.0
        planet.semi_major_axis = 1.0
        planet.eccentricity = 0.0
        star.add_child(planet)
    for body in star_system.get_all_bodies():
        if body.type != star_system.star_type:
            body.temperature = calculate_surface_temperature(body, star)

def get_body_color(body, depth=0):
    type_colors = {
        "STAR_O": "#FFAEC0", "STAR_B": "#B2B2FF", "STAR_A": "#FFFFFF",
        "STAR_F": "#FFFFCC", "STAR_G": "#FFFF66", "STAR_K": "#FF991A",
        "STAR_M": "#FF3318", "WHITE_DWARF": "#6666CC", "BROWN_DWARF": "#800000",
        "GAS_GIANT": "#FFB366", "ICE_GIANT": "#66B2FF", "OCEAN_WORLD": "#3399FF",
        "TERRESTRIAL": "#4A7C4E", "DWARF_PLANET": "#8B7355", "ASTEROID": "#A9A9A9"
    }
    return type_colors.get(body.type, "#CCCCCC")

def get_body_marker(body):
    type_markers = {
        "STAR_O": "*", "STAR_B": "*", "STAR_A": "*", "STAR_F": "*", "STAR_G": "*",
        "STAR_K": "*", "STAR_M": "*", "WHITE_DWARF": "*", "BROWN_DWARF": "*",
        "GAS_GIANT": "o", "ICE_GIANT": "o", "OCEAN_WORLD": "o",
        "TERRESTRIAL": "o", "DWARF_PLANET": ".", "ASTEROID": "."
    }
    return type_markers.get(body.type, "o")

def draw_rings(ax, body, view_angle='top'):
    if not body.has_rings:
        return
    inner_r = body.ring_inner_radius
    outer_r = body.ring_outer_radius
    theta = np.linspace(0, 2*np.pi, 200)
    if view_angle == 'top':
        x_ring = np.concatenate([inner_r * np.cos(theta), outer_r * np.cos(theta[::-1])])
        y_ring = np.concatenate([inner_r * np.sin(theta), outer_r * np.sin(theta[::-1])])
        ax.fill(x_ring, y_ring, color=body.ring_color, alpha=0.6)
    elif view_angle == 'side':
        aspect_ratio = np.sin(body.inclination) if body.inclination > 0 else 0.1
        x_ring = np.concatenate([inner_r * np.cos(theta), outer_r * np.cos(theta[::-1])])
        y_ring = np.concatenate([inner_r * np.sin(theta) * aspect_ratio, outer_r * np.sin(theta[::-1]) * aspect_ratio])
        ax.fill(x_ring, y_ring, color=body.ring_color, alpha=0.6)

def plot_star_system_orbit_diagram_3d(star_system, figsize=(14, 10)):
    bodies = star_system.get_all_bodies()
    if not bodies:
        print("没有天体可绘制")
        return
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'{star_system.name} - 3D视图')
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    
    # 计算最大距离（确保恒星和行星都在可视范围内）
    max_distance = 1.0
    for body in bodies:
        if body.semi_major_axis > max_distance:
            max_distance = body.semi_major_axis * 1.2
    ax.set_xlim(-max_distance, max_distance)
    ax.set_ylim(-max_distance, max_distance)
    ax.set_zlim(-max_distance/2, max_distance/2)
    
    # 分离恒星和行星绘制
    for body in bodies:
        if body.type.startswith("STAR_"):  # 绘制恒星
            current_x, current_y, current_z = 0, 0, 0  # 恒星在原点
            color = get_body_color(body)
            marker = get_body_marker(body)
            size = 50  # 恒星大小（可调整）
            ax.scatter(current_x, current_y, current_z, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
            # 添加恒星名称
            ax.text(current_x, current_y, current_z, body.name, color='black', fontsize=10, ha='center', va='center')
        
        elif body.parent and body.semi_major_axis > 0:  # 绘制行星
            # 绘制轨道
            theta = np.linspace(0, 2*np.pi, 100)
            a = body.semi_major_axis
            e = body.eccentricity
            i = body.inclination
            x_orb = a * (np.cos(theta) - e)
            y_orb = a * np.sqrt(1 - e**2) * np.sin(theta)
            z_orb = np.zeros_like(theta)
            x = x_orb
            y = y_orb * np.cos(i)
            z = y_orb * np.sin(i)
            color = get_body_color(body)
            alpha = 0.6 if body.type.startswith("STAR_") else 0.4
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=0.8)
            
            # 绘制行星当前位置
            current_x = body.semi_major_axis * (np.cos(body.true_anomaly) - body.eccentricity)
            current_y = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(body.true_anomaly)
            current_z = current_y * np.sin(body.inclination)
            marker = get_body_marker(body)
            size = 20  # 行星大小（可调整）
            ax.scatter(current_x, current_y, current_z, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
            # 添加行星名称
            ax.text(current_x, current_y, current_z, body.name, color='black', fontsize=8, ha='center', va='center')
    
    ax.view_init(elev=20, azim=45)
    
    # 保留原有信息文本（天体统计等）
    info_text = f"恒星系统: {star_system.name}\n\n"
    info_text += f"恒星类型: {star_system.star_type}\n"
    info_text += f"已探索: {'是' if star_system.explored else '否'}\n\n"
    info_text += "天体统计:\n"
    info_text += "-" * 30 + "\n"
    body_stats = {}
    for body in bodies:
        btype = body.type
        body_stats[btype] = body_stats.get(btype, 0) + 1
    for btype, count in sorted(body_stats.items()):
        info_text += f"{btype:20s}: {count:3d}\n"
    
    planets = [b for b in bodies if b.type in ["TERRESTRIAL", "OCEAN_WORLD", "GAS_GIANT", "ICE_GIANT"]]
    if planets:
        largest_planet = max(planets, key=lambda b: b.mass)
        info_text += f"\n最大行星: {largest_planet.name}\n"
        info_text += f" 类型: {largest_planet.type}\n"
        info_text += f" 质量: {largest_planet.mass:.3f} 地球质量\n"
        info_text += f" 轨道: {largest_planet.semi_major_axis:.3f} AU\n"
    
    habitable_bodies = [b for b in bodies if 273 <= b.temperature <= 373]
    if habitable_bodies:
        info_text += f"\n宜居带天体: {len(habitable_bodies)} 个\n"
    
    plt.figtext(0.02, 0.98, info_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    filename = f"star_system_{star_system.name.replace(' ', '_')}_3d.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"3D轨道图已保存: {filename}")

def plot_multiple_star_systems_3d(systems_to_plot, figsize=(16, 12)):
    num_systems = len(systems_to_plot)
    fig = plt.figure(figsize=figsize)
    
    if num_systems == 1:
        ax = fig.add_subplot(111, projection='3d')
        axes = [ax]
    else:
        axes = []
        for i in range(num_systems):
            ax = fig.add_subplot(2, num_systems, i+1, projection='3d')
            axes.append(ax)
    
    for idx, (system, title) in enumerate(systems_to_plot):
        bodies = system.get_all_bodies()
        ax = axes[idx] if num_systems > 1 else axes[0]
        ax.set_title(f'{title} - 3D视图')
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        
        # 计算最大距离（确保恒星和行星都在可视范围内）
        max_distance = 1.0
        for body in bodies:
            if body.semi_major_axis > max_distance:
                max_distance = body.semi_major_axis * 1.2
        ax.set_xlim(-max_distance, max_distance)
        ax.set_ylim(-max_distance, max_distance)
        ax.set_zlim(-max_distance/2, max_distance/2)
        
        # 绘制天体（分离恒星和行星）
        for body in bodies:
            if body.type.startswith("STAR_"):  # 绘制恒星
                current_x, current_y, current_z = 0, 0, 0  # 恒星在原点
                color = get_body_color(body)
                marker = get_body_marker(body)
                size = 50  # 恒星大小（可调整）
                ax.scatter(current_x, current_y, current_z, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
                # 添加恒星名称
                ax.text(current_x, current_y, current_z, body.name, color='black', fontsize=10, ha='center', va='center')
            
            elif body.parent and body.semi_major_axis > 0:  # 绘制行星
                # 绘制轨道
                theta = np.linspace(0, 2*np.pi, 100)
                a = body.semi_major_axis
                e = body.eccentricity
                i = body.inclination
                x_orb = a * (np.cos(theta) - e)
                y_orb = a * np.sqrt(1 - e**2) * np.sin(theta)
                z_orb = np.zeros_like(theta)
                x = x_orb
                y = y_orb * np.cos(i)
                z = y_orb * np.sin(i)
                color = get_body_color(body)
                alpha = 0.6 if body.type.startswith("STAR_") else 0.4
                ax.plot(x, y, z, color=color, alpha=alpha, linewidth=0.8)
                
                # 绘制行星当前位置
                current_x = body.semi_major_axis * (np.cos(body.true_anomaly) - body.eccentricity)
                current_y = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(body.true_anomaly)
                current_z = current_y * np.sin(body.inclination)
                marker = get_body_marker(body)
                size = 20  # 行星大小（可调整）
                ax.scatter(current_x, current_y, current_z, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
                # 添加行星名称
                ax.text(current_x, current_y, current_z, body.name, color='black', fontsize=8, ha='center', va='center')
        
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('multiple_star_systems_3d.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("多系统3D对比图已保存")


def plot_star_system_orbit_diagram(star_system, figsize=(14, 10)):
    bodies = star_system.get_all_bodies()
    if not bodies:
        print("没有天体可绘制")
        return
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis('off')
    max_distance = 1.0
    for body in bodies:
        if body.semi_major_axis > max_distance:
            max_distance = body.semi_major_axis * 1.2
    ax_top.set_xlim(-max_distance, max_distance)
    ax_top.set_ylim(-max_distance, max_distance)
    ax_side.set_xlim(-max_distance, max_distance)
    ax_side.set_ylim(-max_distance/4, max_distance/4)
    for body in bodies:
        if body.parent and body.semi_major_axis > 0:
            theta = np.linspace(0, 2*np.pi, 100)
            x_orb = body.semi_major_axis * (np.cos(theta) - body.eccentricity)
            y_orb = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(theta)
            x = x_orb
            y = y_orb * np.cos(body.inclination)
            z = y_orb * np.sin(body.inclination)
            color = get_body_color(body)
            alpha = 0.6 if body.type.startswith("STAR_") else 0.4
            ax_top.plot(x, y, color=color, alpha=alpha, linewidth=0.8)
            ax_side.plot(x, z, color=color, alpha=alpha, linewidth=0.8)
            draw_rings(ax_top, body, 'top')
            draw_rings(ax_side, body, 'side')
            current_x = body.semi_major_axis * (np.cos(body.true_anomaly) - body.eccentricity)
            current_y = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(body.true_anomaly)
            current_x_3d = current_x
            current_y_3d = current_y * np.cos(body.inclination)
            current_z_3d = current_y * np.sin(body.inclination)
            marker = get_body_marker(body)
            size = 50 if body.type.startswith("STAR_") else 20
            ax_top.scatter(current_x_3d, current_y_3d, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
            ax_side.scatter(current_x_3d, current_z_3d, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_xlabel('距离 (AU)')
    ax_top.set_ylabel('距离 (AU)')
    ax_top.set_title(f'{star_system.name} - 俯视图')
    ax_top.set_aspect('equal')
    ax_side.grid(True, alpha=0.3)
    ax_side.set_xlabel('距离 (AU)')
    ax_side.set_ylabel('高度 (AU)')
    ax_side.set_title(f'{star_system.name} - 侧视图')
    ax_side.set_aspect('equal')
    info_text = f"恒星系统: {star_system.name}\n\n"
    info_text += f"恒星类型: {star_system.star_type}\n"
    info_text += f"已探索: {'是' if star_system.explored else '否'}\n\n"
    info_text += "天体统计:\n"
    info_text += "-" * 30 + "\n"
    body_stats = {}
    for body in bodies:
        btype = body.type
        body_stats[btype] = body_stats.get(btype, 0) + 1
    for btype, count in sorted(body_stats.items()):
        info_text += f"{btype:20s}: {count:3d}\n"
    planets = [b for b in bodies if b.type in ["TERRESTRIAL", "OCEAN_WORLD", "GAS_GIANT", "ICE_GIANT"]]
    if planets:
        largest_planet = max(planets, key=lambda b: b.mass)
        info_text += f"\n最大行星: {largest_planet.name}\n"
        info_text += f" 类型: {largest_planet.type}\n"
        info_text += f" 质量: {largest_planet.mass:.3f} 地球质量\n"
        info_text += f" 轨道: {largest_planet.semi_major_axis:.3f} AU\n"
    habitable_bodies = [b for b in bodies if 273 <= b.temperature <= 373]
    if habitable_bodies:
        info_text += f"\n宜居带天体: {len(habitable_bodies)} 个\n"
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    filename = f"star_system_{star_system.name.replace(' ', '_')}_orbits.png"
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"轨道图已保存: {filename}")

def plot_multiple_star_systems(systems_to_plot, figsize=(16, 12)):
    num_systems = len(systems_to_plot)
    fig, axes = plt.subplots(2, num_systems, figsize=figsize)
    if num_systems == 1:
        axes = np.array([axes]).T
    for idx, (system, title) in enumerate(systems_to_plot):
        bodies = system.get_all_bodies()
        ax_top = axes[0, idx] if num_systems > 1 else axes[0]
        ax_top.set_title(f'{title} - 俯视图')
        ax_top.set_xlabel('距离 (AU)')
        ax_top.set_ylabel('距离 (AU)')
        ax_top.grid(True, alpha=0.3)
        ax_side = axes[1, idx] if num_systems > 1 else axes[1]
        ax_side.set_title(f'{title} - 侧视图')
        ax_side.set_xlabel('距离 (AU)')
        ax_side.set_ylabel('高度 (AU)')
        ax_side.grid(True, alpha=0.3)
        max_distance = 1.0
        for body in bodies:
            if body.semi_major_axis > max_distance:
                max_distance = body.semi_major_axis * 1.2
        ax_top.set_xlim(-max_distance, max_distance)
        ax_top.set_ylim(-max_distance, max_distance)
        ax_side.set_xlim(-max_distance, max_distance)
        ax_side.set_ylim(-max_distance/4, max_distance/4)
        for body in bodies:
            if body.parent and body.semi_major_axis > 0:
                theta = np.linspace(0, 2*np.pi, 100)
                x_orb = body.semi_major_axis * (np.cos(theta) - body.eccentricity)
                y_orb = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(theta)
                x = x_orb
                y = y_orb * np.cos(body.inclination)
                z = y_orb * np.sin(body.inclination)
                color = get_body_color(body)
                alpha = 0.6 if body.type.startswith("STAR_") else 0.4
                ax_top.plot(x, y, color=color, alpha=alpha, linewidth=0.8)
                ax_side.plot(x, z, color=color, alpha=alpha, linewidth=0.8)
                draw_rings(ax_top, body, 'top')
                draw_rings(ax_side, body, 'side')
                current_x = body.semi_major_axis * (np.cos(body.true_anomaly) - body.eccentricity)
                current_y = body.semi_major_axis * np.sqrt(1 - body.eccentricity**2) * np.sin(body.true_anomaly)
                current_x_3d = current_x
                current_y_3d = current_y * np.cos(body.inclination)
                current_z_3d = current_y * np.sin(body.inclination)
                marker = get_body_marker(body)
                size = 50 if body.type.startswith("STAR_") else 20
                ax_top.scatter(current_x_3d, current_y_3d, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
                ax_side.scatter(current_x_3d, current_z_3d, color=color, s=size, marker=marker, edgecolors='black', linewidth=0.5)
        ax_top.set_aspect('equal')
        ax_side.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('multiple_star_systems_comparison.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("多系统对比图已保存")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print(" Pioneer 恒星系统内部生成器 — 步骤 4")
    print("=" * 60)
    examples = []
    sol_system = StarSystem(0, 0, 0, 0)
    sol_system.name = "Sol"
    sol_system.star_type = "STAR_G"
    sol_system.explored = True
    sol_system.seed = UNIVERSE_SEED
    sol_rng = RNG(0, 0, 0, UNIVERSE_SEED)
    generate_detailed_star_system(sol_system, sol_rng)
    examples.append((sol_system, "Sol (太阳型)"))
    proxima_system = StarSystem(-1, 0, 0, 0)
    proxima_system.name = "Proxima"
    proxima_system.star_type = "STAR_M"
    proxima_system.explored = True
    proxima_system.seed = UNIVERSE_SEED + 1
    proxima_rng = RNG(-1, 0, 0, UNIVERSE_SEED + 1)
    generate_detailed_star_system(proxima_system, proxima_rng)
    examples.append((proxima_system, "Proxima (红矮星)"))
    rigel_system = StarSystem(50, 20, 0, 0)
    rigel_system.name = "Rigel-Prime"
    rigel_system.star_type = "STAR_B"
    rigel_system.explored = False
    rigel_system.seed = UNIVERSE_SEED + 2
    rigel_rng = RNG(50, 20, 0, UNIVERSE_SEED + 2)
    generate_detailed_star_system(rigel_system, rigel_rng)
    examples.append((rigel_system, "Rigel-Prime (蓝巨星)"))
    print("\n📊 生成的恒星系统统计:")
    for system, title in examples:
        bodies = system.get_all_bodies()
        print(f"\n{title}:")
        print(f" 总天体数: {len(bodies)}")
        print(f" 行星数: {sum(1 for b in bodies if b.type in ['TERRESTRIAL', 'OCEAN_WORLD', 'GAS_GIANT', 'ICE_GIANT'])}")
        print(f" 卫星数: {sum(1 for b in bodies if b.type in ['DWARF_PLANET', 'ASTEROID'] and b.parent and b.parent.type in ['GAS_GIANT', 'ICE_GIANT'])}")
        habitable = [b for b in bodies if 273 <= b.temperature <= 373]
        print(f" 宜居带天体: {len(habitable)}")
    plot_multiple_star_systems_3d(examples)
    plot_star_system_orbit_diagram_3d(sol_system)
    print("\n✅ 恒星系统内部生成完成！")
