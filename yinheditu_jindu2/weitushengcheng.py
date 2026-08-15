#!/usr/bin/env python3
"""
复刻 Pioneer 太空模拟器的程序化宇宙地图生成器
步骤 1-3：坐标系 → 密度图 → 扇区与恒星生成
依赖: numpy, matplotlib, pillow
"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ============================================================
# 全局常量（对应 GalaxyConfig.cpp / Galaxy.cpp / Sector.cpp）
# ============================================================
SECTOR_SIZE = 8.0            # 每个扇区 8 光年见方 (Sector::SIZE)
UNIVERSE_SEED = 42           # 宇宙种子
GALAXY_RADIUS = 50000.0      # 星系半径 50000 光年
SOL_OFFSET_X = 25000.0       # 太阳在密度图中的 X 偏移
SOL_OFFSET_Y = 0.0           # 太阳在密度图中的 Y 偏移
MAP_SIZE = 256               # 密度图分辨率

# 探索范围常量（对应 GalaxyConfig.cpp 默认值）
GALAXY_EXPLORED_MAX = 90     # ~720ly
GALAXY_EXPLORED_MIN = 65     # ~520ly
GALAXY_EXPLORED_MIX = 40     # ~320ly


# ============================================================
# 第 1 步：坐标系与确定性随机数生成器
# ============================================================

class SystemPath:
    """唯一定位宇宙中一个恒星系统的路径"""
    def __init__(self, sx, sy, sz, sys_idx=0, body_idx=0):
        self.sectorX = sx
        self.sectorY = sy
        self.sectorZ = sz
        self.systemIndex = sys_idx
        self.bodyIndex = body_idx

    def __repr__(self):
        return f"({self.sectorX},{self.sectorY},{self.sectorZ},{self.systemIndex})"

    def __hash__(self):
        return hash((self.sectorX, self.sectorY, self.sectorZ, self.systemIndex))

    def __eq__(self, other):
        return (self.sectorX == other.sectorX and
                self.sectorY == other.sectorY and
                self.sectorZ == other.sectorZ and
                self.systemIndex == other.systemIndex)


class RNG:
    """
    确定性随机数生成器。
    根据 SystemPath 生成种子，保证同一坐标永远产生相同的随机序列。
    对应 Pioneer 中 Random(_init, N) 的设计。
    """
    def __init__(self, *seeds):
        combined = "|".join(str(s) for s in seeds)
        h = hashlib.md5(combined.encode()).hexdigest()
        self.state = int(h[:8], 16)

    def _next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7fffffff
        return self.state

    def int32(self, min_val, max_val):
        if max_val <= min_val:
            return min_val
        return min_val + (self._next() % (max_val - min_val + 1))

    def double(self, min_val=0.0, max_val=1.0):
        return min_val + (self._next() / float(0x7fffffff)) * (max_val - min_val)

    def fixed(self):
        return self._next() / float(0x7fffffff)


# ============================================================
# 第 2 步：从文件读取星系密度图
# ============================================================

def load_galaxy_density_map(filename="galaxy_dense.bmp"):
    """
    从 BMP 文件加载密度图。
    对应 Pioneer 中从 galaxy_dense.bmp 读取的逻辑。
    """
    try:
        img = Image.open(filename).convert('L')  # 'L' 模式 = 灰度图
        img = img.resize((MAP_SIZE, MAP_SIZE), Image.Resampling.LANCZOS)  # 确保尺寸正确
        density_map = np.array(img, dtype=np.uint8)
        return density_map
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{filename}'。")
        print("请确保文件存在，并且与脚本在同一文件夹。")
        print("将使用程序化生成的密度图作为备用。")
        return generate_galaxy_density_map()  # 备用：程序化生成
    except Exception as e:
        print(f"加载密度图时出错: {e}")
        print("将使用程序化生成的密度图作为备用。")

def generate_galaxy_density_map(size=MAP_SIZE):
    """
    程序化生成一张螺旋星系密度图（灰度 0~255）。
    Pioneer 从 galaxy_dense.bmp 读取；这里用数学方法替代。
    """
    y, x = np.mgrid[0:size, 0:size]
    x = (x / (size - 1) - 0.5) * 2.0   # [-1, 1]
    y = (y / (size - 1) - 0.5) * 2.0

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # 1. 中心核球
    bulge = np.exp(-(r * 3.5)**2)

    # 2. 旋臂（对数螺旋）
    num_arms = 2
    arm_tightness = 0.5
    arm_phase = (theta - np.log(r + 0.01) / arm_tightness * num_arms) % (2 * np.pi)
    arm_dist = np.minimum(arm_phase, 2 * np.pi - arm_phase)
    arms = np.exp(-(arm_dist * 4.0)**2) * np.exp(-(r * 2.0)**2)

    # 3. 整体盘状
    disk = np.exp(-(r * 2.0)**2)

    density = np.clip(bulge * 0.7 + arms * 0.5 + disk * 0.2, 0, 1)
    return (density * 255).astype(np.uint8)


# 生成全局密度图
density_map = load_galaxy_density_map()
map_w, map_h = density_map.shape[1], density_map.shape[0]


def sample(pixels, x, y, w, h):
    """单点采样（带边界钳制），对应 Galaxy.cpp 的 sample() 函数"""
    tex_x = max(0, min(int(np.floor(x)), w - 1))
    tex_y = max(0, min(int(np.floor(y)), h - 1))
    return pixels[tex_y, tex_x] / 255.0


def get_sector_density(sx, sy, sz):
    """
    根据扇区坐标返回密度值 (0~255)。
    完全对应 Galaxy.cpp 中 DensityMapGalaxy::GetSectorDensity 的逻辑：
    坐标映射 → 四点采样 → 双线性插值 → Z轴衰减。
    """
    # 1. 映射到密度图 UV [0,1]
    offset_x = (((sx * SECTOR_SIZE + SOL_OFFSET_X) / GALAXY_RADIUS) + 1.0) * 0.5
    offset_y = (((-sy * SECTOR_SIZE + SOL_OFFSET_Y) / GALAXY_RADIUS) + 1.0) * 0.5

    # 2. 映射到像素坐标
    fx = offset_x * (map_w - 1)
    fy = offset_y * (map_h - 1)
    mf = fx - int(fx)
    my = fy - int(fy)

    # 3. 四点采样
    val1 = sample(density_map, fx,     fy,     map_w, map_h)
    val2 = sample(density_map, fx + 1, fy,     map_w, map_h)
    val3 = sample(density_map, fx,     fy + 1, map_w, map_h)
    val4 = sample(density_map, fx + 1, fy + 1, map_w, map_h)

    # 4. 双线性插值
    interp_1 = val1 * (1.0 - mf) + val2 * mf
    interp_2 = val3 * (1.0 - mf) + val4 * mf
    val = interp_1 * (1.0 - my) + interp_2 * my

    # 5. Z 轴衰减
    dropoff_scale = max(1.0, 3.5 - val * 2.5)
    dropoff = max(0.0, 1.0 - min(abs(sz) / 256.0 * dropoff_scale, 1.0))
    val = val * dropoff

    return int(max(0, min(255, val * 256.0)))


# ============================================================
# 第 3 步：恒星类型定义与权重表
# ============================================================

STAR_TYPES = {
    "BROWN_DWARF":        {"color": (128,   0,   0), "scale": 0.6, "label": "褐矮星"},
    "WHITE_DWARF":        {"color": (102, 102, 204), "scale": 0.5, "label": "白矮星"},
    "STAR_M":             {"color": (255,  51,   0), "scale": 0.7, "label": "M型红矮星"},
    "STAR_K":             {"color": (255, 153,  26), "scale": 0.8, "label": "K型橙矮星"},
    "STAR_G":             {"color": (255, 255, 102), "scale": 0.8, "label": "G型黄矮星(如太阳)"},
    "STAR_F":             {"color": (255, 255, 204), "scale": 0.9, "label": "F型黄白矮星"},
    "STAR_A":             {"color": (255, 255, 255), "scale": 1.0, "label": "A型白矮星"},
    "STAR_B":             {"color": (178, 178, 255), "scale": 1.1, "label": "B型蓝矮星"},
    "STAR_O":             {"color": (255, 178, 255), "scale": 1.1, "label": "O型蓝巨星"},
    "STAR_M_GIANT":       {"color": (255,  51,   0), "scale": 1.3, "label": "M型巨星"},
    "STAR_K_GIANT":       {"color": (255, 153,  26), "scale": 1.2, "label": "K型巨星"},
    "STAR_G_GIANT":       {"color": (255, 255, 102), "scale": 1.2, "label": "G型巨星"},
    "STAR_M_SUPER_GIANT": {"color": (255,  51,   0), "scale": 1.8, "label": "M型超巨星"},
    "STAR_O_SUPER_GIANT": {"color": (255, 178, 255), "scale": 1.3, "label": "O型超巨星"},
    "STAR_B_HYPER_GIANT": {"color": (178, 178, 255), "scale": 2.0, "label": "B型超巨星"},
    "STAR_O_HYPER_GIANT": {"color": (255, 178, 255), "scale": 1.9, "label": "O型超巨星"},
    "STAR_M_WF":          {"color": (255,  51,   0), "scale": 1.1, "label": "M型沃尔夫-拉叶星"},
    "STAR_B_WF":          {"color": (178, 178, 255), "scale": 1.3, "label": "B型沃尔夫-拉叶星"},
    "STAR_O_WF":          {"color": (255, 178, 255), "scale": 1.6, "label": "O型沃尔夫-拉叶星"},
    "STAR_S_BH":          {"color": ( 76, 178,  76), "scale": 1.0, "label": "恒星级黑洞"},
    "STAR_IM_BH":         {"color": ( 51, 230,  51), "scale": 2.5, "label": "中等质量黑洞"},
    "STAR_SM_BH":         {"color": (  0, 255,   0), "scale": 4.0, "label": "超大质量黑洞"},
}

# 权重表（远离中心 freq > 100 时使用，包含稀有类型）
WEIGHT_TABLE_FAR = [
    (1,       "STAR_IM_BH"),
    (3,       "STAR_S_BH"),
    (5,       "STAR_O_WF"),
    (8,       "STAR_B_WF"),
    (12,      "STAR_M_WF"),
    (15,      "STAR_M_SUPER_GIANT"),
    (23,      "STAR_O_HYPER_GIANT"),
    (48,      "STAR_M_SUPER_GIANT"),
    (158,     "STAR_M_SUPER_GIANT"),
    (208,     "STAR_M_GIANT"),
    (500,     "STAR_M_GIANT"),
    (800,     "STAR_O"),
    (2000,    "STAR_B"),
    (8000,    "STAR_A"),
    (37300,   "STAR_F"),
    (113300,  "STAR_G"),
    (234300,  "STAR_K"),
    (250000,  "WHITE_DWARF"),
    (900000,  "STAR_M"),
    (1000001, "BROWN_DWARF"),
]

# 权重表（靠近中心 freq <= 100 时使用，只有普通恒星）
WEIGHT_TABLE_NEAR = [
    (100,     "STAR_O"),
    (1300,    "STAR_B"),
    (7300,    "STAR_A"),
    (37300,   "STAR_F"),
    (113300,  "STAR_G"),
    (234300,  "STAR_K"),
    (250000,  "WHITE_DWARF"),
    (900000,  "STAR_M"),
    (1000001, "BROWN_DWARF"),
]

def pick_star_type(rng, sx, sy):
    """根据距离选择恒星类型，对应 SectorGenerator.cpp 的权重逻辑"""
    freq = 1 + sx * sx + sy * sy
    weight = rng.int32(0, 999999)
    table = WEIGHT_TABLE_FAR if freq > 100 else WEIGHT_TABLE_NEAR
    for threshold, star_type in table:
        if weight < threshold:
            return star_type
    return "BROWN_DWARF"


# ============================================================
# 探索状态判定（对应 SectorGenerator.cpp）
# ============================================================

def is_explored(rng, sx, sy, sz):
    """
    复刻探索状态判定逻辑：
    0 - ~500ly: explored
    ~500ly - ~700ly: gradual
    ~700ly+: unexplored
    """
    dist = 1 + sx * sx + sy * sy + sz * sz
    if (dist <= GALAXY_EXPLORED_MAX ** 2 and
        (dist <= GALAXY_EXPLORED_MIN ** 2 or
         rng.int32(0, dist) <= GALAXY_EXPLORED_MIX ** 2)):
        return True
    return False


# ============================================================
# 名称生成器（对应 NameGenerator.cpp 的 FrontierNames）
# ============================================================

NAME_FRAGMENTS = [
    "en", "la", "can", "be", "and", "phi", "eth", "ol", "ve", "ho",
    "a", "lia", "an", "ar", "ur", "mi", "in", "ti", "qu", "so",
    "ed", "ess", "ex", "io", "ce", "ze", "fa", "ay", "wa", "de",
    "ack", "gre", "le", "du", "do", "ne"
]

def gen_system_name(rng):
    """复刻 FrontierNames::GetName 的逻辑"""
    name = ""
    length = rng.int32(2, 3)
    for _ in range(length):
        name += NAME_FRAGMENTS[rng.int32(0, len(NAME_FRAGMENTS) - 1)]
    return name[0].upper() + name[1:]


# ============================================================
# 扇区生成器（对应 SectorGenerator.cpp）
# ============================================================

class StarSystem:
    """一个恒星系统"""
    def __init__(self, sx, sy, sz, sys_idx):
        self.sx, self.sy, self.sz = sx, sy, sz
        self.sys_idx = sys_idx
        self.pos = [0.0, 0.0, 0.0]       # 扇区内位置 (0~8 ly)
        self.name = ""
        self.star_type = "STAR_M"
        self.num_stars = 1
        self.explored = False
        self.seed = 0

    def abs_pos(self):
        """返回绝对光年坐标"""
        return (self.sx * SECTOR_SIZE + self.pos[0],
                self.sy * SECTOR_SIZE + self.pos[1],
                self.sz * SECTOR_SIZE + self.pos[2])

    def distance_to_sol(self):
        x, y, z = self.abs_pos()
        return np.sqrt(x**2 + y**2 + z**2)


def generate_sector(sx, sy, sz):
    """
    生成一个扇区内的所有恒星系统。
    对应 SectorGenerator.cpp 中 SectorRandomSystemsGenerator::Apply。
    """
    rng = RNG(sx, sy, sz, UNIVERSE_SEED)

    # 计算恒星数量: (rng.Int32(4, 20) * density) >> 8
    density = get_sector_density(sx, sy, sz)
    num_systems = (rng.int32(4, 20) * density) >> 8

    systems = []
    for i in range(num_systems):
        s = StarSystem(sx, sy, sz, i)

        # 随机位置
        s.pos[0] = rng.double(0, SECTOR_SIZE)
        s.pos[1] = rng.double(0, SECTOR_SIZE)
        s.pos[2] = rng.double(0, SECTOR_SIZE)

        # 恒星数量 (1~4)
        switch_val = rng.int32(0, 14)
        if switch_val == 0:   s.num_stars = 4
        elif switch_val <= 2: s.num_stars = 3
        elif switch_val <= 6: s.num_stars = 2
        else:                 s.num_stars = 1

        # 恒星类型
        s.star_type = pick_star_type(rng, sx, sy)

        # 探索状态
        s.explored = is_explored(rng, sx, sy, sz)

        # 名称
        s.name = gen_system_name(rng)

        # 种子
        s.seed = rng.int32(0, 0x7fffffff)

        systems.append(s)

    return systems


# ============================================================
# 可视化函数
# ============================================================

def plot_density_map():
    """绘制密度图"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(density_map, cmap='hot', origin='lower', extent=[-1, 1, -1, 1])
    ax.set_title("Galaxy Density Map", fontsize=12)
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    plt.tight_layout()
    plt.savefig('galaxy_density_map.png', dpi=100)
    plt.show()


def plot_density_sampled():
    """绘制采样后的扇区密度俯视图和侧视图"""
    scan_range = 60
    density_grid = np.zeros((scan_range * 2, scan_range * 2), dtype=np.uint8)
    for sx in range(-scan_range, scan_range):
        for sy in range(-scan_range, scan_range):
            density_grid[sy + scan_range, sx + scan_range] = get_sector_density(sx, sy, 0)

    z_scan = 80
    z_grid = np.zeros((z_scan * 2, z_scan * 2), dtype=np.uint8)
    for sx in range(-z_scan, z_scan):
        for sz in range(-z_scan, z_scan):
            z_grid[sz + z_scan, sx + z_scan] = get_sector_density(sx, 0, sz)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    extent = [-scan_range * SECTOR_SIZE, scan_range * SECTOR_SIZE,
              -scan_range * SECTOR_SIZE, scan_range * SECTOR_SIZE]
    axes[0].imshow(density_grid, cmap='hot', origin='lower', extent=extent)
    axes[0].plot(0, 0, 'c*', markersize=15, label='Sol')
    axes[0].set_title("Sector Density (Top View, Z=0)", fontsize=12)
    axes[0].set_xlabel("X (ly)")
    axes[0].set_ylabel("Y (ly)")
    axes[0].legend()

    z_extent = [-z_scan * SECTOR_SIZE, z_scan * SECTOR_SIZE,
                -z_scan * SECTOR_SIZE, z_scan * SECTOR_SIZE]
    axes[1].imshow(z_grid, cmap='hot', origin='lower', extent=z_extent, aspect='auto')
    axes[1].plot(0, 0, 'c*', markersize=15, label='Sol')
    axes[1].set_title("Sector Density (Side View, Y=0)", fontsize=12)
    axes[1].set_xlabel("X (ly)")
    axes[1].set_ylabel("Z (ly)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('galaxy_density_sampled.png', dpi=100)
    plt.show()


def plot_star_map(scan_range=60):
    """绘制星系恒星散点图"""
    print(f"正在生成星系数据（扫描 {scan_range*2+1}x{scan_range*2+1} 扇区）...")

    all_stars = []
    for sx in range(-scan_range, scan_range + 1):
        for sy in range(-scan_range, scan_range + 1):
            sz = 0
            for s in generate_sector(sx, sy, sz):
                ax, ay, az = s.abs_pos()
                all_stars.append((ax, ay, s.star_type, s.explored, s.name, s.num_stars))

    print(f"共生成 {len(all_stars)} 个恒星系统")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：按恒星类型着色
    type_groups = {}
    for x, y, st, explored, name, nstars in all_stars:
        if st not in type_groups:
            type_groups[st] = ([], [])
        type_groups[st][0].append(x)
        type_groups[st][1].append(y)

    draw_order = [
        "STAR_M", "BROWN_DWARF", "STAR_K", "WHITE_DWARF", "STAR_G",
        "STAR_F", "STAR_A", "STAR_B", "STAR_O",
        "STAR_M_GIANT", "STAR_K_GIANT", "STAR_G_GIANT",
        "STAR_M_SUPER_GIANT", "STAR_O_SUPER_GIANT", "STAR_O_HYPER_GIANT",
        "STAR_B_HYPER_GIANT",
        "STAR_M_WF", "STAR_B_WF", "STAR_O_WF",
        "STAR_S_BH", "STAR_IM_BH", "STAR_SM_BH"
    ]

    for st in draw_order:
        if st not in type_groups:
            continue
        xs, ys = type_groups[st]
        info = STAR_TYPES[st]
        color = tuple(c / 255.0 for c in info["color"])
        size = info["scale"] * 8
        alpha = 0.8 if info["scale"] >= 1.0 else 0.5
        axes[0].scatter(xs, ys, c=[color], s=size, alpha=alpha,
                        label=info["label"], edgecolors='none')

    axes[0].plot(0, 0, marker='$☀$', markersize=20, color='cyan', zorder=10)
    axes[0].set_title("Galaxy Map — by Star Type", fontsize=13)
    axes[0].set_xlabel("X (ly)")
    axes[0].set_ylabel("Y (ly)")
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=7, loc='upper right', framealpha=0.8, ncol=2)

    # 右图：按探索状态着色
    exp_x = [x for x, y, st, exp, name, ns in all_stars if exp]
    exp_y = [y for x, y, st, exp, name, ns in all_stars if exp]
    unexp_x = [x for x, y, st, exp, name, ns in all_stars if not exp]
    unexp_y = [y for x, y, st, exp, name, ns in all_stars if not exp]

    axes[1].scatter(unexp_x, unexp_y, c='darkred', s=3, alpha=0.4, label='Unexplored', edgecolors='none')
    axes[1].scatter(exp_x, exp_y, c='cyan', s=5, alpha=0.7, label='Explored', edgecolors='none')
    axes[1].plot(0, 0, marker='$☀$', markersize=20, color='yellow', zorder=10)

    for r_sectors, label, ls in [(65, "~520ly", '--'), (90, "~720ly", ':')]:
        r_ly = r_sectors * SECTOR_SIZE
        circle = plt.Circle((0, 0), r_ly, fill=False, color='white', linestyle=ls, linewidth=1, alpha=0.5)
        axes[1].add_patch(circle)
        axes[1].text(r_ly * 0.7, r_ly * 0.7, label, color='white', fontsize=8, alpha=0.7)

    axes[1].set_title("Galaxy Map — by Exploration Status", fontsize=13)
    axes[1].set_xlabel("X (ly)")
    axes[1].set_ylabel("Y (ly)")
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('galaxy_star_map.png', dpi=120)
    plt.show()

    # 打印统计
    type_counts = {}
    explored_count = 0
    for x, y, st, exp, name, ns in all_stars:
        type_counts[st] = type_counts.get(st, 0) + 1
        if exp:
            explored_count += 1

    print(f"\n总系统数: {len(all_stars)}")
    print(f"已探索: {explored_count} ({100*explored_count/len(all_stars):.1f}%)")
    print(f"未探索: {len(all_stars)-explored_count} ({100*(len(all_stars)-explored_count)/len(all_stars):.1f}%)")
    print("\n恒星类型分布:")
    for st in draw_order:
        if st in type_counts:
            info = STAR_TYPES[st]
            pct = 100 * type_counts[st] / len(all_stars)
            print(f"  {info['label']:20s}: {type_counts[st]:5d} ({pct:5.1f}%)")

    print("\nSol 附近的星系名称:")
    nearby = sorted(all_stars, key=lambda s: s[0]**2 + s[1]**2)[:10]
    for x, y, st, exp, name, ns in nearby:
        dist = np.sqrt(x**2 + y**2)
        print(f"  {name:15s}  距离Sol: {dist:6.1f}ly  类型: {STAR_TYPES[st]['label']:12s}  探索: {'是' if exp else '否'}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 即使没显式用到，也需要注册3D投影
import numpy as np

# ... (保留您原有的所有代码：类定义、常量、密度图生成逻辑等) ...

def plot_3d_galaxy(scan_range=60):
    """
    绘制 3D 星系图，实现 LOD 和交互。
    scan_range: 扫描半径（扇区数）。建议 60 左右，扫描范围过大会导致 Python 变慢。
    """
    print(f"正在生成 3D 星系数据（扫描 {scan_range*2+1}x{scan_range*2+1} 扇区）...")
    
    # 存储数据列表
    near_stars = []  # 距离 < 500 ly，高细节
    far_stars = []   # 距离 > 500 ly，低细节
    
    LOD_THRESHOLD = 500.0  # 光年阈值
    
    # 1. 数据生成循环 (参考 SectorGenerator.cpp 逻辑)
    # 为了性能，我们只扫描一个立方体区域，或者您可以限制扫描扇区数
    for sx in range(-scan_range, scan_range + 1):
        for sy in range(-scan_range, scan_range + 1):
            for sz in range(-scan_range, scan_range + 1):
                # 密度筛选优化：如果密度为0，跳过 (参考 C++ 密度图逻辑)
                density = get_sector_density(sx, sy, sz)
                if density == 0: continue

                # 生成扇区系统
                systems = generate_sector(sx, sy, sz)
                
                for s in systems:
                    # 计算绝对光年坐标 (参考 Sector::System::abs_pos)
                    # C++: pos + Sector::SIZE * sector_coords
                    ax = s.sx * SECTOR_SIZE + s.pos[0]
                    ay = s.sy * SECTOR_SIZE + s.pos[1]
                    az = s.sz * SECTOR_SIZE + s.pos[2]
                    
                    # 计算到原点 的欧几里得距离
                    dist = np.sqrt(ax**2 + ay**2 + az**2)
                    
                    star_data = {
                        'pos': (ax, ay, az),
                        'type': s.star_type,
                        'name': s.name,
                        'explored': s.explored
                    }
                    
                    # 分层存储
                    if dist < LOD_THRESHOLD:
                        near_stars.append(star_data)
                    else:
                        far_stars.append(star_data)
                        
    print(f"生成完成。近处系统: {len(near_stars)}, 远处系统: {len(far_stars)}")

    # 2. 3D 绘图初始化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置中文字体 (如果在 plt.rcParams 中已设置，这里可省略)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 3. 绘制远处恒星 (LOD: 两色点)
    # 颜色参考 C++ 逻辑：已探索 (Cyan), 未探索
    if far_stars:
        # 提取坐标
        fx = [s['pos'][0] for s in far_stars]
        fy = [s['pos'][1] for s in far_stars]
        fz = [s['pos'][2] for s in far_stars]
        
        # 颜色列表
        colors = ['cyan' if s['explored'] else 'darkred' for s in far_stars]
        
        # 绘制散点，点大小较小，透明度较高
        ax.scatter(fx, fy, fz, c=colors, s=2, alpha=0.3, marker='.', depthshade=True)
        
    # 4. 绘制近处恒星 (LOD: 真实颜色 + 名字)
    if near_stars:
        for s in near_stars:
            x, y, z = s['pos']
            
            # 获取恒星类型定义 (参考 NameGenerator.cpp / StarSystem 定义)
            stype = STAR_TYPES.get(s['type'], STAR_TYPES['STAR_M'])
            
            # 颜色归一化到 0-1 范围给 matplotlib
            color = tuple(c / 255.0 for c in stype['color'])
            
            # 绘制恒星点
            ax.scatter(x, y, z, c=[color], s=stype['scale'] * 15, alpha=0.9, 
                       edgecolors='white', linewidths=0.5, depthshade=False)
            
            # 添加名字 (参考 C++ Sector::Dump 输出逻辑)
            # 为了避免视觉拥堵，我们可以限制只显示最近的若干个名字，或者全部显示由用户自己旋转查看
            # 这里为了效果，我们对所有近处恒星显示名字
            ax.text(x, y, z, s['name'], fontsize=8, color='white', 
                    ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.3, lw=0))

    # 5. 绘制原点
    ax.scatter(0, 0, 0, marker='o', s=100, c='yellow', label='Sol (Origin)', edgecolors='black')

    # 6. 设置视区与交互
    ax.set_xlabel('X (ly)')
    ax.set_ylabel('Y (ly)')
    ax.set_zlabel('Z (ly)')
    ax.set_title(f'Pioneer Galaxy 3D View\nNear (<{LOD_THRESHOLD}ly): Colored & Named | Far: Status Points')
    
    # 设置等比例坐标轴（模拟真实空间感）
    # 由于 matplotlib 3D 的限制，很难做到完美等比例，但可以设置范围
    max_r = max(LOD_THRESHOLD, scan_range * SECTOR_SIZE)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)
    
    # 背景色设为黑色，模拟太空
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存并显示
    plt.savefig('galaxy_3d_map.png', dpi=120, facecolor='black')
    print("✅ 3D 图已保存为 galaxy_3d_map.png")
    
    # 显示交互窗口 (支持鼠标拖拽旋转)
    plt.show()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Pioneer 宇宙地图生成器 — 步骤 1~3")
    print("=" * 60)

    print("\n[1/3] 验证确定性随机数...")
    rng1 = RNG(0, 0, 0, UNIVERSE_SEED)
    rng2 = RNG(0, 0, 0, UNIVERSE_SEED)
    rng3 = RNG(1, 0, 0, UNIVERSE_SEED)
    print(f"  (0,0,0) -> {rng1.int32(0,999)}, {rng1.double():.4f}")
    print(f"  (0,0,0) -> {rng2.int32(0,999)}, {rng2.double():.4f}  [一致: {rng1.int32(0,999)==rng2.int32(0,999)}]")
    print(f"  (1,0,0) -> {rng3.int32(0,999)}, {rng3.double():.4f}")

    print("\n[2/3] 生成密度图...")
    print(f"  Sol (0,0,0) 密度: {get_sector_density(0, 0, 0)}")
    print(f"  远处 (50,0,0) 密度: {get_sector_density(50, 0, 0)}")
    print(f"  高处 (0,0,100) 密度: {get_sector_density(0, 0, 100)}")
    plot_density_map()
    plot_density_sampled()

    print("\n[3/3] 生成星系恒星地图...")
    plot_star_map(scan_range=60)


    # 在最后添加 3D 绘图调用
    print("\n[4/4] 正在生成 3D 星图...")
    plot_3d_galaxy(scan_range=35) # 扫描半径设为35个扇区 (约280ly)，包含500ly阈值过渡区

