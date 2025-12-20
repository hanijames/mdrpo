import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.path import Path
import matplotlib.patches as mpatches

# === CONFIGURATION ===
INSTANCE_INFO_DIR = "../R20_Alpha2/instance_info"
ROUTE_INFO_DIR = "../R20_Alpha2/route_info"
OUTPUT_DIR = "../R20_Alpha2/animations"
TARGETS_LIST = [25]
OBSTACLES_LIST = [25]
SEED_LIST = [23]
ALPHA = 2.0
FPS = 30
PLAYBACK_SPEED = 5
# =====================

def load_svg_marker(svg_filename, svgpath2mpl=None):
    try:
        from svgpath2mpl import parse_path
        import xml.etree.ElementTree as ET

        tree = ET.parse(svg_filename)
        root = tree.getroot()

        for elem in root.iter():
            if elem.tag.endswith('path') or elem.tag == 'path':
                if 'd' in elem.attrib:
                    return parse_path(elem.get('d'))

        raise ValueError(f"No path element found in {svg_filename}")
    except Exception:
        return None

def create_boat_marker():
    marker = load_svg_marker('svg_files/boat.svg')
    if marker:
        return marker
    # backup if svg doesn't load
    verts = [
        (0., 0.6), (-0.25, 0.2), (-0.35, -0.3), (-0.3, -0.5),
        (0.3, -0.5), (0.35, -0.3), (0.25, 0.2), (0., 0.6),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * 6 + [Path.CLOSEPOLY]
    return Path(verts, codes)

def create_drone_marker():
    marker = load_svg_marker('svg_files/drone.svg')
    if marker:
        return marker
    verts = [
        (0., 0.85), (-0.16, 0.16), (-0.7, 0.08), (-0.7, -0.08),
        (-0.23, -0.16), (-0.23, -0.55), (-0.08, -0.62), (0.08, -0.62),
        (0.23, -0.55), (0.23, -0.16), (0.7, -0.08), (0.7, 0.08),
        (0.16, 0.16), (0., 0.85),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * 12 + [Path.CLOSEPOLY]
    return Path(verts, codes)

def calculate_heading(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return -np.degrees(np.arctan2(dx, dy))

def parse_point(s):
    s = s.strip().strip('()').replace('"', '').replace("'", '')
    parts = s.split(',')
    return (float(parts[0]), float(parts[1]))

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def load_instance_data(json_path, csv_path, alpha):
    with open(json_path, 'r') as f:
        data = json.load(f)

    obstacles = [data[k] for k in sorted(data.keys()) if k.startswith('obstacle_')]
    targets = [tuple(data[k]) for k in sorted(data.keys()) if k.startswith('target_')]
    depot = tuple(data['depot'])

    with open(csv_path, 'r') as f:
        all_lines = f.readlines()

    total_time_indices = [i for i, line in enumerate(all_lines) if 'Total Time' in line]

    if not total_time_indices:
        start_idx = 1
        end_idx = len(all_lines)
    else:
        end_idx = total_time_indices[-1]
        start_idx = 0
        for i in range(end_idx - 1, -1, -1):
            if 'Mothership' in all_lines[i] and 'Distance' in all_lines[i]:
                start_idx = i + 1
                break

    mothership_points = []
    drone_launches = []

    for i in range(start_idx, end_idx):
        line = all_lines[i].strip()
        if not line:
            continue

        parts = []
        current = ""
        in_parens = False

        for char in line:
            if char == '(':
                in_parens = True
                current += char
            elif char == ')':
                in_parens = False
                current += char
            elif char == ',' and not in_parens:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        parts.append(current.strip())

        if len(parts) < 1:
            continue

        mothership_str = parts[0].replace('"', '').replace("'", '').strip()
        drone_out_str = parts[2].strip() if len(parts) > 2 else ''
        drone_return_str = parts[3].strip() if len(parts) > 3 else ''
        target_str = parts[4].replace('"', '').replace("'", '').strip() if len(parts) > 4 else ''

        if mothership_str and mothership_str.startswith('(') and mothership_str.endswith(')'):
            try:
                point = parse_point(mothership_str)
                point_idx = len(mothership_points)
                mothership_points.append(point)

                if drone_out_str and target_str:
                    try:
                        drone_out = float(drone_out_str)
                        target = parse_point(target_str)
                        drone_launches.append({
                            'point_idx': point_idx,
                            'launch_point': point,
                            'target': target,
                            'drone_out': drone_out
                        })
                    except ValueError:
                        pass

                if drone_return_str:
                    try:
                        drone_return = float(drone_return_str)
                        if drone_launches:
                            last_launch = drone_launches[-1]
                            if 'drone_return' not in last_launch or last_launch.get('return_point') is None:
                                last_launch['return_point'] = point
                                last_launch['return_idx'] = point_idx
                                last_launch['drone_return'] = drone_return
                    except ValueError:
                        pass
            except ValueError:
                pass

    segment_distances = []
    for i in range(len(mothership_points)-1):
        segment_distances.append(distance(mothership_points[i], mothership_points[i+1]))
    drone_trips = []
    for launch in drone_launches:
        if 'return_point' not in launch:
            continue

        launch_idx = launch['point_idx']
        return_idx = launch['return_idx']
        ship_dist = sum(segment_distances[launch_idx:return_idx])
        drone_time = (launch['drone_out'] + launch['drone_return']) / alpha

        if drone_time > ship_dist and ship_dist > 0:
            ship_speed_factor = ship_dist / drone_time
            drone_speed_factor = 1.0
        elif ship_dist > 0:
            ship_speed_factor = 1.0
            drone_speed_factor = drone_time / ship_dist
        else:
            ship_speed_factor = 1.0
            drone_speed_factor = 1.0

        drone_trips.append({
            'launch_idx': launch_idx,
            'return_idx': return_idx,
            'launch_point': launch['launch_point'],
            'return_point': launch['return_point'],
            'target': launch['target'],
            'drone_out': launch['drone_out'],
            'drone_return': launch['drone_return'],
            'drone_speed': alpha * drone_speed_factor * PLAYBACK_SPEED,
            'ship_speed_factor': ship_speed_factor,
        })

    segment_times = []
    for i in range(len(segment_distances)):
        speed_factor = 1.0
        for trip in drone_trips:
            if trip['launch_idx'] <= i < trip['return_idx']:
                speed_factor = min(speed_factor, trip['ship_speed_factor'])
        segment_times.append(segment_distances[i] / speed_factor)

    wait_at_waypoint = {}
    for trip in drone_trips:
        ship_dist = sum(segment_distances[trip['launch_idx']:trip['return_idx']])
        if ship_dist == 0:
            drone_time = (trip['drone_out'] + trip['drone_return']) / alpha
            wait_at_waypoint[trip['return_idx']] = drone_time

    adjusted_times =[0.0]
    for i, seg_time in enumerate(segment_times):
        wait = wait_at_waypoint.get(i+1, 0.0)
        adjusted_times.append(adjusted_times[-1] + (seg_time + wait) / PLAYBACK_SPEED)

    for trip in drone_trips:
        trip['actual_launch_time'] = adjusted_times[trip['launch_idx']]
        trip['actual_return_time'] = adjusted_times[trip['return_idx']]

    return {
        'obstacles': obstacles,
        'targets': targets,
        'depot': depot,
        'mothership_points': mothership_points,
        'mothership_times': adjusted_times,
        'drone_trips': drone_trips,
        'total_time': adjusted_times[-1],
    }

def create_animation(targets, obstacles, seed, instance_info_dir, route_info_dir, output_dir, alpha, fps):
    instance_name = f"T{targets}_O{obstacles}_S{seed}"

    json_path = os.path.join(instance_info_dir, f"{instance_name}.json")
    csv_path = os.path.join(route_info_dir, f"{instance_name}_route.csv")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading {instance_name}...")
    data = load_instance_data(json_path, csv_path, alpha)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{instance_name}_animation.mp4")

    total_frames = int(data['total_time'] * fps)
    print(f"Creating {total_frames} frames ({data['total_time']:.2f}s at {fps} fps)")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 120)
    ax.set_aspect('equal')
    ax.grid(False)

    for obs_coords in data['obstacles']:
        poly = Polygon(obs_coords, facecolor='#bbbbbb', edgecolor='#888888', alpha=0.8, linewidth=2, zorder=2)
        ax.add_patch(poly)

    depot = data['depot']
    ax.scatter([depot[0]], [depot[1]], s=80, marker='s', color='purple', label='Depot', zorder=8)

    for i, target in enumerate(data['targets']):
        label = 'Target' if i == 0 else None
        ax.scatter([target[0]], [target[1]], s=40, marker='x', color='purple', label=label, zorder=8)

    boat_path = create_boat_marker()
    drone_path = create_drone_marker()

    mothership_line, = ax.plot([], [], '-', linewidth=2.0, color='black', label='Mothership', zorder=5)
    mothership_marker = mpatches.PathPatch(boat_path, facecolor='black', edgecolor='white', linewidth=1, zorder=6)
    ax.add_patch(mothership_marker)

    max_drones = len(data['drone_trips'])
    drone_out_lines = []
    drone_return_lines = []

    for i in range(max_drones):
        line_out, = ax.plot([], [], '--', linewidth=1.5, color='blue', label='Drone Out' if i == 0 else None, zorder=6)
        line_ret, = ax.plot([], [], '--', linewidth=1.5, color='red', label='Drone Return' if i == 0 else None, zorder=6)
        drone_out_lines.append(line_out)
        drone_return_lines.append(line_ret)

    drone_marker = mpatches.PathPatch(drone_path, facecolor='blue', edgecolor='white', linewidth=0.5, zorder=7)
    drone_marker.set_visible(False)
    ax.add_patch(drone_marker)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

    mothership_points = np.array(data['mothership_points'])
    mothership_times = data['mothership_times']
    drone_trips = data['drone_trips']

    def get_mothership_position(t):
        for i in range(len(mothership_times) - 1):
            if mothership_times[i] <= t <= mothership_times[i + 1]:
                segment_duration = mothership_times[i + 1] - mothership_times[i]
                if segment_duration == 0:
                    return mothership_points[i], i
                progress = (t - mothership_times[i]) / segment_duration
                pos = mothership_points[i] + progress * (mothership_points[i + 1] - mothership_points[i])
                return pos, i
        return mothership_points[-1], len(mothership_points) - 1

    def animate(frame):
        current_time = frame / fps

        pos, segment_idx = get_mothership_position(current_time)

        trail_points = [mothership_points[0]]
        for i in range(1, len(mothership_points)):
            if mothership_times[i] <= current_time:
                trail_points.append(mothership_points[i])
            else:
                break
        trail_points.append(pos)
        trail = np.array(trail_points)
        mothership_line.set_data(trail[:, 0], trail[:, 1])

        if segment_idx < len(mothership_points) - 1:
            heading = calculate_heading(mothership_points[segment_idx], mothership_points[segment_idx + 1])
        else:
            heading = 0

        transform = mpatches.transforms.Affine2D().scale(4).rotate_deg(heading).translate(pos[0], pos[1]) + ax.transData
        mothership_marker.set_transform(transform)
        mothership_marker.set_visible(True)

        drone_is_flying = False

        for i, trip in enumerate(drone_trips):
            launch_time = trip['actual_launch_time']
            return_time = trip['actual_return_time']

            if current_time < launch_time:
                drone_out_lines[i].set_data([], [])
                drone_return_lines[i].set_data([], [])

            elif launch_time <= current_time < return_time:
                elapsed = current_time - launch_time
                drone_dist_traveled = elapsed * trip['drone_speed']

                if drone_dist_traveled < trip['drone_out']:
                    progress = drone_dist_traveled / trip['drone_out']
                    drone_pos = (
                        trip['launch_point'][0] + progress * (trip['target'][0] - trip['launch_point'][0]),
                        trip['launch_point'][1] + progress * (trip['target'][1] - trip['launch_point'][1])
                    )
                    drone_out_lines[i].set_data([trip['launch_point'][0], drone_pos[0]], [trip['launch_point'][1], drone_pos[1]])
                    drone_return_lines[i].set_data([], [])

                    if not drone_is_flying:
                        heading = calculate_heading(trip['launch_point'], trip['target'])
                        transform = mpatches.transforms.Affine2D().scale(3.5).rotate_deg(heading).translate(drone_pos[0], drone_pos[1]) + ax.transData
                        drone_marker.set_transform(transform)
                        drone_marker.set_visible(True)
                        drone_is_flying = True
                else:
                    return_dist_traveled = drone_dist_traveled - trip['drone_out']
                    progress = return_dist_traveled / trip['drone_return']
                    drone_pos = (
                        trip['target'][0] + progress * (trip['return_point'][0] - trip['target'][0]),
                        trip['target'][1] + progress * (trip['return_point'][1] - trip['target'][1])
                    )
                    drone_out_lines[i].set_data([trip['launch_point'][0], trip['target'][0]], [trip['launch_point'][1], trip['target'][1]])
                    drone_return_lines[i].set_data([trip['target'][0], drone_pos[0]], [trip['target'][1], drone_pos[1]])

                    if not drone_is_flying:
                        heading = calculate_heading(trip['target'], trip['return_point'])
                        transform = mpatches.transforms.Affine2D().scale(3.5).rotate_deg(heading).translate(drone_pos[0], drone_pos[1]) + ax.transData
                        drone_marker.set_transform(transform)
                        drone_marker.set_visible(True)
                        drone_is_flying = True
            else:
                drone_out_lines[i].set_data([trip['launch_point'][0], trip['target'][0]], [trip['launch_point'][1], trip['target'][1]])
                drone_return_lines[i].set_data([trip['target'][0], trip['return_point'][0]], [trip['target'][1], trip['return_point'][1]])

        if not drone_is_flying:
            drone_marker.set_visible(False)

        ax.set_title(f'{instance_name}', fontsize=14)
        return [mothership_line, mothership_marker]

    print("Rendering animation...")
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=True)

    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"Saved: {output_path}")

    plt.close(fig)

if __name__ == "__main__":
    for targets in TARGETS_LIST:
        for obstacles in OBSTACLES_LIST:
            for seed in SEED_LIST:
                create_animation(targets, obstacles, seed, INSTANCE_INFO_DIR, ROUTE_INFO_DIR, OUTPUT_DIR, ALPHA, FPS)
