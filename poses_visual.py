import math

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from pylinkage.synthesis import Pose

import csv

poses = []

with open('poses.csv', mode='r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        x = 3*float(row['X'])
        y = float(row['Y'])
        phi_deg = float(row['PHI'])

        adjusted_angle_rad = math.radians(phi_deg + 90)
        
        poses.append(Pose(x, y, adjusted_angle_rad))

for i, p in enumerate(poses):
    print(f"Pose {i + 1}: x={p.x:.2f}, y={p.y:.2f}, angle={math.degrees(p.angle):.1f} deg")

# Visualize the target poses as rectangles with orientation arrows
fig, ax = plt.subplots(figsize=(10, 6))

rect_w, rect_h = 1.0, 0.4  # Body dimensions for visualization
#colors = ['#2196F3', '#FF9800', '#4CAF50', '#2196F3', '#FF9800', '#4CAF50', '#2196F3', '#FF9800', '#4CAF50']
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fdae61', '#f46d43', '#d73027', '#a50026']

for i, (pose, color) in enumerate(zip(poses, colors, strict=False)):
    # Draw a rotated rectangle at each pose
    rect = Rectangle(
        (-rect_w / 2, -rect_h / 2), rect_w, rect_h,
        linewidth=2, edgecolor=color, facecolor=color, alpha=0.3,
    )
    t = (
        mtransforms.Affine2D()
        .rotate(pose.angle)
        .translate(pose.x, pose.y)
        + ax.transData
    )
    rect.set_transform(t)
    ax.add_patch(rect)

    # Orientation arrow
    dx = 0.6 * math.cos(pose.angle)
    dy = 0.6 * math.sin(pose.angle)
    ax.annotate(
        '', xy=(pose.x + dx, pose.y + dy), xytext=(pose.x, pose.y),
        arrowprops=dict(arrowstyle='->', color=color, lw=2),
    )
    ax.plot(pose.x, pose.y, 'o', color=color, markersize=8)
    ax.annotate(
        f'Pose {i + 1}\n({pose.x}, {pose.y}, {math.degrees(pose.angle):.0f}$^\\circ$)',
        (pose.x, pose.y), textcoords='offset points', xytext=(10, 10),
        fontsize=9, color=color,
    )

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Target Poses for Motion Generation')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()