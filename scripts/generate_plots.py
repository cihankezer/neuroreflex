"""Render actual simulated geometry from the demo trace; no synthetic illustration."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from simulation_core import forward_kinematics, ARM_RADIUS, BALL_RADIUS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=Path("outputs/demo/ticks.csv"))
    parser.add_argument("--output", type=Path, default=Path("assets/demos/synthetic_reflex.gif"))
    args = parser.parse_args()
    with args.ticks.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 4.5), facecolor="#101b2b")
    ax = fig.add_subplot(projection="3d", facecolor="#101b2b")
    ax.set(xlim=(-.15, .85), ylim=(-.05, .85), zlim=(-.8, 2.1),
           xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.tick_params(colors="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
    arm, = ax.plot([], [], [], "o-", color="#58d6c7", linewidth=9, markersize=5)
    obstacle, = ax.plot([], [], [], "o", color="#ffab69", markersize=12)
    title = fig.suptitle("", color="white", fontsize=10, y=0.97)
    def update(i):
        row=rows[i]
        ex, ey, hx, hy = forward_kinematics(float(row["shoulder_angle_deg"]), float(row["elbow_angle_deg"]))
        arm.set_data_3d([0, ex, hx], [0, ey, hy], [0, 0, 0])
        obstacle.set_data_3d([float(row["true_x"])], [float(row["true_y"])], [float(row["true_z"])])
        title.set_text(f"NeuroReflex | synthetic trajectory | t={float(row['time_s']):.2f} s\n{row['reflex_state']} | P controller | CPU simulation")
        return arm, obstacle, title
    ax.view_init(elev=22, azim=-58)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.04, right=0.94)
    FuncAnimation(fig, update, frames=range(0, len(rows), 2), interval=40).save(args.output, writer=PillowWriter(fps=25))
    plt.close(fig)
    print(args.output)

if __name__ == "__main__":
    main()
