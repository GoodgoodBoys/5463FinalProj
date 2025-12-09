import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Button

show_animation = True
fig = plt.figure()
ax = fig.add_subplot(111)

point = []
is_drawing = False

# Setting lenth of arms
l1 = 1
l2 = 0.5

dt = 0.01

# previous location
prev_theta1 = 0.0
prev_theta2 = 0.0

if show_animation:
    plt.ion()

# get theta1
def get_theta1(x, y, theta2):
    alpha = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = math.atan2(y, x) - alpha
    return theta1

def two_joint_arm(path_points):
    global prev_theta1, prev_theta2

    for pt in path_points:
        x, y = pt
        r = np.hypot(x, y)
        theta_dir = math.atan2(y, x)
        r_min = abs(l1 - l2)
        r_max = l1 + l2

        if r < r_min:
            x = r_min * math.cos(theta_dir)
            y = r_min * math.sin(theta_dir)
        elif r > r_max:
            x = r_max * math.cos(theta_dir)
            y = r_max * math.sin(theta_dir)

        try:
            # calculate theta2(two result)
            beta = (x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
            beta = np.clip(beta, -1.0, 1.0)
            theta2_a = math.acos(beta)
            theta2_b = -theta2_a

            # calculate theta1
            theta1_a = get_theta1(x, y, theta2_a)
            theta1_b = get_theta1(x, y, theta2_b)

            # choose location close to the last one
            dist_a = (theta1_a - prev_theta1) ** 2 + (theta2_a - prev_theta2) ** 2
            dist_b = (theta1_b - prev_theta1) ** 2 + (theta2_b - prev_theta2) ** 2

            if dist_a < dist_b:
                theta1, theta2 = theta1_a, theta2_a
            else:
                theta1, theta2 = theta1_b, theta2_b

            # renew the previous location
            prev_theta1, prev_theta2 = theta1, theta2

            # calculate actual x, y
            x_actual = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
            y_actual = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)

            plot_arm(theta1, theta2, x_actual, y_actual)

        except Exception as e:
            # print(f"Error: {e}")
            continue


def plot_arm(theta1, theta2, target_x, target_y):
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + \
            np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    if show_animation:
        ax.cla()

        circle_out = plt.Circle((0, 0), l1 + l2, color='lightgray', fill=False, linestyle='-')
        ax.add_patch(circle_out)
        circle_inner = plt.Circle((0, 0), l1 - l2, color='lightgray', fill=False, linestyle='-')
        ax.add_patch(circle_inner)

        if len(point) > 0:
            path_arr = np.array(point)
            ax.plot(path_arr[:, 0], path_arr[:, 1], 'g-', linewidth=2)

        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')
        ax.plot(shoulder[0], shoulder[1], 'ro')
        ax.plot(elbow[0], elbow[1], 'ro')
        ax.plot(wrist[0], wrist[1], 'ro')

        ax.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
        ax.plot(target_x, target_y, 'g.')

        # robotic arm information
        info_text = (
            f"End-Effector Pos:\n"
            f"  X: {wrist[0]:.3f}\n"
            f"  Y: {wrist[1]:.3f}\n"
            f"Joint Angles:\n"
            f"  θ1: {np.degrees(theta1):.2f}°\n"
            f"  θ2: {np.degrees(theta2):.2f}°"
        )

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9, family='monospace')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

        plt.draw()
        plt.pause(dt)

    return wrist


# Generate square
def generate_square_event(event):
    global point

    corners = np.array([
        [0.4, -0.4],
        [1.2, -0.4],
        [1.2, 0.4],
        [0.4, 0.4],
        [0.4, -0.4]
    ])

    full_path = []
    step_size = 0.02

    for i in range(len(corners) - 1):
        start_pt = corners[i]
        end_pt = corners[i + 1]

        dist = np.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        num_steps = int(dist / step_size)

        x_segment = np.linspace(start_pt[0], end_pt[0], num_steps)
        y_segment = np.linspace(start_pt[1], end_pt[1], num_steps)

        segment = np.column_stack((x_segment, y_segment))

        if len(full_path) == 0:
            full_path = segment
        else:
            full_path = np.vstack((full_path, segment))

    point = full_path.tolist()
    two_joint_arm(full_path)


# generate circle
def generate_circle_event(event):
    global point

    # configuration
    cx, cy = 0.8, 0.0
    r = 0.5
    num_points = 200

    # generate circle
    t = np.linspace(0, 2 * np.pi, num_points)
    x = cx + r * np.cos(t)
    y = cy + r * np.sin(t)

    full_path = np.column_stack((x, y))

    point = full_path.tolist()
    two_joint_arm(full_path)


# generate 8 shape
def generate_infinity_event(event):
    global point

    # configuration
    scale = 0.5
    center_x = 0.8
    center_y = 0.0

    num_points = 300
    t = np.linspace(-np.pi, np.pi, num_points)

    x_raw = np.sin(t)
    y_raw = np.sin(t) * np.cos(t)

    # zoom
    x = center_x + scale * x_raw
    y = center_y + scale * y_raw

    full_path = np.column_stack((x, y))

    point = full_path.tolist()
    two_joint_arm(full_path)


# Press method
def on_press(event):
    global is_drawing, point
    if event.inaxes == ax and event.button == 1:
        is_drawing = True
        point = []
        point.append([event.xdata, event.ydata])
        ax.plot(event.xdata, event.ydata, 'r.')
        fig.canvas.draw()


def on_move(event):
    global is_drawing, point
    if is_drawing and event.inaxes == ax:
        point.append([event.xdata, event.ydata])
        ax.plot(event.xdata, event.ydata, 'r.', markersize=2)
        fig.canvas.draw()


def on_release(event):
    global is_drawing, point
    if event.button == 1 and is_drawing:
        is_drawing = False
        if len(point) < 2: return

        raw_points = np.array(point)
        dists = np.sqrt(np.sum(np.diff(raw_points, axis=0) ** 2, axis=1))
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        total_length = cum_dist[-1]

        if total_length == 0: return

        step_size = 0.02
        num_points = int(total_length / step_size)
        if num_points < 2: num_points = 2

        target_dists = np.linspace(0, total_length, num_points)
        new_x = np.interp(target_dists, cum_dist, raw_points[:, 0])
        new_y = np.interp(target_dists, cum_dist, raw_points[:, 1])

        path_points = np.column_stack((new_x, new_y))
        point = path_points.tolist()

        two_joint_arm(path_points)


# Main
def main():
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plot_arm(0, 0, l1 + l2, 0)

    button_refs = []

    # Square Button
    ax_btn_sq = plt.axes([0.70, 0.9, 0.09, 0.075])
    btn_square = Button(ax_btn_sq, 'Square')
    btn_square.on_clicked(generate_square_event)
    button_refs.append(btn_square)

    # Circle Button
    ax_btn_cir = plt.axes([0.80, 0.9, 0.09, 0.075])
    btn_circle = Button(ax_btn_cir, 'Circle')
    btn_circle.on_clicked(generate_circle_event)
    button_refs.append(btn_circle)

    # 8 Button
    ax_btn_inf = plt.axes([0.90, 0.9, 0.1, 0.075])
    btn_infinity = Button(ax_btn_inf, '8-Shape')
    btn_infinity.on_clicked(generate_infinity_event)
    button_refs.append(btn_infinity)

    plt.show()
    return button_refs


if __name__ == "__main__":
    btn_refs = main()
    plt.pause(1000)