"""Headless GIF generation for the driving eval path.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
import numpy as np
import yaml
from PIL import Image

from f110_gym.envs.f1_driving_env import MAP_CONFIGS


# Pyglet renderer constants (taken fro gym/f110_gym/envs/rendering.py).
BG_COLOR = (9 / 255, 32 / 255, 87 / 255)
WALL_COLOR = (183 / 255, 193 / 255, 222 / 255)
CAR_COLOR = '#d65ecf'  # approximates the magenta used in the pyglet screenshot
OPP_COLOR = '#5fd9a4'  # green for opponent, distinguishes from ego magenta
CAR_LENGTH_M = 0.58
CAR_WIDTH_M = 0.31
DEFAULT_VIEW_HALF_M = 12.0  # half-width of the camera-follow view in meters


def _load_map_obstacle_points(map_path: str, map_ext: str):
    """Load a pyglet-style obstacle point cloud from the map image + yaml.

    Returns
    -------
    (x, y) : 1D float arrays of obstacle pixel centers in world meters.
    """

    yaml_path = map_path + '.yaml'
    with open(yaml_path, 'r') as yaml_stream:
        meta = yaml.safe_load(yaml_stream)
    resolution = float(meta['resolution'])
    origin_x = float(meta['origin'][0])
    origin_y = float(meta['origin'][1])

    img_path = map_path + map_ext
    img = np.array(Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.29, 0.57, 0.14])

    mask = img == 0.0
    if not np.any(mask):
        mask = img < 64.0

    rows, cols = np.where(mask)
    x = cols.astype(np.float64) * resolution + origin_x
    y = rows.astype(np.float64) * resolution + origin_y
    return x, y


def _car_rect_transform(ax, x: float, y: float, yaw: float):
    """Return a matplotlib transform that places the car rectangle at (x, y, yaw).

    The rectangle is anchored at its center. Matplotlib `Rectangle` is created
    with its lower-left at origin, so we translate by (-L/2, -W/2), rotate, then
    translate into world coordinates.
    """

    t = (
        mtransforms.Affine2D()
        .translate(-CAR_LENGTH_M / 2.0, -CAR_WIDTH_M / 2.0)
        .rotate(yaw)
        .translate(x, y)
    )
    return t + ax.transData


def _style_panel_axis(ax, *, title: str):
    """Style a small bottom HUD panel axis in the pyglet-look palette."""
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color((0.75, 0.78, 0.88))
        spine.set_linewidth(0.5)
    ax.tick_params(colors=(0.75, 0.78, 0.88), labelsize=7, length=2)
    ax.set_title(title, color='white', fontsize=9, pad=2)


def write_driving_gif(
    *,
    traces: dict,
    map_name: str,
    out: str,
    fps: int = 30,
    stride: int = 4,
    title: str = 'Driving policy trajectory',
    trail: int = 80,
    view_half_m: float = DEFAULT_VIEW_HALF_M,
    v_max_mps: float = 8.0,
    steer_max_rad: float = 0.4189,
) -> None:
    """Write a pyglet-style GIF of one rollout to `out`.

    Layout: big top track view (camera-follow) + bottom gauges row showing
    velocity, SOC bar, and signed deploy command. HUD text overlays lap count,
    lap time, and cumulative reward.

    Parameters
    ----------
    traces : dict
        Must contain at least `t`, `x`, `y`. Optional fields used for the HUD
        and panels: `yaw`, `lap_count`, `v`, `soc`, `deploy`, `reward_cum`.
        Missing optional fields are treated as zeros so older traces still
        render.
    map_name : str
        Key into `MAP_CONFIGS`.
    """
    t = np.asarray(traces['t'])[::stride]
    x = np.asarray(traces['x'])[::stride]
    y = np.asarray(traces['y'])[::stride]

    yaw = np.asarray(traces.get('yaw', np.zeros_like(x)))[::stride]
    lap_count = np.asarray(traces.get('lap_count', np.zeros_like(x, dtype=np.int64)))[::stride]
    v = np.asarray(traces.get('v', np.zeros_like(x)))[::stride]
    soc = np.asarray(traces.get('soc', np.zeros_like(x)))[::stride]
    deploy = np.asarray(traces.get('deploy', np.zeros_like(x)))[::stride]
    reward_cum = np.asarray(traces.get('reward_cum', np.zeros_like(x)))[::stride]
    v_set_cmd = np.asarray(traces.get('v_set_cmd', np.zeros_like(x)))[::stride]
    steer_cmd = np.asarray(traces.get('steer_cmd', np.zeros_like(x)))[::stride]
    steer_actual = np.asarray(traces.get('steer_actual', np.zeros_like(x)))[::stride]

    # NOTE: removed cuz didn't work. opponent pose (optional, multi-agent)
    opp_x = traces.get('opp_x', None)
    opp_y = traces.get('opp_y', None)
    opp_yaw = traces.get('opp_yaw', None)
    if opp_x is not None and opp_y is not None and opp_yaw is not None:
        opp_x = np.asarray(opp_x)[::stride]
        opp_y = np.asarray(opp_y)[::stride]
        opp_yaw = np.asarray(opp_yaw)[::stride]
        draw_opp = (opp_x.shape[0] == x.shape[0])
    else:
        draw_opp = False

    # LiDAR fan inputs: per-frame min-pooled distances (T, NBINS) in meters and
    # per-bin angle offsets in the ego frame.
    lidar_bins_m = traces.get('lidar_bins_m', None)
    lidar_bin_angles = traces.get('lidar_bin_angles', None)
    if lidar_bins_m is not None and lidar_bin_angles is not None:
        lidar_bins_m = np.asarray(lidar_bins_m)[::stride]
        lidar_bin_angles = np.asarray(lidar_bin_angles)
        draw_lidar = lidar_bins_m.ndim == 2 and lidar_bins_m.shape[1] == lidar_bin_angles.shape[0]
    else:
        draw_lidar = False

    n = len(t)
    if n == 0:
        raise ValueError('No trace samples after stride; nothing to render.')

    cfg = MAP_CONFIGS[map_name]
    map_path = cfg['map_path']
    map_ext = cfg['map_ext']
    obs_x, obs_y = _load_map_obstacle_points(map_path, map_ext)

    # Figure with top track view + bottom gauge row (4 panels).
    fig = plt.figure(figsize=(9, 7.4), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[5.4, 1.0],
        left=0.0, right=1.0, top=1.0, bottom=0.02,
        wspace=0.25, hspace=0.12,
    )

    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.scatter(obs_x, obs_y, s=1.0, c=[WALL_COLOR], marker='s', linewidths=0, zorder=1)

    car = mpatches.Rectangle(
        (0.0, 0.0),
        CAR_LENGTH_M,
        CAR_WIDTH_M,
        facecolor=CAR_COLOR,
        edgecolor='white',
        linewidth=0.6,
        zorder=3,
    )
    ax.add_patch(car)

    opp_car = None
    if draw_opp:
        opp_car = mpatches.Rectangle(
            (0.0, 0.0),
            CAR_LENGTH_M,
            CAR_WIDTH_M,
            facecolor=OPP_COLOR,
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )
        ax.add_patch(opp_car)

    lidar_rays = LineCollection(
        [], colors=[(0.30, 0.85, 0.95, 0.55)], linewidths=0.6, zorder=2,
    )
    ax.add_collection(lidar_rays)

    # HUD text: lap time, lap count, cumulative reward.
    hud = ax.text(
        0.5,
        0.035,
        '',
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        color='white',
        fontsize=13,
        zorder=5,
    )

    # Optional small title strip at the top (same text color as HUD to match look).
    if title:
        ax.text(
            0.5,
            0.965,
            title,
            transform=ax.transAxes,
            ha='center',
            va='top',
            color=(0.75, 0.78, 0.88),
            fontsize=10,
            zorder=5,
        )

    ax_v = fig.add_subplot(gs[1, 0])
    _style_panel_axis(ax_v, title='Velocity [m/s] (white=cmd)')
    ax_v.set_xlim(0.0, v_max_mps)
    ax_v.set_ylim(0.0, 1.0)
    ax_v.set_yticks([])
    v_bar = ax_v.barh([0.5], [0.0], height=0.5, color='#4ec9b0', edgecolor='white', linewidth=0.5)[0]
    v_cmd_line = ax_v.axvline(0.0, color='white', linewidth=1.4, alpha=0.95)
    v_text = ax_v.text(
        0.98, 0.5, '', transform=ax_v.transAxes,
        ha='right', va='center', color='white', fontsize=10,
    )

    ax_st = fig.add_subplot(gs[1, 1])
    _style_panel_axis(ax_st, title='Steer [rad] (bar=actual, |=cmd)')
    ax_st.set_xlim(-steer_max_rad, steer_max_rad)
    ax_st.set_ylim(0.0, 1.0)
    ax_st.set_yticks([])
    ax_st.axvline(0.0, color=(0.75, 0.78, 0.88), linewidth=0.6, alpha=0.7)
    steer_bar = ax_st.barh([0.5], [0.0], left=0.0, height=0.5,
                           color='#c8a2ff', edgecolor='white', linewidth=0.5)[0]
    steer_cmd_line = ax_st.axvline(0.0, color='white', linewidth=1.4, alpha=0.95)
    steer_text = ax_st.text(
        0.98, 0.5, '', transform=ax_st.transAxes,
        ha='right', va='center', color='white', fontsize=9,
    )

    ax_s = fig.add_subplot(gs[1, 2])
    _style_panel_axis(ax_s, title='SOC')
    ax_s.set_xlim(0.0, 1.0)
    ax_s.set_ylim(0.0, 1.0)
    ax_s.set_yticks([])
    soc_bar = ax_s.barh([0.5], [0.0], height=0.5, color='#f5d76e', edgecolor='white', linewidth=0.5)[0]
    soc_text = ax_s.text(
        0.98, 0.5, '', transform=ax_s.transAxes,
        ha='right', va='center', color='white', fontsize=10,
    )

    ax_d = fig.add_subplot(gs[1, 3])
    _style_panel_axis(ax_d, title='Deploy (+) / Harvest (-)')
    ax_d.set_xlim(-1.0, 1.0)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_yticks([])
    ax_d.axvline(0.0, color=(0.75, 0.78, 0.88), linewidth=0.6, alpha=0.7)
    deploy_bar = ax_d.barh([0.5], [0.0], left=0.0, height=0.5,
                           color='#d95f5f', edgecolor='white', linewidth=0.5)[0]
    deploy_text = ax_d.text(
        0.98, 0.5, '', transform=ax_d.transAxes,
        ha='right', va='center', color='white', fontsize=10,
    )

    def _set_camera(i: int):
        ax.set_xlim(x[i] - view_half_m, x[i] + view_half_m)
        top_h_in = 7.2 * (5.4 / 6.4)
        vh = view_half_m * (top_h_in / 8.0)
        ax.set_ylim(y[i] - vh, y[i] + vh)

    def _update_car(i: int):
        car.set_transform(_car_rect_transform(ax, float(x[i]), float(y[i]), float(yaw[i])))
        if opp_car is not None:
            opp_car.set_transform(_car_rect_transform(ax, float(opp_x[i]), float(opp_y[i]), float(opp_yaw[i])))

    def _update_lidar(i: int):
        if not draw_lidar:
            return
        bins = lidar_bins_m[i]
        thetas = float(yaw[i]) + lidar_bin_angles
        xs_end = float(x[i]) + bins * np.cos(thetas)
        ys_end = float(y[i]) + bins * np.sin(thetas)
        segs = np.stack(
            [
                np.stack([np.full_like(xs_end, float(x[i])), np.full_like(ys_end, float(y[i]))], axis=-1),
                np.stack([xs_end, ys_end], axis=-1),
            ],
            axis=1,
        )
        lidar_rays.set_segments(segs)

    def _update_hud(i: int):
        hud.set_text(
            f'Lap Time: {float(t[i]):.2f}s  |  Lap: {int(lap_count[i])}  |  Return: {float(reward_cum[i]):.1f}'
        )

    def _update_panels(i: int):
        vi = float(np.clip(v[i], 0.0, v_max_mps))
        v_bar.set_width(vi)
        v_cmd_line.set_xdata([float(np.clip(v_set_cmd[i], 0.0, v_max_mps))] * 2)
        v_text.set_text(f'{float(v[i]):.2f} / {float(v_set_cmd[i]):.2f}')

        sa = float(np.clip(steer_actual[i], -steer_max_rad, steer_max_rad))
        steer_bar.set_width(sa)
        steer_bar.set_color('#c8a2ff' if sa >= 0.0 else '#ffb0c5')
        steer_cmd_line.set_xdata([float(np.clip(steer_cmd[i], -steer_max_rad, steer_max_rad))] * 2)
        steer_text.set_text(f'{sa:+.2f} / {float(steer_cmd[i]):+.2f}')

        si = float(np.clip(soc[i], 0.0, 1.0))
        soc_bar.set_width(si)
        soc_text.set_text(f'{si * 100.0:.1f}%')

        di = float(np.clip(deploy[i], -1.0, 1.0))
        deploy_bar.set_width(di)
        # color flips based on sign: red=deploy, blue=harvest
        deploy_bar.set_color('#d95f5f' if di >= 0.0 else '#5fa8d9')
        deploy_text.set_text(f'{di:+.2f}')

    def init():
        _set_camera(0)
        _update_car(0)
        _update_lidar(0)
        _update_hud(0)
        _update_panels(0)
        return (car, hud, v_bar, soc_bar, deploy_bar, lidar_rays, steer_bar)

    def update(i):
        _set_camera(i)
        _update_car(i)
        _update_lidar(i)
        _update_hud(i)
        _update_panels(i)
        return (car, hud, v_bar, soc_bar, deploy_bar, lidar_rays, steer_bar)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n,
        interval=1000 / max(1, int(fps)),
        blit=False,  
    )
    out_str = str(out_path)
    if out_str.lower().endswith('.mp4'):
        anim.save(out_str, writer=animation.FFMpegWriter(fps=int(fps),
                                                          codec='libx264',
                                                          bitrate=2400))
    else:
        anim.save(out_str, writer=animation.PillowWriter(fps=int(fps)))
    plt.close(fig)
