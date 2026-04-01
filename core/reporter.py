import math
import time
import textwrap 

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF export
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from cycler import cycler


palette = [
    "#E74C3C",  # Red
    "#3F51B5",  # Indigo
    "#E67E22",  # Orange
    "#F1C40F",  # Yellow
    "#2ECC71",  # Green
    "#3498DB",  # Blue
    "#9B59B6",  # Violet
    "#222222",  # Black
]

# Global Matplotlib style (clean, readable for PDF)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    # Subplot margins
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.05,
    'figure.subplot.top': 0.95,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Color palette (avoid yellow)
    'axes.prop_cycle': cycler('color', palette)
})

# Default marker size for time-series plots
DEFAULT_MARKER_SIZE = 1


def add_section_header(fig, title, *, banner_height=0.06, page_num=None):
    """
    Draw a slim banner on top of the figure with a white, bold title.
    """
    header_color = "#0F172A"  # slate-900
    fig.add_artist(Rectangle(
        (0, 1.0 - banner_height), 1.0, banner_height,
        transform=fig.transFigure, color=header_color, zorder=1000
    ))
    fig.text(
        0.06, 1.0 - banner_height / 2.0, title,
        ha="left", va="center",
        fontsize=16, fontweight="bold", color="white",
        path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.15)],
        zorder=1001
    )
    if page_num is not None:
        fig.text(
            0.98, 1.0 - banner_height / 2.0, str(page_num),
            ha="right", va="center",
            fontsize=8, color="white",
            path_effects=[pe.withStroke(linewidth=1, foreground="black", alpha=0.15)],
            zorder=1001
        )


class Reporter:
    def __init__(self, report_path, config):
        """
        Initialize a report generator.
        Args:
            report_path: Output PDF path.
            config: Dict of configuration for rendering the 'Configuration' table.
        """
        self.report_path = report_path
        self.config = config
        self.history = {}
        self.timesteps = 0

    def write_info(self, info):
        """
        Append one timestep of logged info.
        """
        self.timesteps += 1
        for key, value in info.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def _build_config_rows(self, config, indent=0):
        """
        Flatten nested dict into 2-column rows: [Parameter, Value].
        Indentation is represented by 4 spaces per nesting level.
        """
        rows = []
        indent_str = "    " * indent
        for key, value in config.items():
            if isinstance(value, dict):
                rows.append([f"{indent_str}{key}", ""])
                rows.extend(self._build_config_rows(value, indent + 1))
            elif isinstance(value, list):
                joined = ", ".join(map(str, value))
                rows.append([f"{indent_str}{key}", joined])
            else:
                rows.append([f"{indent_str}{key}", str(value)])
        return rows

    def _wrap_long_text(self, s: str, *, width=38, max_lines=8):
        """
        Wrap a long string to a fixed width. If it exceeds max_lines,
        trim and add a tail marker like: '… (+N more)'.
        """
        if not isinstance(s, str):
            s = str(s)
        wrapped_lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
        if len(wrapped_lines) <= max_lines:
            return "\n".join(wrapped_lines), len(wrapped_lines)
        hidden = len(wrapped_lines) - max_lines
        trimmed = wrapped_lines[:max_lines] + [f"… (+{hidden} more)"]
        return "\n".join(trimmed), len(trimmed)

    def generate_report(self):
        """
        Build the multi-page PDF report.
        """
        PAGE_SIZE = (8.27, 11.69)  # A4 portrait
        dt = float(self.history.get('dt', [1])[0])
        times = np.arange(self.timesteps) * dt

        # Safe defaults for page counts
        PAGE_NUM_IN_SECTION_1 = 0
        PAGE_NUM_IN_SECTION_2 = 0
        PAGE_NUM_IN_SECTION_3 = 0

        with PdfPages(self.report_path) as pdf:
            # -------------------------------
            # Cover page
            # -------------------------------
            fig_cover = plt.figure(figsize=PAGE_SIZE)
            bg_ax = fig_cover.add_axes([0, 0, 1, 1], zorder=0)
            bg_ax.axis("off")

            # Soft vertical gradient background
            h, w = 800, 1200
            top_rgb = np.array([246, 248, 252]) / 255.0
            bottom_rgb = np.array([230, 236, 255]) / 255.0
            t = np.linspace(0.0, 1.0, h)[:, None]
            grad = (top_rgb * (1.0 - t) + bottom_rgb * t)
            grad = np.repeat(grad[None, ...], w, axis=0).transpose(1, 0, 2)
            bg_ax.imshow(grad, aspect="auto", extent=[0, 1, 0, 1], origin="lower")

            header_color = "#0F172A"
            fig_cover.add_artist(Rectangle((0, 0.82), 1.0, 0.18, color=header_color, zorder=1))

            fig_cover.text(
                0.08, 0.90, "Test Report",
                ha="left", va="center",
                fontsize=34, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.15)]
            )
            fig_cover.text(
                0.08, 0.86, "Performance Summary",
                ha="left", va="center",
                fontsize=14, color="#CBD5E1"
            )

            generated_on = time.strftime("%Y-%m-%d %H:%M:%S")
            duration_sec = int(self.timesteps * dt)
            mins, secs = divmod(duration_sec, 60)
            hours, mins = divmod(mins, 60)
            duration_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
            env_id = (self.config.get("env", {}) or {}).get("id", "N/A")

            meta_items = [
                ("Env ID", str(env_id)),
                ("Duration", duration_str),
                ("Generated", generated_on),
            ]

            chip_y = 0.66
            chip_w = 0.22
            chip_h = 0.08
            chip_gap = 0.045

            total_w = 3 * chip_w + 2 * chip_gap
            x_start = 0.5 - total_w / 2.0
            chip_xs = [x_start, x_start + chip_w + chip_gap, x_start + 2 * (chip_w + chip_gap)]

            for (label, value), cx in zip(meta_items, chip_xs):
                chip = FancyBboxPatch(
                    (cx, chip_y), chip_w, chip_h,
                    boxstyle="round,pad=0.013,rounding_size=0.022",
                    linewidth=1.0, edgecolor="#E2E8F0", facecolor="white", zorder=2, transform=fig_cover.transFigure
                )
                fig_cover.add_artist(chip)
                fig_cover.text(cx + 0.014, chip_y + chip_h*0.62, label,
                               ha="left", va="center", fontsize=10, color="#64748B")
                fig_cover.text(cx + 0.014, chip_y + chip_h*0.27, value,
                               ha="left", va="center", fontsize=13, fontweight="bold", color="#0F172A")

            fig_cover.add_artist(Rectangle((0.08, 0.605), 0.84, 0.002, color="#CBD5E1", zorder=1))

            description = (
                "This report presents a concise overview of key performance metrics,\n"
                "including the following analyses:"
            )
            desc_y = 0.56
            fig_cover.text(0.5, desc_y, description, ha="center", va="top", fontsize=13, fontweight="bold", color="#0F172A")

            bullets = [
                "•  Set Points vs. States",
                "•  Command Inputs vs. Measured Outputs",
                "•  Action Oscillation",
                "•  Applied Torques",
            ]
            line_gap = 0.045
            by = desc_y - 0.07
            for b in bullets:
                fig_cover.text(0.5, by, b, ha="center", va="center", fontsize=12, color="#111827")
                by -= line_gap

            fig_cover.add_artist(Rectangle((0.08, 0.08), 0.84, 0.0015, color="#E5E7EB", zorder=1))
            current_year = time.strftime("%Y")
            fig_cover.text(
                0.08, 0.05, f"{current_year} COCELO Ltd. - Automatically Generated",
                ha="left", va="center", fontsize=9, color="#6B7280"
            )

            pdf.savefig(fig_cover)
            plt.close(fig_cover)

            # -------------------------------
            # 1) Set Points vs. States
            # -------------------------------
            if ('set_points' in self.history and 'state' in self.history):
                set_points = np.array(self.history['set_points'], dtype=float)
                state = np.array(self.history['state'], dtype=float)
                if set_points.ndim == 1:
                    set_points = set_points.reshape(-1, 1)
                    state = state.reshape(-1, 1)

                n_dims = set_points.shape[1]
                MAX_PLOTS_PER_PAGE_IN_SECTION_1 = 8
                N_COLS = 2

                START_PAGE_NO = 1
                PAGE_NUM_IN_SECTION_1 = len(list(range(0, n_dims, MAX_PLOTS_PER_PAGE_IN_SECTION_1)))
                for i, start in enumerate(range(0, n_dims, MAX_PLOTS_PER_PAGE_IN_SECTION_1)):
                    end = min(start + MAX_PLOTS_PER_PAGE_IN_SECTION_1, n_dims)
                    dims_this_page = end - start
                    n_rows = math.ceil(dims_this_page / N_COLS)

                    fig, axes = plt.subplots(n_rows, N_COLS, figsize=PAGE_SIZE)
                    add_section_header(fig, "Set Points vs. States", page_num=START_PAGE_NO + i)

                    if isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    else:
                        axes = np.array([axes])

                    for local_idx, dim_idx in enumerate(range(start, end)):
                        ax = axes[local_idx]
                        ax.plot(times, set_points[:, dim_idx], linestyle='-',
                                markersize=DEFAULT_MARKER_SIZE, label="Set Point")
                        ax.plot(times, state[:, dim_idx], linestyle='-.',
                                markersize=DEFAULT_MARKER_SIZE, label="State")
                        ax.set_xlabel("Time (s)", fontsize=10)
                        ax.set_ylabel(f"Set Point", fontsize=10)
                        ax.set_title(f"Dimension {dim_idx}", fontsize=12)
                        ax.legend(fontsize=8)
                        ax.grid(True)

                    for j in range(dims_this_page, len(axes)):
                        fig.delaxes(axes[j])

                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    pdf.savefig(fig)
                    plt.close(fig)

            # -------------------------------
            # 2) Command Inputs vs. Measured Outputs
            # -------------------------------
            if "user_command_2" in self.history:
                command_keys = [
                ("user_command_0", "lin_vel_x", "Linear Velocity X"),
                ("user_command_1", "lin_vel_y", "Linear Velocity Y"),
                ("user_command_2", "ang_vel_yaw", "Angular Velocity Yaw"),
            ]
            else:
                command_keys = [
                    ("user_command_0", "lin_vel_x", "Linear Velocity X"),
                    ("user_command_1", "ang_vel_yaw", "Angular Velocity Yaw"),
                ]

            plot_keys = [
                (cmd_key, actual_key, label)
                for (cmd_key, actual_key, label) in command_keys
                if (cmd_key in self.history and actual_key in self.history)
            ]

            MAX_PLOTS_PER_PAGE_IN_SECTION_2 = 4

            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + 1
            PAGE_NUM_IN_SECTION_2 = len(list(range(0, len(plot_keys), MAX_PLOTS_PER_PAGE_IN_SECTION_2)))
            if plot_keys:
                for i, start in enumerate(range(0, len(plot_keys), MAX_PLOTS_PER_PAGE_IN_SECTION_2)):
                    page_keys = plot_keys[start:start + MAX_PLOTS_PER_PAGE_IN_SECTION_2]
                    n_rows = len(page_keys)

                    fig, axes = plt.subplots(n_rows, 1, figsize=PAGE_SIZE)
                    add_section_header(fig, "Command Inputs vs. Measured Outputs", page_num=START_PAGE_NO + i)

                    if not isinstance(axes, (list, np.ndarray)):
                        axes = [axes]

                    for idx, (cmd_key, actual_key, label) in enumerate(page_keys):
                        cmd_values = np.array(self.history[cmd_key], dtype=float)
                        actual_values = np.array(self.history[actual_key], dtype=float)
                        ax = axes[idx]

                        ax.plot(times, cmd_values, linestyle='-', markersize=DEFAULT_MARKER_SIZE,
                                label=f"{cmd_key}")
                        ax.plot(times, actual_values, linestyle='--', markersize=DEFAULT_MARKER_SIZE,
                                label=f"{label}" if label != 'Position Z' else f"{label} (Absolute)")
                        ax.set_xlabel("Time (s)", fontsize=10)
                        ax.set_ylabel(label, fontsize=10)
                        ax.set_title(label, fontsize=12)
                        ax.legend(fontsize=8)
                        ax.grid(True)

                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    pdf.savefig(fig)
                    plt.close(fig)

            # -------------------------------
            # 3) Action Oscillation and Applied Torques
            # -------------------------------
            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + PAGE_NUM_IN_SECTION_2 + 1
            PAGE_NUM_IN_SECTION_3 = 0
            if ('torque' in self.history and 'action_diff_RMSE' in self.history):
                diffs = np.array(self.history['action_diff_RMSE'], dtype=float)
                torque_arr = np.array(self.history['torque'], dtype=float)
                if torque_arr.ndim == 1:
                    torque_arr = torque_arr.reshape(-1, 1)

                fig, axes = plt.subplots(3, 1, figsize=PAGE_SIZE)
                add_section_header(fig, "Action Oscillation and Applied Torques", page_num=START_PAGE_NO)
                PAGE_NUM_IN_SECTION_3 = 1

                # (a) Δa (RMSE) time-series
                axes[0].plot(times, diffs, linestyle='-', label="Δa (RMSE)")

                # --- Moving Average (same graph) ---
                # window from config if provided; default = 20
                ma_win = max(1, min(20, self.timesteps//2))
   
                # compute simple moving average and align to times
                kernel = np.ones(ma_win, dtype=float) / float(ma_win)
                ma_vals = np.convolve(diffs, kernel, mode='valid')
                # pad the front with NaNs so length matches times
                ma_aligned = np.concatenate([np.full(ma_win - 1, np.nan), ma_vals])

                axes[0].plot(times, ma_aligned, linestyle='--', label=f"Δa (RMSE) Moving Average (window={ma_win})")
                # -----------------------------------

                axes[0].set_xlabel("Time (s)", fontsize=10)
                axes[0].set_ylabel("Δa (RMSE)", fontsize=10)
                axes[0].set_title("Action Oscillation", fontsize=12)
                axes[0].legend(fontsize=8)
                axes[0].grid(True)

                # (b) Torques per joint
                n_torques = torque_arr.shape[1]
                LINESTYLES = ["-", "--", ":", "-."]

                for i in range(n_torques):
                    color = palette[i % 8]
                    ls = LINESTYLES[(i // 8) % len(LINESTYLES)]
                    axes[1].plot(times, torque_arr[:, i],linestyle=ls, color=color, label=f"Torque {i}")

                axes[1].set_xlabel("Time (s)", fontsize=10)
                axes[1].set_ylabel("Torque", fontsize=10)
                axes[1].set_title("Applied Torque of Each Joint", fontsize=12)
                
                if n_torques <= 8:
                    axes[1].legend(fontsize=7, ncol=2)
                elif n_torques <= 16:
                    axes[1].legend(fontsize=6, ncol=4)
                else:
                    axes[1].legend(fontsize=6, ncol=6)

                axes[1].grid(True)

                # (c) Torque distribution
                all_torque = torque_arr.ravel()
                p5, p95 = np.percentile(all_torque, [5, 95])
                axes[2].set_xlabel("Torque (N·m)", fontsize=10)
                axes[2].set_ylabel("Count", fontsize=10)
                axes[2].set_title("Torque Distribution of All Joints", fontsize=12)
                axes[2].hist(all_torque, color="#B1DBF7", bins=60, alpha=0.9, edgecolor='black')
                axes[2].axvline(p5, color="#E74C3C", linestyle='--', linewidth=1.2, label=f"5th: {p5:.3g}")
                axes[2].axvline(p95, color="#E74C3C", linestyle=':', linewidth=1.2, label=f"95th: {p95:.3g}")

                mean_val = np.mean(all_torque)
                mean_abs_val = np.mean(np.abs(all_torque))

                axes[2].axvline(mean_val, color="#00FF6A", linestyle='-.', linewidth=1.2, label=f"Average: {mean_val:.3g}")
                axes[2].axvline(mean_abs_val, color="#222222", linestyle='-.', linewidth=1.2, label=f"|Average|: {mean_abs_val:.3g}")

                axes[2].legend(fontsize=8)
                axes[2].grid(True)


                fig.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig)
                plt.close(fig)

            # -------------------------------
            # Final) Configuration Table
            # -------------------------------
            filtered_config = {k: v for k, v in self.config.items() if k in ["env", "policy", "settings", "random", "hardware", "action_scales"]}
            table_data = self._build_config_rows(filtered_config)
            MAX_ROWS_PER_PAGE = 50
            total_rows = len(table_data)
            n_pages = math.ceil(total_rows / MAX_ROWS_PER_PAGE) if total_rows > 0 else 1

            START_PAGE_NO = PAGE_NUM_IN_SECTION_1 + PAGE_NUM_IN_SECTION_2 + PAGE_NUM_IN_SECTION_3 + 1
            for i, page in enumerate(range(n_pages)):
                start_idx = page * MAX_ROWS_PER_PAGE
                end_idx = start_idx + MAX_ROWS_PER_PAGE
                page_data = table_data[start_idx:end_idx]

                fig_config = plt.figure(figsize=PAGE_SIZE)
                ax = fig_config.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                add_section_header(fig_config, "Configuration", page_num=START_PAGE_NO + i)

                table = ax.table(
                    cellText=page_data,
                    colLabels=["Parameter", "Value"],
                    loc="upper center",
                    cellLoc='left',
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.2)  # Keep original scale; DO NOT change column widths

                # Header/body base styling; detect a base row height
                base_h = None
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_text_props(fontweight='bold', color='white')
                        cell.set_facecolor("#40466e")
                    else:
                        if base_h is None:
                            base_h = cell.get_height()
                        cell.set_facecolor("#f1f1f2")
                        if col == 0:
                            left_text = cell.get_text().get_text().strip()
                            if left_text in ["env", "policy", "settings", "random", "hardware", "action_scales"]:
                                cell.set_text_props(fontweight='bold')

                if base_h is None:
                    base_h = 0.05  # Fallback if table implementation changes

                # Make left-column padding and alignment UNIFORM for all body rows
                for (row, col), cell in table.get_celld().items():
                    if row > 0 and col == 0:
                        # Set consistent left padding and alignment so all labels start at same x
                        try:
                            cell.PAD = 0.02
                        except Exception:
                            pass
                        cell.get_text().set_ha("left")
                        cell.get_text().set_va("center")

                # (1) Force the "settings" section header's Value cell to be empty
                settings_header_row = None
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        continue
                    if col == 0:
                        left_key = cell.get_text().get_text().strip()
                        if left_key == "settings":
                            settings_header_row = row
                            if (row, 1) in table.get_celld():
                                vcell = table.get_celld()[(row, 1)]
                                vcell.get_text().set_text("")  # Keep blank (never show text)

                # (2) Wrap ONLY the two target keys; center-align; adjust row height; keep others untouched
                target_keys = {"stacked_obs_order", "non_stacked_obs_order"}
                for (row, col), cell in list(table.get_celld().items()):
                    # Only body rows, Value column
                    if row == 0 or col != 1:
                        continue
                    # Never touch the settings header's Value cell
                    if settings_header_row is not None and row == settings_header_row:
                        continue

                    # Read the left key for this row (strip indentation)
                    if row - 1 < len(page_data):
                        left_key = page_data[row - 1][0].lstrip()
                    else:
                        left_key = ""

                    if left_key not in target_keys:
                        continue

                    value_text = cell.get_text().get_text()
                    if not value_text:
                        continue

                    # Wrap text and set it back
                    wrapped, n_lines = self._wrap_long_text(value_text, width=50, max_lines=8)
                    vtxt = cell.get_text()
                    vtxt.set_text(wrapped)
                    vtxt.set_ha("left")
                    vtxt.set_va("center")  # vertical center for consistent look

                    # Row height with generous growth and safety for multi-line
                    growth = 0.70                                # per extra line multiplier
                    safety = 0.012 if n_lines > 1 else 0.0       # absolute safety margin
                    new_h = base_h * (1.0 + growth * (n_lines - 1)) + safety

                    # Apply same height to Parameter cell and keep its text vertically centered
                    if (row, 0) in table.get_celld():
                        lcell = table.get_celld()[(row, 0)]
                        lcell.set_height(new_h)
                        lcell.get_text().set_va("center")
                        lcell.get_text().set_ha("left")  # ensure left-aligned label

                    cell.set_height(new_h)

                fig_config.tight_layout(rect=[0, 0, 1, 0.92])
                pdf.savefig(fig_config)
                plt.close(fig_config)

        print(f"Report successfully saved to {self.report_path}")
