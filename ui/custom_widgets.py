import math

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QComboBox, QPushButton, QSlider, QWidget


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class NonClickableButton(QPushButton):
    def mousePressEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()


class MujocoOverlayWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__(None)
        self._payload = {}
        self._is_exporting = False
        self._plot_bg_color = "#000000"
        self._torque_color = "#7DD3FC"
        self._velocity_color = "#F59E0B"
        self.resize(1120, 760)
        self.setMinimumSize(860, 560)
        self.setWindowTitle("Motor Time-Series Monitor")
        self.setStyleSheet("background-color: #000000;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)

    def clear_overlay(self):
        self._payload = {}
        self.hide()

    def update_overlay(self, payload: dict):
        if not payload or not payload.get("joints"):
            self.clear_overlay()
            return
        self._payload = dict(payload)
        env_id = str(self._payload.get("env_id", "robot")).replace("_", " ").title()
        self.setWindowTitle(f"Motor Time-Series Monitor | {env_id}")
        self.show()
        self.raise_()
        self.update()

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _draw_text(self, painter, x, y, text, color, size=10, bold=False, align=Qt.AlignLeft):
        font = QFont("DejaVu Sans", max(1, int(size)))
        font.setBold(bold)
        painter.setFont(font)
        painter.setPen(QColor(color))
        rect = QRectF(x, y - size * 1.4, 640, size * 2.0)
        painter.drawText(rect, align | Qt.AlignVCenter, text)

    def _fit_text(self, text, size, bold, max_width):
        font = QFont("DejaVu Sans", max(1, int(size)))
        font.setBold(bold)
        metrics = QFontMetrics(font)
        return metrics.elidedText(str(text), Qt.ElideRight, max(24, int(max_width)))

    @staticmethod
    def _short_joint_label(name):
        label = str(name).replace("_joint", "")
        if label.startswith("left_"):
            label = "L " + label[5:]
        elif label.startswith("right_"):
            label = "R " + label[6:]
        return label.replace("_", " ")

    def _draw_shell(self, painter, rect):
        shell = QPainterPath()
        shell.addRoundedRect(rect, 16, 16)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(8, 8, 8))
        painter.fillPath(shell, grad)
        painter.setPen(QPen(QColor(185, 185, 185), 1.2))
        painter.drawPath(shell)

    def _draw_card(self, painter, rect):
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(10, 10, 10))
        grad.setColorAt(1.0, QColor(20, 20, 20))
        painter.fillPath(path, grad)
        painter.setPen(QPen(QColor(110, 110, 110), 1.0))
        painter.drawPath(path)

    def _draw_badge(self, painter, rect, label, value, accent):
        badge = QPainterPath()
        badge.addRoundedRect(rect, 8, 8)
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(14, 14, 14))
        painter.fillPath(badge, grad)
        painter.setPen(QPen(QColor(accent), 1.0))
        painter.drawPath(badge)
        self._draw_text(painter, rect.left() + 10, rect.top() + 16, label, "#FFFFFF", 8, True)
        self._draw_text(painter, rect.left() + 10, rect.top() + 38, value, "#F8FAFC", 12, True)

    def _draw_usage_bar(self, painter, rect, ratio):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(28, 28, 28))
        painter.drawRoundedRect(rect, 5, 5)

        fill = QRectF(rect)
        fill.setWidth(rect.width() * self._clamp(ratio, 0.0, 1.0))
        if fill.width() <= 0:
            return
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawRoundedRect(fill, 5, 5)

    def _compute_plot_range(self, values, fallback_limit):
        recent_window = 48
        numeric_values = [float(v) for v in values[-recent_window:]]
        if not numeric_values:
            return -fallback_limit, fallback_limit

        vmin = min(numeric_values)
        vmax = max(numeric_values)
        span = vmax - vmin
        latest = numeric_values[-1]

        if span < 1e-6:
            base_margin = max(abs(latest) * 0.12, 0.15)
            return latest - base_margin, latest + base_margin

        margin = max(span * 0.18, abs(latest) * 0.05, 0.08)
        lower = vmin - margin
        upper = vmax + margin

        if lower > 0.0:
            lower = min(0.0, lower - margin * 0.35)
        if upper < 0.0:
            upper = max(0.0, upper + margin * 0.35)
        return lower, upper

    def _draw_time_series_plot(self, painter, rect, values, limit, line_color, label, dt, inline_label=None):
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#000000"))
        painter.drawRoundedRect(rect, 10, 10)

        path = QPainterPath()
        path.addRoundedRect(rect, 10, 10)
        painter.fillPath(path, QColor(self._plot_bg_color))
        painter.setPen(QPen(QColor("#000000"), 1.0))
        painter.drawPath(path)

        left = rect.left() + 58
        right = rect.right() - 12
        top = rect.top() + 22
        bottom = rect.bottom() - 30
        inner = QRectF(left, top, right - left, bottom - top)

        painter.fillRect(inner, QColor("#000000"))

        painter.setClipRect(inner.adjusted(-1, -1, 1, 1))
        grid_pen = QPen(QColor(70, 70, 70), 1.0, Qt.DashLine)
        painter.setPen(grid_pen)
        for i in range(5):
            x = inner.left() + inner.width() * i / 4.0
            painter.drawLine(QPointF(x, inner.top()), QPointF(x, inner.bottom()))
        for i in range(5):
            y = inner.top() + inner.height() * i / 4.0
            painter.drawLine(QPointF(inner.left(), y), QPointF(inner.right(), y))

        plot_min, plot_max = self._compute_plot_range(values, limit)
        plot_span = max(plot_max - plot_min, 1e-6)
        if plot_min <= 0.0 <= plot_max:
            zero_ratio = (plot_max - 0.0) / plot_span
            zero_y = inner.top() + self._clamp(zero_ratio, 0.0, 1.0) * inner.height()
            painter.setPen(QPen(QColor(185, 185, 185), 1.0))
            painter.drawLine(QPointF(inner.left(), zero_y), QPointF(inner.right(), zero_y))

        if len(values) >= 2:
            points = []
            for idx, value in enumerate(values):
                x = inner.left() + inner.width() * idx / max(1, len(values) - 1)
                y_ratio = (plot_max - float(value)) / plot_span
                y = inner.top() + self._clamp(y_ratio, 0.0, 1.0) * inner.height()
                points.append(QPointF(x, y))

            painter.setPen(QPen(QColor(line_color), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a, b)

            last = points[-1]
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
            painter.setBrush(QColor(line_color))
            painter.drawEllipse(last, 3.8, 3.8)

        painter.setClipping(False)
        header_text = label if not inline_label else f"{label}  |  {inline_label}"
        fitted_header = self._fit_text(header_text, 8, True, rect.width() - 20)
        self._draw_text(painter, rect.left() + 10, rect.top() + 14, fitted_header, "#FFFFFF", 8, True)
        self._draw_text(painter, rect.left() + 8, inner.top() + 6, f"{plot_max:+.1f}", "#FFFFFF", 9, True)
        self._draw_text(painter, rect.left() + 8, rect.bottom() - 28, f"{plot_min:+.1f}", "#FFFFFF", 9, True)
        duration = max(0.0, (len(values) - 1) * dt)
        self._draw_text(painter, inner.left(), rect.bottom() - 6, "0.0s", "#FFFFFF", 8, True)
        self._draw_text(painter, inner.center().x() - 20, rect.bottom() - 6, "time", "#FFFFFF", 8, True)
        self._draw_text(painter, inner.right() - 46, rect.bottom() - 6, f"{duration:.1f}s", "#FFFFFF", 8, True)
        painter.restore()

    def _draw_joint_panel(self, painter, rect, joint, dt):
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#000000"))
        painter.drawRoundedRect(rect, 12, 12)
        self._draw_card(painter, rect)
        name = str(joint.get("joint", "joint"))
        torque = float(joint.get("torque", 0.0))
        velocity = float(joint.get("velocity", 0.0))
        torque_limit = max(abs(float(joint.get("torque_limit", 1.0))), 1.0)
        velocity_limit = max(abs(float(joint.get("velocity_limit", 1.0))), 1.0)
        history = list(joint.get("history", []))
        torque_values = [float(tau) for _, tau in history]
        velocity_values = [float(vel) for vel, _ in history]
        short_name = self._short_joint_label(name)

        pad = 16
        title_max_width = rect.width() - 360 if not self._is_exporting else rect.width() - 40
        fitted_name = self._fit_text(short_name, 12, True, title_max_width)
        self._draw_text(painter, rect.left() + pad, rect.top() + 26, fitted_name, "#FFFFFF", 12, True)
        plot_top = rect.top() + 78
        plot_height = (rect.height() - 118) * 0.48
        if not self._is_exporting:
            badge_w = 150
            badge_h = 50
            torque_badge = QRectF(rect.right() - (badge_w * 2 + 26), rect.top() + 14, badge_w, badge_h)
            speed_badge = QRectF(rect.right() - (badge_w + 12), rect.top() + 14, badge_w, badge_h)
            self._draw_badge(painter, torque_badge, "LIVE TORQUE", f"{torque:+6.1f} Nm", self._torque_color)
            self._draw_badge(painter, speed_badge, "LIVE SPEED", f"{velocity:+6.2f} rad/s", self._velocity_color)
            self._draw_text(painter, rect.left() + pad, rect.top() + 54, f"Samples {len(history):4d}", "#FFFFFF", 10, False)
        else:
            self._draw_text(painter, rect.left() + pad, rect.top() + 54, f"Samples {len(history):4d}", "#FFFFFF", 10, False)
            plot_top = rect.top() + 70
            plot_height = (rect.height() - 110) * 0.48

        top_plot = QRectF(rect.left() + pad, plot_top, rect.width() - pad * 2, plot_height)
        bottom_plot = QRectF(rect.left() + pad, top_plot.bottom() + 16, rect.width() - pad * 2, plot_height)
        self._draw_time_series_plot(painter, top_plot, torque_values, torque_limit, self._torque_color, "torque (Nm)", dt, short_name)
        self._draw_time_series_plot(painter, bottom_plot, velocity_values, velocity_limit, self._velocity_color, "velocity (rad/s)", dt, short_name)
        painter.restore()

    def paintEvent(self, event):
        if not self._payload:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#000000"))
        shell_rect = QRectF(self.rect().adjusted(10, 10, -10, -10))
        self._draw_shell(painter, shell_rect)

        env_id = str(self._payload.get("env_id", "robot")).replace("_", " ").title()
        joints = list(self._payload.get("joints", []))
        dt = float(self._payload.get("dt", 0.02))
        self._draw_text(painter, shell_rect.left() + 18, shell_rect.top() + 28, "Motor Time-Series Monitor", "#FFFFFF", 15, True)
        self._draw_text(painter, shell_rect.left() + 18, shell_rect.top() + 54, f"{env_id} | {len(joints)} channels | dt {dt:.3f}s", "#FFFFFF", 10, False)

        columns = 2 if len(joints) > 1 else 1
        rows = int(math.ceil(max(1, len(joints)) / columns))
        top = shell_rect.top() + 72
        left = shell_rect.left() + 16
        gap = 14
        panel_w = (shell_rect.width() - 32 - gap * (columns - 1)) / columns
        panel_h = (shell_rect.height() - 80 - gap * (rows - 1)) / rows

        for idx, joint in enumerate(joints):
            row = idx // columns
            col = idx % columns
            rect = QRectF(
                left + col * (panel_w + gap),
                top + row * (panel_h + gap),
                panel_w,
                panel_h,
            )
            self._draw_joint_panel(painter, rect, joint, dt)

    def save_timeseries_report(self, output_path: str, payload: dict):
        if not payload or not payload.get("joints"):
            raise ValueError("No monitor data available to save.")

        image = QImage(self.size(), QImage.Format_ARGB32)
        image.fill(QColor("#000000"))
        original_payload = self._payload
        original_export_state = self._is_exporting
        self._payload = dict(payload)
        self._is_exporting = True
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.render(painter)
        painter.end()
        self._payload = original_payload
        self._is_exporting = original_export_state

        if not image.save(output_path):
            raise RuntimeError(f"Failed to save monitor report to {output_path}")

    def closeEvent(self, event):
        self._payload = {}
        self.closed.emit()
        super().closeEvent(event)
