"""Optional target-center filters used by the YOLO detector."""

import numpy as np


TARGET_FILTER_NAMES = ("none", "kalman")


class PixelKalmanFilter:
    """Constant-acceleration Kalman filter for image-space target centers."""

    def __init__(self, process_noise=800.0, measurement_noise=25.0):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initialized = False
        self.last_time = None
        self.state = np.zeros((6, 1), dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64) * 1000.0

    def reset(self):
        self.initialized = False
        self.last_time = None
        self.state.fill(0.0)
        self.covariance = np.eye(6, dtype=np.float64) * 1000.0

    def update(self, x, y, timestamp):
        if not self.initialized:
            self.state = np.array(
                [[x], [y], [0.0], [0.0], [0.0], [0.0]],
                dtype=np.float64,
            )
            self.covariance = np.diag(
                [25.0, 25.0, 2500.0, 2500.0, 10000.0, 10000.0]
            ).astype(np.float64)
            self.initialized = True
            self.last_time = timestamp
            return

        dt = max(1e-3, min(timestamp - self.last_time, 1.0))
        self.last_time = timestamp
        dt2 = dt * dt

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        dt5 = dt4 * dt
        dt6 = dt3 * dt3
        q = self.process_noise
        process = q * np.array(
            [
                [dt6 / 36.0, 0.0, dt5 / 12.0, 0.0, dt4 / 6.0, 0.0],
                [0.0, dt6 / 36.0, 0.0, dt5 / 12.0, 0.0, dt4 / 6.0],
                [dt5 / 12.0, 0.0, dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt5 / 12.0, 0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt4 / 6.0, 0.0, dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt4 / 6.0, 0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

        measurement = np.array([[x], [y]], dtype=np.float64)
        observation = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        noise = np.eye(2, dtype=np.float64) * self.measurement_noise
        residual = measurement - observation @ self.state
        residual_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(residual_cov)

        self.state = self.state + gain @ residual
        identity = np.eye(6, dtype=np.float64)
        self.covariance = (identity - gain @ observation) @ self.covariance

    def predict(self, lead_time):
        if not self.initialized:
            return None
        lead_time = max(0.0, float(lead_time))
        lead_time2 = lead_time * lead_time
        x = (
            self.state[0, 0]
            + self.state[2, 0] * lead_time
            + 0.5 * self.state[4, 0] * lead_time2
        )
        y = (
            self.state[1, 0]
            + self.state[3, 0] * lead_time
            + 0.5 * self.state[5, 0] * lead_time2
        )
        return float(x), float(y)


def create_target_filter(
    filter_name,
    *,
    kalman_process_noise=800.0,
    kalman_measurement_noise=25.0,
):
    """Create the selected target filter; ``none`` is a true bypass."""
    if filter_name == "none":
        return None
    if filter_name == "kalman":
        return PixelKalmanFilter(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
    raise ValueError(f"Unsupported target filter: {filter_name}")
