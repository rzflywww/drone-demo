"""Optional target filters for 3D world coordinates."""

import numpy as np


TARGET_FILTER_NAMES = ("none", "kalman")
DEFAULT_WORLD_PROCESS_NOISE = 10.0
DEFAULT_WORLD_MEASUREMENT_NOISE = 0.04


class ConstantAccelerationKalmanFilter:
    """Constant-acceleration Kalman filter for an arbitrary coordinate count."""

    def __init__(
        self,
        dimensions,
        process_noise,
        measurement_noise,
        *,
        initial_position_variance,
        initial_velocity_variance,
        initial_acceleration_variance,
    ):
        self.dimensions = int(dimensions)
        if self.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_position_variance = float(initial_position_variance)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.initial_acceleration_variance = float(initial_acceleration_variance)
        self.initialized = False
        self.last_time = None
        self.state = np.zeros((self.dimensions * 3, 1), dtype=np.float64)
        self.covariance = np.eye(self.dimensions * 3, dtype=np.float64) * 1000.0

    def reset(self):
        self.initialized = False
        self.last_time = None
        self.state.fill(0.0)
        self.covariance = np.eye(self.dimensions * 3, dtype=np.float64) * 1000.0

    def update(self, *measurement_and_timestamp):
        expected = self.dimensions + 1
        if len(measurement_and_timestamp) != expected:
            raise ValueError(
                f"expected {self.dimensions} coordinates and a timestamp"
            )

        *coordinates, timestamp = measurement_and_timestamp
        measurement = np.asarray(coordinates, dtype=np.float64).reshape(
            self.dimensions, 1
        )

        if not self.initialized:
            self.state.fill(0.0)
            self.state[: self.dimensions] = measurement
            diagonal = (
                [self.initial_position_variance] * self.dimensions
                + [self.initial_velocity_variance] * self.dimensions
                + [self.initial_acceleration_variance] * self.dimensions
            )
            self.covariance = np.diag(diagonal).astype(np.float64)
            self.initialized = True
            self.last_time = float(timestamp)
            return

        dt = max(1e-3, min(float(timestamp) - self.last_time, 1.0))
        self.last_time = float(timestamp)
        dt2 = dt * dt
        identity = np.eye(self.dimensions, dtype=np.float64)
        zero = np.zeros_like(identity)
        transition = np.block(
            [
                [identity, dt * identity, 0.5 * dt2 * identity],
                [zero, identity, dt * identity],
                [zero, zero, identity],
            ]
        )

        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        dt5 = dt4 * dt
        dt6 = dt3 * dt3
        temporal_process = np.array(
            [
                [dt6 / 36.0, dt5 / 12.0, dt4 / 6.0],
                [dt5 / 12.0, dt4 / 4.0, dt3 / 2.0],
                [dt4 / 6.0, dt3 / 2.0, dt2],
            ],
            dtype=np.float64,
        )
        process = self.process_noise * np.kron(temporal_process, identity)

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

        observation = np.hstack((identity, zero, zero))
        noise = identity * self.measurement_noise
        residual = measurement - observation @ self.state
        residual_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(residual_cov)

        self.state = self.state + gain @ residual
        state_identity = np.eye(self.dimensions * 3, dtype=np.float64)
        self.covariance = (state_identity - gain @ observation) @ self.covariance

    def predict(self, lead_time):
        if not self.initialized:
            return None

        lead_time = max(0.0, float(lead_time))
        lead_time2 = lead_time * lead_time
        position = self.state[: self.dimensions, 0]
        velocity = self.state[self.dimensions : self.dimensions * 2, 0]
        acceleration = self.state[self.dimensions * 2 :, 0]
        predicted = (
            position
            + velocity * lead_time
            + 0.5 * acceleration * lead_time2
        )
        return tuple(float(value) for value in predicted)


class WorldKalmanFilter(ConstantAccelerationKalmanFilter):
    """Constant-acceleration Kalman filter for 3D world positions in meters."""

    def __init__(
        self,
        process_noise=DEFAULT_WORLD_PROCESS_NOISE,
        measurement_noise=DEFAULT_WORLD_MEASUREMENT_NOISE,
    ):
        super().__init__(
            dimensions=3,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_position_variance=measurement_noise,
            initial_velocity_variance=25.0,
            initial_acceleration_variance=100.0,
        )


def create_world_target_filter(
    filter_name,
    *,
    kalman_process_noise=DEFAULT_WORLD_PROCESS_NOISE,
    kalman_measurement_noise=DEFAULT_WORLD_MEASUREMENT_NOISE,
):
    """Create a world-space target filter; ``none`` is a true bypass."""
    if filter_name == "none":
        return None
    if filter_name == "kalman":
        return WorldKalmanFilter(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
    raise ValueError(f"Unsupported world target filter: {filter_name}")
