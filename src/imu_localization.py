import csv
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from raw_data import RawData
from datetime import datetime

class IMULocalizer:
    raw_data: RawData

    def __init__(self, data_path: str):
        """
        Initialize the IMU localizer.
        
        Args:
            data_path: Path to the IMU data directory
        """
        self.raw_data = RawData(data_path)
        self.trajectory: list[Any] = []
        self.timestamps: list[Any] = []
        
    def process_data(self) -> None:
        """Process all IMU data through the Kalman filter."""

        assert self.raw_data.data, "Raw data is empty"
        assert len(self.raw_data.data) > 1, "Not enough data points to process"

        total_time = (self.raw_data.data[-1].timestamp - self.raw_data.data[0].timestamp).total_seconds()
        dt = total_time / (len(self.raw_data.data) - 1)
        
        print(f"Processing {len(self.raw_data.data)} data points with dt={dt:.4f}s")
        
        P_old_covariance = np.eye(15)
        P_old_covariance[0:3, 0:3] *= 0.5 # on position
        P_old_covariance[3:6, 3:6] *= 0.1 ** 2 # on velocity
        P_old_covariance[6:9, 6:9] *= (1.0 * np.pi / 180) ** 2 # on attitude
        P_old_covariance[9:12, 9:12] *= 0.1 ** 2 # on accel-bias
        P_old_covariance[12:15, 12:15] *= (0.01 * np.pi / 180) ** 2 # on gyro-bias

        # Initialize Kalman filter
        self.kalman_filter_old_covariance = KalmanFilter(dt=dt, first=self.raw_data.data[0], P=P_old_covariance)

        P = np.eye(15)
        P[0:3, 0:3] *= 1e-9 # on position
        P[3:6, 3:6] *= 1e-6 # on velocity
        P[6:9, 6:9] *= 1e-3 # on rpy
        P[9:12, 9:12] *= 2e-4 # on accel-bias
        P[12:15, 12:15] *= 6e-9 # on gyro-bias
        self.kalman_filter = KalmanFilter(dt=dt, first=self.raw_data.data[0], P=P)
        
        # Process each data point
        for i, data in enumerate(self.raw_data.data[1:]):
            # Run prediction step
            self.kalman_filter_old_covariance.step(data)
            self.kalman_filter.step(data)
            
            # Store trajectory for old covariance
            displacement_old_covariance = self.kalman_filter_old_covariance.position()
            actual_displacement = self.kalman_filter_old_covariance.gps_displacement(data)
            position = self.kalman_filter_old_covariance.position()
            velocity = self.kalman_filter_old_covariance.velocity()
            actual_velocity = np.array([data.vn, data.ve, data.vu])
            orientation = self.kalman_filter_old_covariance.orientation()
            displacement_new_covariance = self.kalman_filter.position()

            self.trajectory.append({
                'timestamp': data.timestamp,
                'position': position,
                'displacement_old_covariance': displacement_old_covariance,
                'velocity': velocity,
                'orientation': orientation,
                'original_data': data,
                'actual_velocity': actual_velocity,
                'actual_displacement': actual_displacement,
                'displacement_new_covariance': displacement_new_covariance,
            })

            # Store timestamps
            self.timestamps.append(data.timestamp)
            
            if i % 20 == 0:
                print(f"Processed {i + 1}/{len(self.raw_data.data)} data points")
        
        print("Data processing complete!")
    
    def get_trajectory_summary(self) -> dict:
        """Get summary statistics of the trajectory."""

        assert self.trajectory and len(self.trajectory) > 0, "No trajectory data available. Run process_data() first."
        
        displacements = np.array([t['displacement_old_covariance'] for t in self.trajectory])
        positions = np.array([t['position'] for t in self.trajectory])
        velocities = np.array([t['velocity'] for t in self.trajectory])
        
        return {
            'total_displacement': displacements[-1],
            'max_displacement': np.max(np.linalg.norm(displacements, axis=1)),
            'final_position': positions[-1],
            'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
            'avg_velocity': np.mean(np.linalg.norm(velocities, axis=1)),
            'duration': (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        }
    
    def plot_trajectory(self, save_path: str | None = None) -> None:
        """
        Plot the 3D trajectory and displacement.
        
        Args:
            save_path: Optional path to save the plot
        """
        
        assert self.trajectory and len(self.trajectory) > 0, "No trajectory data available. Run process_data() first."
        
        # Extract data for plotting
        displacements = np.array([t['displacement_old_covariance'] for t in self.trajectory])
        new_displacements = np.array([t['displacement_new_covariance'] for t in self.trajectory])
        actual_displacements = np.array([t['actual_displacement'] for t in self.trajectory])
        actual_velocity = np.array([t['actual_velocity'] for t in self.trajectory])
        times = [(t['timestamp'] - self.timestamps[0]).total_seconds() for t in self.trajectory]
        
        # Create subplots
        fig = plt.figure(figsize=(15, 12))

        # 3D trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(displacements[:, 0], displacements[:, 1], displacements[:, 2], 'b-', linewidth=2)
        ax1.plot(actual_displacements[:, 0], actual_displacements[:, 1], actual_displacements[:, 2], 'g--', linewidth=1, label='Actual Displacement')
        ax1.scatter(0, 0, 0, color='green', s=100, label='Start') # type: ignore
        ax1.scatter(displacements[-1, 0], displacements[-1, 1], displacements[-1, 2], color='red', s=100, label='End') # type: ignore
        ax1.scatter(actual_displacements[-1, 0], actual_displacements[-1, 1], actual_displacements[-1, 2], color='orange', s=100, label='Actual End') # type: ignore
        ax1.set_xlabel('X Displacement (m)')
        ax1.set_ylabel('Y Displacement (m)')
        ax1.set_zlabel('Z Displacement (m)') # type: ignore
        ax1.set_title('3D Trajectory')
        ax1.grid(True)
        ax1.legend()
        ax1.view_init(elev=20, azim=-20) # type: ignore
        ax1.axis('equal')
        
        # 2D trajectory (top view)
        ax2 = fig.add_subplot(222)
        ax2.plot(displacements[:, 0], displacements[:, 1], 'b-', linewidth=2, label='Old Covariance Displacement')
        ax2.plot(new_displacements[:, 0], new_displacements[:, 1], 'r-', linewidth=2, label='New Covariance Displacement')
        ax2.plot(actual_displacements[:, 0], actual_displacements[:, 1], 'g--', linewidth=1, label='Actual Displacement')
        ax2.scatter(0, 0, color='green', s=100, label='Start')
        ax2.scatter(displacements[-1, 0], displacements[-1, 1], color='red', s=100, label='End')
        ax2.scatter(actual_displacements[-1, 0], actual_displacements[-1, 1], color='orange', s=100, label='Actual End')
        ax2.set_xlabel('X Displacement (m)')
        ax2.set_ylabel('Y Displacement (m)')
        ax2.set_title('2D Trajectory (Top View)')
        ax2.grid(True)
        ax2.legend()
        ax2.axis('equal')
        
        # Actual displacement components over time
        ax3 = fig.add_subplot(223)
        ax3.plot(times, displacements[:, 0], label='X', linewidth=2)
        ax3.plot(times, displacements[:, 1], label='Y', linewidth=2)
        ax3.plot(times, displacements[:, 2], label='Z', linewidth=2)

        ax3.plot(times, actual_displacements[:, 0], '--', label='Actual X', linewidth=1)
        ax3.plot(times, actual_displacements[:, 1], '--', label='Actual Y', linewidth=1)
        ax3.plot(times, actual_displacements[:, 2], '--', label='Actual Z', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Displacement (m)')
        ax3.set_title('Displacement Components vs Time')
        ax3.grid(True)
        ax3.legend()
        
        # Velocity magnitude over time
        ax4 = fig.add_subplot(224)
        velocities = np.array([t['velocity'] for t in self.trajectory])
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        actual_velocity_magnitude = np.linalg.norm(actual_velocity, axis=1)
        ax4.plot(times, velocity_magnitude, 'r-', linewidth=2, label='Estimated Velocity Magnitude')
        ax4.plot(times, actual_velocity_magnitude, '--', linewidth=1, label='Actual Velocity Magnitude')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity Magnitude (m/s)')
        ax4.set_title('Velocity Magnitude vs Time')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_trajectory(self, filename: str) -> None:
        """
        Export trajectory data to CSV file.
        
        Args:
            filename: Output CSV filename
        """

        assert self.trajectory and len(self.trajectory) > 0, "No trajectory data available. Run process_data() first."
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'time_s', 'x_disp', 'y_disp', 'z_disp', 
                         'x_pos', 'y_pos', 'z_pos', 'vx', 'vy', 'vz', 
                         'roll', 'pitch', 'yaw']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, traj in enumerate(self.trajectory):
                time_s = (traj['timestamp'] - self.timestamps[0]).total_seconds()
                writer.writerow({
                    'timestamp': traj['timestamp'].isoformat(),
                    'time_s': time_s,
                    'x_disp': traj['displacement_old_covariance'][0],
                    'y_disp': traj['displacement_old_covariance'][1],
                    'z_disp': traj['displacement_old_covariance'][2],
                    'x_pos': traj['position'][0],
                    'y_pos': traj['position'][1],
                    'z_pos': traj['position'][2],
                    'vx': traj['velocity'][0],
                    'vy': traj['velocity'][1],
                    'vz': traj['velocity'][2],
                    'roll': traj['orientation'][0],
                    'pitch': traj['orientation'][1],
                    'yaw': traj['orientation'][2]
                })
        
        print(f"Trajectory exported to {filename}")