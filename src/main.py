from raw_data import RawData
from imu_localization import IMULocalizer

# seems good: 2011_09_26_drive_0056 (1.2 GB)

# a bit dubious: 2011_09_29_drive_0071_sync

# results are good: 2011_09_26_drive_0036_sync
# 2011_09_26_drive_0009_sync

# wtf: 2011_09_26_drive_0093_sync

PATH: str = "./sim_rectangle_rounded_noisy"

if __name__ == "__main__":
    print("IMU Data Processing with Kalman Filter")
    print("=" * 50)
    
    # Show raw data sample
    raw_data = RawData(PATH)
    print(f"\nLoaded {len(raw_data.data)} data points")
    
    # Run Kalman filter localization
    print("Running Kalman Filter localization...")
    localizer = IMULocalizer(PATH)
    localizer.process_data()
    
    # Show results
    summary = localizer.get_trajectory_summary()
    print("\n" + "=" * 50)
    print("LOCALIZATION RESULTS")
    print("=" * 50)
    print(f"Duration: {summary['duration']:.2f} seconds")
    print(f"Total displacement from start:")
    print(f"  X: {summary['total_displacement'][0]:.2f} meters")
    print(f"  Y: {summary['total_displacement'][1]:.2f} meters") 
    print(f"  Z: {summary['total_displacement'][2]:.2f} meters")
    print(f"Total distance traveled: {summary['max_displacement']:.2f} meters")
    print(f"Average speed: {summary['avg_velocity']:.2f} m/s")
    print(f"Maximum speed: {summary['max_velocity']:.2f} m/s")
    
    # Generate plots and export data
    print("\nGenerating trajectory plot...")
    localizer.plot_trajectory("trajectory_plot.png")
    print("Exporting trajectory data...")
    localizer.export_trajectory("trajectory_data.csv")
    print("Done! Check trajectory_plot.png and trajectory_data.csv for detailed results.")