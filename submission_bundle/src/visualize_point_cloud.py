import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    # Paths
    point_cloud_path = Path("output/point_cloud_scaled_intrinsics_assumption_000000.npy")
    output_image_path = Path("output/point_cloud_000000_3d.png")

    # Load point cloud
    try:
        points = np.load(point_cloud_path)
    except FileNotFoundError:
        print(f"Error: File not found: {point_cloud_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        sys.exit(1)

    # Validate shape
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"Error: Expected point cloud of shape (N, 3), got shape {points.shape}")
        sys.exit(1)

    print(f"Loaded array shape: {points.shape}")
    print(f"Number of points: {points.shape[0]}")

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    s=1, alpha=0.5, c='b', marker='.')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud (Scaled Intrinsics - Unverified Same-FOV Assumption)')

    # Equal aspect ratio: try to set equal scaling
    # Get the range of each axis
    max_range = np.array([points[:,0].max()-points[:,0].min(),
                          points[:,1].max()-points[:,1].min(),
                          points[:,2].max()-points[:,2].min()]).max() / 2.0

    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Save figure
    plt.savefig(output_image_path, dpi=150)
    print(f"Saved visualization to: {output_image_path}")

    # Optional: show plot (commented out to avoid blocking)
    # plt.show()

if __name__ == "__main__":
    main()