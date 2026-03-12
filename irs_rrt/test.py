import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import RigidTransform, Quaternion, RotationMatrix



def cosine_weighted_cone(theta_max):
    """
    Generate a cosine-weighted quaternion
    inside cone around +Z.
    """
    u1 = np.random.rand()
    u2 = np.random.rand()

    sin_theta = np.sqrt(u1) * np.sin(theta_max)
    cos_theta = np.sqrt(1.0 - sin_theta**2)

    phi = 2.0 * np.pi * u2

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta

    z_axis = np.array([0,0,1])

    dir = np.array([x,y,z])
    dot = np.dot(z_axis, dir)
    # Aligned
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = np.cross(z_axis, dir)
    axis /= np.linalg.norm(axis)

    angle = np.arccos(dot)

    half = 0.5 * angle

    w = np.cos(half)
    xyz = axis * np.sin(half)

    return np.array([w, xyz[0], xyz[1], xyz[2]])

def main():
    N = 1000
    theta_max = np.radians(30)

    rots = [Quaternion(cosine_weighted_cone(theta_max)).rotation() for _ in range(N)]
    xyz = np.array([rot @ np.array([0, 0, 1]) for rot in rots])

    x, y, z = xyz.T

    # -------------------------
    # 3D Scatter Plot
    # -------------------------

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121, projection="3d")

    ax.scatter(x, y, z, s=1, alpha=0.3)

    # Draw cone boundary
    t = np.linspace(0, 2*np.pi, 200)

    r = np.sin(theta_max)
    zc = np.cos(theta_max)

    ax.plot(
        r*np.cos(t),
        r*np.sin(t),
        zc*np.ones_like(t),
        linewidth=2
    )

    ax.set_title("Cosine-Weighted Cone Samples")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    ax.view_init(elev=25, azim=45)

    # -------------------------
    # Polar Angle Histogram
    # -------------------------

    theta = np.arccos(z)

    ax2 = fig.add_subplot(122)

    bins = 60
    hist, edges = np.histogram(
        theta,
        bins=bins,
        range=(0, theta_max),
        density=True
    )

    centers = 0.5 * (edges[:-1] + edges[1:])

    ax2.plot(centers, hist, label="Sampled")

    # Theoretical PDF (normalized)
    pdf = (
        np.cos(centers) * np.sin(centers)
        / (0.5 * np.sin(theta_max)**2)
    )

    ax2.plot(centers, pdf, "--", label="Theory")

    ax2.set_xlabel("Polar Angle θ (rad)")
    ax2.set_ylabel("Density")
    ax2.set_title("Angular Distribution")

    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
