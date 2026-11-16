import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular


def lorenz(u, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = u

    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
        ])


def jacobian(u, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = u

    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
        ])


def rk4_step(u, V, dt):

    def derivatives(u, V):
        du = lorenz(u)
        dV = jacobian(u) @ V
        return du, dV

    k1u, k1V = derivatives(u, V)

    k2u, k2V = derivatives(
            u + dt*k1u/2,
            V + dt*k1V/2
            )

    k3u, k3V = derivatives(
            u + dt*k2u/2,
            V + dt*k2V/2
            )

    k4u, k4V = derivatives(
            u + dt*k3u,
            V + dt*k3V
            )

    u_new = u + dt/6 * (
            k1u + 2*k2u + 2*k3u + k4u
            )

    V_new = V + dt/6 * (
            k1V + 2*k2V + 2*k3V + k4V
            )

    return u_new, V_new


def forward_qr(u0, n_transient, n_record,
               n_backward, dt=0.01):

    u = np.array(u0, dtype=float)
    Q = np.eye(3)

    Q_history = []
    R_history = []
    trajectory = []

    lyapunov_sum = np.zeros(3)

    total = n_transient + n_record + n_backward

    for n in range(total):

        # Store the basis at the BEGINNING of the interval
        if n >= n_transient:
            Q_history.append(Q.copy())
            trajectory.append(u.copy())

        # Integrate state and tangent vectors together
        u, V = rk4_step(u, Q, dt)

        # QR decomposition
        Q_new, R = np.linalg.qr(V)

        # Make diagonal entries of R positive
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1

        S = np.diag(signs)

        Q = Q_new @ S
        R = S @ R

        if n >= n_transient:

            R_history.append(R.copy())

            if n < n_transient + n_record:
                lyapunov_sum += np.log(np.diag(R))

    # Store the final basis and state
    Q_history.append(Q.copy())
    trajectory.append(u.copy())

    # Lyapunov exponents per unit physical time
    lyapunov_exponents = (
            lyapunov_sum / (n_record * dt)
            )

    return (
            np.array(Q_history),
            np.array(R_history),
            np.array(trajectory),
            lyapunov_exponents
            )


def backward_clv(Q_history, R_history, n_record):

    n_total = len(R_history)
    dim = Q_history.shape[1]

    rng = np.random.default_rng(42)

    A = np.triu(rng.normal(size=(dim, dim)))

    A /= np.linalg.norm(A, axis=0)

    CLVs = np.zeros((n_record + 1, dim, dim))

    for k in range(n_total - 1, -1, -1):

        if k + 1 == n_record:
            Gamma = Q_history[n_record] @ A
            Gamma /= np.linalg.norm(Gamma, axis=0)
            CLVs[n_record] = Gamma

        A = solve_triangular(
            R_history[k],
            A,
            lower=False
            )

        A /= np.linalg.norm(A, axis=0)

        if k < n_record:
            Gamma = Q_history[k] @ A
            Gamma /= np.linalg.norm(Gamma, axis=0)
            CLVs[k] = Gamma

    return CLVs


def verify_covariance(trajectory, CLVs, dt):

    n_record = len(CLVs) - 1
    errors = np.zeros((n_record, 3))

    for k in range(n_record):

        _, propagated = rk4_step(
                trajectory[k],
                CLVs[k],
                dt
                )

        for j in range(3):

            v = propagated[:, j]
            w = CLVs[k+1, :, j]

            cosine = abs(
                    np.dot(v, w) /
                    (np.linalg.norm(v) * np.linalg.norm(w))
                    )

            errors[k, j] = 1 - np.clip(cosine, 0, 1)

    return errors


if __name__ == "__main__":

    dt = 0.01

    n_transient = 5000
    n_record = 5000
    n_backward = 3000

    u0 = [1.0, 1.0, 1.0]

    Q_history, R_history, trajectory, LE = forward_qr(
            u0,
            n_transient,
            n_record,
            n_backward,
            dt
            )

    CLVs = backward_clv(
            Q_history,
            R_history,
            n_record
            )

    print("Lyapunov exponents:")
    print(LE)

    print("\nSum of exponents:")
    print(np.sum(LE))

    print("\nExpected sum:")
    print(-(10 + 1 + 8/3))

    errors = verify_covariance(
            trajectory[:n_record+1],
            CLVs,
            dt
            )

    print("\nMean covariance errors:")
    print(np.mean(errors, axis=0))

    # Plot Lorenz trajectory
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    xyz = trajectory[:n_record+1]

    ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            linewidth=0.5,
            alpha=0.6
            )

    scale = 4.0

    for k in range(0, n_record, 250):

        point = xyz[k]

        for j, color in enumerate(["red", "green", "blue"]):

            direction = CLVs[k, :, j]

            ax.quiver(
                    *point,
                    *(scale * direction),
                    color=color,
                    length=1,
                    normalize=False
                    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title("Lorenz Attractor with Covariant Lyapunov Vectors")

    plt.tight_layout()
    plt.show()
