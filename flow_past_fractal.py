import os
import jax.numpy as jnp
import numpy as np
from src.utils import *
from jax import config
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from src.models import BGKSim, KBCSim
from src.boundary_conditions import *
from src.lattice import LatticeD2Q9

FRACTAL_VALUE = 1
EMPTY_VALUE = 0
script_name = os.path.basename(__file__)
OUTPUT_DIR = Path(Path(f"output_{script_name}").stem)
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

forces_log_h = open(OUTPUT_DIR / "forces_log.txt", "w")
print(
    "timestep\tf_total_x\tf_total_y\tf_total_magnitude", file=forces_log_h, flush=True
)
DPI = 300


def load_fractal(image_path: Path, threshold: int = 127) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image[image < threshold] = FRACTAL_VALUE
    image[image >= threshold] = EMPTY_VALUE
    return image


class FlowPastFractal(KBCSim):
    def __init__(self, inlet_velocity: float, fractal: np.ndarray, **kwargs):
        self.fractal = fractal
        self.inlet_velocity = inlet_velocity
        # set the number of pixels to be such that the fractal is 1/3 of the domain
        kwargs["nx"] = fractal.shape[0] * 5
        kwargs["ny"] = fractal.shape[1] * 3
        kwargs["nz"] = 0
        self.fractal_indices = None
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        stationary_walls = np.concatenate(
            (self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"])
        )
        self.fractal_indices = np.argwhere(self.fractal == FRACTAL_VALUE)
        # offset such that fractal is in the middle of the domain
        self.fractal_indices[:, 0] += 2 * self.fractal.shape[0]
        self.fractal_indices[:, 1] += self.fractal.shape[1]

        # whole left boundary is inlet
        inlet = self.boundingBoxIndices["left"]
        rho_wall = np.ones(
            (inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        )
        vel_wall = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = self.inlet_velocity

        self.BCs.append(
            BounceBackHalfway(
                tuple(self.fractal_indices.T), self.gridInfo, self.precisionPolicy
            )
        )
        self.BCs.append(
            BounceBack(tuple(stationary_walls.T), self.gridInfo, self.precisionPolicy)
        )
        self.BCs.append(
            DoNothing(
                tuple(self.boundingBoxIndices["right"].T),
                self.gridInfo,
                self.precisionPolicy,
            )
        )
        self.BCs.append(
            EquilibriumBC(
                tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall
            )
        )

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)

        u = np.array(kwargs["u"][..., 1:-1, :])
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))

        FRACTAL_BC_IDX = 0
        fractal = self.BCs[FRACTAL_BC_IDX]
        forces_on_fractal = fractal.momentum_exchange_force(
            kwargs["f_poststreaming"], kwargs["f_postcollision"]
        )
        forces_on_fractal = np.sum(np.array(forces_on_fractal), axis=0)
        print(f"forces = {forces_on_fractal}")
        print(
            f"{kwargs['timestep']}\t{forces_on_fractal[0]}\t{forces_on_fractal[1]}\t{np.linalg.norm(forces_on_fractal, 2)}",
            file=forces_log_h,
            flush=True,
        )
        print(f"error = {err:07.6f}")
        domain = np.zeros((self.nx, self.ny))
        domain[self.fractal_indices[:, 0], self.fractal_indices[:, 1]] = FRACTAL_VALUE
        domain = np.flip(domain, axis=1)
        u_flip = np.flip(u, axis=1)
        u_val = np.sqrt(u_flip[..., 0] ** 2 + u_flip[..., 1] ** 2)

        plt.matshow((domain * 255).T, cmap="gray")
        plt.imshow(u_val.T, cmap="viridis", alpha=0.5)
        plt.savefig(
            OUTPUT_DIR / f"frame_{kwargs['timestep']:08d}.png",
            dpi=DPI,
        )
        plt.clf()
        plt.close()


def main():
    img = load_fractal(Path("./assets/fractals/dla_fractal.jpg"))

    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)
    Re = 100.0
    prescribed_vel = 0.1
    clength = img.shape[0] * 3 - 1
    visc = prescribed_vel * clength / Re

    omega = 1.0 / (3.0 * visc + 0.5)
    assert omega < 1.98, "omega must be less than 2.0"

    kwargs = {
        "lattice": lattice,
        "omega": omega,
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "return_fpost": True,
    }

    sim = FlowPastFractal(prescribed_vel, img, **kwargs)
    sim.run(100_000)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        forces_log_h.flush()
        forces_log_h.close()
