import numpy as np
import jax.numpy as jnp
import os
import shutil
import matplotlib.pyplot as plt
from pathlib import Path

from src.boundary_conditions import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD2Q9
from src.utils import *


class FlowOverStep(KBCSim):
    def __init__(self, step_indices, inlet_velocity, outdir=None, **kwargs):
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = "."
        self.outdir = Path(outdir).resolve(True)

        self.step_indices = step_indices
        self.inlet_velocity = inlet_velocity
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        stationary_walls = np.concatenate(
            (
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.step_indices,
            )
        )
        outlet = self.boundingBoxIndices["right"]

        # all x,y indices on the left boundary that are not part of the step are the inlet
        step_left_wall_indices = self.step_indices[self.step_indices[:, 0] == 0][:, 1]
        inlet = np.array(
            [
                x
                for x in self.boundingBoxIndices["left"]
                if x[1] not in step_left_wall_indices
            ]
        )

        rho_wall = np.ones(
            (inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        )
        vel_wall = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = self.inlet_velocity

        self.BCs.append(
            BounceBack(tuple(stationary_walls.T), self.gridInfo, self.precisionPolicy)
        )
        self.BCs.append(DoNothing(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        self.BCs.append(
            EquilibriumBC(
                tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall
            )
        )

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)

        rho = np.array(kwargs["rho"][..., 1:-1, :])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))
        save_image(
            timestep, np.flip(u, axis=1), prefix=str(self.outdir.absolute()) + os.sep
        )


def main():
    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)
    nx = 10_000
    ny = 1000
    Re = 2100.0
    prescribed_vel = 0.1
    clength = nx - 1
    output_path = "parabolic_sim_results"
    visc = prescribed_vel * clength / Re

    omega = 1.0 / (3.0 * visc + 0.5)
    assert omega < 1.98, "omega must be less than 2.0"
    shutil.rmtree(output_path, ignore_errors=True)

    # a step starting from the middle of left wall
    step_indices = np.array(
        [[x, y] for x in range(nx // 8) for y in range(ny // 2, ny)]
    )

    kwargs = {
        "lattice": lattice,
        "omega": omega,
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
    }
    sim = FlowOverStep(step_indices, prescribed_vel, outdir=output_path, **kwargs)
    sim.run(20000)


if __name__ == "__main__":
    main()
