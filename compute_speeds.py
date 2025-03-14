"""
Transform the data array from a series of time snapshots to omega space.
"""

import numpy as np
import h5py
import torch

from abc import ABCMeta, abstractmethod

from tqdm import tqdm

from src.core.read_dns import ReadDNS

from src.core.vector_field import VelocityField
from src.core.computational_grid import GridPhysicalDomain, GridFourierDomain
import timeit
from src.core.derivative_operator import (
    FirstDerivativeXFourier,
    FirstDerivativeZFourier,
    SecondDerivativeXFourier,
    SecondDerivativeZFourier,
    SecondDerivativeYFiniteDiffNeumann,
)


# Start by just doing it for a z plan


class ComputeSpeeds(ReadDNS):
    def __init__(self, FileNameWithPath):
        super().__init__()
        self.gridFourierDomain = None
        self.gridPhysicalDomain = None
        self.velocityFieldFourier = None
        self.vorticityFieldFourier = None
        self.FileNameWithPath = FileNameWithPath
        self._getDataFromHdfFile()

    def _getDataFromHdfFile(self):
        with h5py.File(self.FileNameWithPath, "r") as f:
            self.gridFourierDomain = GridFourierDomain.fromHdfFile(f)
            self.gridPhysicalDomain = GridPhysicalDomain.fromHdfFile(f)
            self.firstDerivativeXFourier = FirstDerivativeXFourier.fromHdfFile(f)
            self.firstDerivativeZFourier = FirstDerivativeZFourier.fromHdfFile(f)
            self.secondDerivativeXFourier = SecondDerivativeXFourier.fromHdfFile(f)
            self.secondDerivativeZFourier = SecondDerivativeZFourier.fromHdfFile(f)
            self.secondDerivativeYFiniteDiff = (
                SecondDerivativeYFiniteDiffNeumann.fromHdfFile(f)
            )

    def choose_c(self, c_choice, equality="=="):
        omegas = np.fft.fftfreq(self.nt, d=self.dt)
        kx_inv = np.zeros_like(self.kx)
        kx_inv[1:] = 1 / self.kx[1:]  # Skip k=0 to avoid /0 error
        kx_inv[0] = 0  

        c_mat = omegas[:, None] @ kx_inv[None, :]
        c_mat = c_mat / self.u_tau # Change to wall units
        # Set reference wavenumber as c_mat[:, 2]. This is kx = 1 for my case
        c_ammend = omegas[(np.abs(c_mat[:, 2] - c_choice)).argmin()] / self.u_tau
        print(f"Chosen c: {c_ammend}")

        if equality == "==":
            c_inds = c_mat == c_ammend
        elif equality == ">":
            c_inds = c_mat > c_ammend
        elif equality == "<":
            c_inds = c_mat < c_ammend
        return np.argwhere(~c_inds)

    def _get_and_save_xy_plane(self):
        # First read and compute FFT
        with h5py.File("data/u_xy_plane.h5", "w") as fplane:
            fplane.create_dataset(
                "u_xy_plane",
                (self.nt, self.ny // 2, self.gridFourierDomain.getStreamwiseGridSize()),
                dtype="complex64",
            )
            with h5py.File(self.FileNameWithPath, "r") as f:
                groupId = f["velocityFieldsFourier"]
                for idt in tqdm(range(self.nt)):
                    u = torch.tensor(
                        groupId["uRealPart"][idt, : self.ny // 2, :, :]
                        + 1.0j * groupId["uImaginaryPart"][idt, : self.ny // 2, :, :],
                        device="cuda",
                        dtype=torch.complex64,
                    )
                    u_plane = torch.fft.ifftn(u, dim=(1))[
                        :, self.nkz // 2, :
                    ]  # * self.nkx # If I need to run again
                    fplane["u_xy_plane"][idt] = u_plane.cpu().numpy()

    def _get_save_c10(self):
        with h5py.File("data/u_xy_plane.h5", "r") as f:
            u = f["u_xy_plane"][:]
            # u[:, :, 0] = 0
            # u -= np.mean(u, axis=0)
        u_hat = np.fft.fft(u, axis=0)
        y_plus = (1 + self.y)[: self.ny // 2] * self.retau
        c_choice = self.u_000[np.argmin(np.abs(y_plus - 10))] / self.u_tau
        print(c_choice)
        indices = self.choose_c(-c_choice, equality="==")

        # # Set values to zero directly instead of using masked array
        for omega_idx, kx_idx in indices:
            u_hat[omega_idx, :, kx_idx] = 0.0

        u = np.fft.ifft(u_hat, axis=0)
        with h5py.File("data/u_xy_10.h5", "w") as fshear:
            fshear.create_dataset("u_xy_plane", data=u)


if __name__ == "__main__":
    import os

    fileName = "data/uvwp.h5"
    fileName = "/mnt/095e6b80-7091-4ee2-8f4c-975e0f56f54e/channel_data/uvwp.h5"
    plane = ComputeSpeeds(fileName)
    # plane._get_and_save_z_plane()
    plane._get_save_c10()
    os.system("python src/plotting/pressure_flucs/u_speeds.py")

    # y_plus = shear.gridPhysicalDomain.y*shear.reynoldsNumber
    # print(shear.u_000.max())
    # # shear._get_and_save_dudy_idt()
    # shear._get_save_c20()
    # os.system("python src/plotting/pressure_flucs/shear_plane.py")
