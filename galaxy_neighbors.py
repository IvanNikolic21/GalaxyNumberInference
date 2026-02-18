"""
galaxy_neighbors.py
-------------------
Modular tools for finding faint galaxy neighbors around bright galaxies
in 21cmFAST halo catalogs, for both fiducial and stochastic models.

Usage
-----
    from galaxy_neighbors import AnalysisConfig, GalaxyModel, run_neighbor_analysis

    cfg = AnalysisConfig(
        bright_limits=[-22.0, -21.5, -21.0],
        faint_limits=[-17.5, -17.75, -18.0, -18.25, -18.5, -18.75],
        redshift=10.145,
    )

    fiducial = GalaxyModel.from_hdf5("catalog_fiducial.h5", cfg)
    stochastic = GalaxyModel.from_hdf5("catalog_stoch.h5", cfg)

    results_fid = fiducial.run()
    results_stoc = stochastic.run()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """All tunable parameters in one place.

    Parameters
    ----------
    bright_limits : list of float
        UV magnitude thresholds that define "bright" galaxies.
        A galaxy is considered bright at threshold M if M_UV < M.
        Defaults to [-22.0, -21.5, -21.0].
    faint_limits : list of float
        UV magnitude thresholds for the faint neighbor sample.
        Defaults to [-17.5, -17.75, -18.0, -18.25, -18.5, -18.75].
    preselect_faint_limit : float
        Coarse pre-selection cut applied once before the nested loop.
        Should be >= the brightest (least negative) value in faint_limits.
    redshift : float
        Redshift of the simulation snapshot, used to compute comoving
        angular diameter distances for the search aperture.
    survey_area_arcmin2 : float
        On-sky survey area in arcmin² used to define the search box
        side length (Lx, Ly).  Defaults to NIRSpec MSA (12.24 arcmin²).
    """

    bright_limits: List[float] = field(
        default_factory=lambda: [-22.0, -21.5, -21.0]
    )
    faint_limits: List[float] = field(
        default_factory=lambda: [-17.5, -17.75, -18.0, -18.25, -18.5, -18.75]
    )
    preselect_faint_limit: float = -17.5
    redshift: float = 10.145
    survey_area_arcmin2: float = 12.24  # NIRSpec MSA

    def __post_init__(self):
        # Validate that preselect cut is not stricter than the faintest limit
        if self.preselect_faint_limit < min(self.faint_limits):
            warnings.warn(
                "preselect_faint_limit is more negative than the faintest "
                "faint_limit — some galaxies may be excluded unexpectedly."
            )

    @property
    def search_box_mpc(self) -> float:
        """Comoving half-side length of the search box in Mpc."""
        side = np.sqrt(self.survey_area_arcmin2) * u.arcmin
        return (side * cosmo.kpc_comoving_per_arcmin(z=self.redshift).to(u.Mpc / u.arcmin)).value

    @property
    def bright_names(self) -> List[str]:
        return [_mag_to_key(m) for m in self.bright_limits]

    @property
    def faint_names(self) -> List[str]:
        return [_mag_to_key(m) for m in self.faint_limits]


def _mag_to_key(mag: float) -> str:
    """Convert a magnitude float to a readable dict key.

    Examples
    --------
    -21.5  ->  'M21.5'
    -18.0  ->  'M18.0'
    """
    return f"M{abs(mag):.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NeighborResult:
    """Neighbor galaxies around a single bright galaxy, at one (bright, faint) limit pair.

    Attributes
    ----------
    bright_mag : float
        UV magnitude of the central bright galaxy.
    bright_coord : np.ndarray, shape (3,)
        (x, y, z) comoving coordinates of the bright galaxy [Mpc].
    faint_mags : np.ndarray
        UV magnitudes of the faint neighbors.
    faint_coords : np.ndarray, shape (N, 3)
        Comoving (x, y, z) coordinates of the faint neighbors [Mpc].
    distances : np.ndarray
        3-D Euclidean distances from bright galaxy to each neighbor [Mpc].
    bright_limit : float
        The bright magnitude threshold applied.
    faint_limit : float
        The faint magnitude threshold applied.
    """

    bright_mag: float
    bright_coord: np.ndarray
    faint_mags: np.ndarray
    faint_coords: np.ndarray
    distances: np.ndarray
    bright_limit: float
    faint_limit: float

    @property
    def n_neighbors(self) -> int:
        return len(self.faint_mags)

    def __repr__(self):
        return (
            f"NeighborResult(bright_limit={self.bright_limit}, "
            f"faint_limit={self.faint_limit}, "
            f"bright_mag={self.bright_mag:.2f}, "
            f"n_neighbors={self.n_neighbors})"
        )


# ---------------------------------------------------------------------------
# Halo catalog loader
# ---------------------------------------------------------------------------

def load_halo_catalog(halo_catalog_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load halo masses and coordinates from an HDF5 halo catalog.

    Parameters
    ----------
    halo_catalog_path : str or Path

    Returns
    -------
    coords : np.ndarray, shape (N, 3)
        Comoving coordinates [Mpc] for halos with mass > 0.
    log_masses : np.ndarray, shape (N,)
        log10 of the halo masses for halos with mass > 0.
    """
    path = Path(halo_catalog_path)
    with h5py.File(path, "r") as f:
        coords_raw = np.array(f["HaloCatalog"]["OutputFields"]["halo_coords"])
        masses_raw = np.array(f["HaloCatalog"]["OutputFields"]["halo_masses"])

    mask = masses_raw > 0.0
    coords = coords_raw[mask]
    log_masses = np.log10(masses_raw[mask])
    return coords, log_masses


def load_muv_catalog(muv_catalog_path: str | Path, index: int = 0) -> np.ndarray:
    """Load UV magnitudes from a model catalog HDF5 file.

    Parameters
    ----------
    muv_catalog_path : str or Path
    index : int
        Which realization to load (row index in the 'data' dataset).

    Returns
    -------
    muvs : np.ndarray, shape (N,)
    """
    path = Path(muv_catalog_path)
    with h5py.File(path, "r") as f:
        muvs = np.array(f["data"][index])
    return muvs


# ---------------------------------------------------------------------------
# Core neighbor-finding logic
# ---------------------------------------------------------------------------

def find_neighbors_in_box(
    bright_coord: np.ndarray,
    faint_coords: np.ndarray,
    faint_mags: np.ndarray,
    half_side: float,
    faint_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find faint galaxies within a cuboid search box around a bright galaxy.

    Parameters
    ----------
    bright_coord : array-like, shape (3,)
        (x, y, z) of the bright galaxy.
    faint_coords : np.ndarray, shape (M, 3)
        Coordinates of the faint galaxy pool.
    faint_mags : np.ndarray, shape (M,)
        UV magnitudes of the faint galaxy pool.
    half_side : float
        Half-side length of the search box (Lx = Ly = Lz = half_side).
    faint_limit : float
        Magnitude cut: keep only galaxies with M_UV < faint_limit.

    Returns
    -------
    matched_mags : np.ndarray
    matched_coords : np.ndarray, shape (N, 3)
    distances : np.ndarray, shape (N,)
    """
    bx, by, bz = bright_coord
    box_mask = (
        (faint_coords[:, 0] >= bx - half_side) & (faint_coords[:, 0] <= bx + half_side) &
        (faint_coords[:, 1] >= by - half_side) & (faint_coords[:, 1] <= by + half_side) &
        (faint_coords[:, 2] >= bz - half_side) & (faint_coords[:, 2] <= bz + half_side) &
        (faint_mags < faint_limit)
    )

    matched_coords = faint_coords[box_mask]
    matched_mags = faint_mags[box_mask]

    if len(matched_coords) == 0:
        return matched_mags, matched_coords, np.array([])

    dx = matched_coords - bright_coord  # shape (N, 3)
    distances = np.sqrt((dx ** 2).sum(axis=1))

    return matched_mags, matched_coords, distances


# ---------------------------------------------------------------------------
# Galaxy model: wraps catalog + config + analysis
# ---------------------------------------------------------------------------

class GalaxyModel:
    """Encapsulates a single galaxy model (fiducial or stochastic).

    Parameters
    ----------
    halo_coords : np.ndarray, shape (N, 3)
        Comoving coordinates of all halos [Mpc].
    muvs : np.ndarray, shape (N,)
        UV magnitudes for all halos.
    config : AnalysisConfig
    name : str, optional
        Human-readable label (e.g. 'fiducial', 'stochastic').
    """

    def __init__(
        self,
        halo_coords: np.ndarray,
        muvs: np.ndarray,
        config: AnalysisConfig,
        name: str = "model",
    ):
        self.config = config
        self.name = name

        # Pre-select pools once to avoid repeated masking in the inner loop
        bright_cut = min(config.bright_limits)  # most negative = brightest threshold
        faint_cut = config.preselect_faint_limit

        bright_mask = muvs < bright_cut
        faint_mask = muvs < faint_cut

        self.bright_coords = halo_coords[bright_mask]
        self.bright_mags = muvs[bright_mask]
        self.faint_coords = halo_coords[faint_mask]
        self.faint_mags = muvs[faint_mask]

    @classmethod
    def from_hdf5(
        cls,
        halo_catalog_path: str | Path,
        muv_catalog_path: str | Path,
        config: AnalysisConfig,
        muv_index: int = 0,
        name: str = "model",
    ) -> "GalaxyModel":
        """Construct a GalaxyModel directly from HDF5 files."""
        halo_coords, _ = load_halo_catalog(halo_catalog_path)
        muvs = load_muv_catalog(muv_catalog_path, index=muv_index)
        return cls(halo_coords, muvs, config, name=name)

    def run(self) -> dict[str, dict[str, list[NeighborResult]]]:
        """Run the full neighbor search over all bright/faint limit combinations.

        Returns
        -------
        results : dict
            Nested dict: results[bright_key][faint_key] = list of NeighborResult
            One NeighborResult per bright galaxy that passes the bright_limit cut.
        """
        cfg = self.config
        half_side = cfg.search_box_mpc

        # Initialise result container
        results: dict[str, dict[str, list]] = {
            bname: {fname: [] for fname in cfg.faint_names}
            for bname in cfg.bright_names
        }

        for ibri, bright_coord in enumerate(self.bright_coords):
            bright_mag = self.bright_mags[ibri]

            for bright_limit, bright_key in zip(cfg.bright_limits, cfg.bright_names):
                if bright_mag >= bright_limit:
                    continue  # galaxy not bright enough for this threshold

                for faint_limit, faint_key in zip(cfg.faint_limits, cfg.faint_names):
                    matched_mags, matched_coords, distances = find_neighbors_in_box(
                        bright_coord=bright_coord,
                        faint_coords=self.faint_coords,
                        faint_mags=self.faint_mags,
                        half_side=half_side,
                        faint_limit=faint_limit,
                    )
                    results[bright_key][faint_key].append(
                        NeighborResult(
                            bright_mag=bright_mag,
                            bright_coord=bright_coord.copy(),
                            faint_mags=matched_mags,
                            faint_coords=matched_coords,
                            distances=distances,
                            bright_limit=bright_limit,
                            faint_limit=faint_limit,
                        )
                    )

        return results

    def __repr__(self):
        return (
            f"GalaxyModel(name='{self.name}', "
            f"n_bright={len(self.bright_coords)}, "
            f"n_faint={len(self.faint_coords)})"
        )


# ---------------------------------------------------------------------------
# Convenience: run both models in one call
# ---------------------------------------------------------------------------

def run_neighbor_analysis(
    fiducial_halo_path: str | Path,
    fiducial_muv_path: str | Path,
    stochastic_halo_path: str | Path,
    stochastic_muv_path: str | Path,
    config: Optional[AnalysisConfig] = None,
    muv_index: int = 0,
) -> tuple[dict, dict]:
    """Run the full analysis for both fiducial and stochastic models.

    Parameters
    ----------
    fiducial_halo_path, fiducial_muv_path : paths
        HDF5 files for the fiducial model.
    stochastic_halo_path, stochastic_muv_path : paths
        HDF5 files for the stochastic model.
    config : AnalysisConfig, optional
        If not provided, default config is used.
    muv_index : int
        Which realization index to load from the MUV catalogs.

    Returns
    -------
    results_fid, results_stoc : dicts
        Nested dicts of NeighborResult lists (see GalaxyModel.run).
    """
    if config is None:
        config = AnalysisConfig()

    fiducial = GalaxyModel.from_hdf5(
        fiducial_halo_path, fiducial_muv_path, config, muv_index=muv_index, name="fiducial"
    )
    stochastic = GalaxyModel.from_hdf5(
        stochastic_halo_path, stochastic_muv_path, config, muv_index=muv_index, name="stochastic"
    )

    results_fid = fiducial.run()
    results_stoc = stochastic.run()
    return results_fid, results_stoc
