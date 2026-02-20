"""
galaxy_neighbors.py
-------------------
Modular tools for finding faint galaxy neighbors around bright galaxies
in 21cmFAST halo catalogs, for both fiducial and stochastic models.

Usage
-----
    from galaxy_neighbors import RedshiftConfig, AnalysisConfig, GalaxyModel, run_neighbor_analysis

    Z10 = RedshiftConfig(
        redshift=10.5,
        halo_catalog_path=Path('/lustre/.../10.5000/HaloCatalog.h5'),
        muv_fiducial_path=Path('/lustre/.../catalog_fiducial_bigger_new_save.h5'),
        muv_stochastic_path=Path('/lustre/.../catalog_stoch_bigger_new3.h5'),
    )

    cfg = AnalysisConfig(
        bright_limits=[-22.0, -21.5, -21.0],
        faint_limits=[-17.5, -17.75, -18.0, -18.25, -18.5, -18.75],
        redshift_cfg=Z10,
    )

    results_fid, results_stoc = run_neighbor_analysis(Z10, cfg)
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
# Redshift configuration
# ---------------------------------------------------------------------------

@dataclass
class RedshiftConfig:
    """All redshift-specific file paths and the redshift value itself.

    One instance per redshift snapshot. Collect all four in a registry
    dict in run_analysis.py and select via --redshift on the CLI.

    Parameters
    ----------
    redshift : float
        Redshift of the snapshot (e.g. 8.0, 10.5, 12.0, 14.0).
    halo_catalog_path : Path
        Path to the HaloCatalog.h5 file for this snapshot.
    muv_fiducial_path : Path
        Path to the fiducial MUV catalog HDF5 file.
    muv_stochastic_path : Path
        Path to the stochastic MUV catalog HDF5 file.
    label : str, optional
        Short label used in filenames and plot annotations.
        Defaults to 'z{redshift}'.
    """

    redshift: float
    halo_catalog_path: Path
    muv_fiducial_path: Path
    muv_stochastic_path: Path
    label: str = ""

    def __post_init__(self):
        self.halo_catalog_path   = Path(self.halo_catalog_path)
        self.muv_fiducial_path   = Path(self.muv_fiducial_path)
        self.muv_stochastic_path = Path(self.muv_stochastic_path)
        if not self.label:
            self.label = f"z{self.redshift}"

    @property
    def cache_subdir(self) -> str:
        """Subdirectory name to isolate cache/output for this redshift."""
        return self.label   # e.g. 'z8.0', 'z10.5', 'z12.0', 'z14.0'


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """All tunable analysis parameters in one place.

    Parameters
    ----------
    bright_limits : list of float
        UV magnitude thresholds that define "bright" galaxies.
        A galaxy is considered bright at threshold M if M_UV < M.
    faint_limits : list of float
        UV magnitude thresholds for the faint neighbor sample.
    preselect_faint_limit : float
        Coarse pre-selection cut applied once before the nested loop.
        Should be >= the brightest (least negative) value in faint_limits.
    survey_area_arcmin2 : float
        On-sky survey area in arcmin^2. Defaults to NIRSpec MSA (12.24).
    """

    bright_limits: List[float] = field(
        default_factory=lambda: [-22.0, -21.5, -21.0]
    )
    faint_limits: List[float] = field(
        default_factory=lambda: [-17.5, -17.75, -18.0, -18.25, -18.5, -18.75]
    )
    preselect_faint_limit: float = -17.5
    survey_area_arcmin2: float = 12.24  # NIRSpec MSA

    def __post_init__(self):
        if self.preselect_faint_limit < min(self.faint_limits):
            warnings.warn(
                "preselect_faint_limit is more negative than the faintest "
                "faint_limit -- some galaxies may be excluded unexpectedly."
            )

    def search_box_mpc(self, redshift: float) -> float:
        """Comoving half-side length of the search box in Mpc.

        Takes redshift explicitly so the same AnalysisConfig instance can
        be reused across different RedshiftConfigs without mutation.
        """
        side = np.sqrt(self.survey_area_arcmin2) * u.arcmin
        return (side * cosmo.kpc_comoving_per_arcmin(z=redshift).to(u.Mpc / u.arcmin)).value

    @property
    def bright_names(self) -> List[str]:
        return [_mag_to_key(m) for m in self.bright_limits]

    @property
    def faint_names(self) -> List[str]:
        return [_mag_to_key(m) for m in self.faint_limits]


def _mag_to_key(mag: float) -> str:
    """Convert a magnitude float to a readable dict key, e.g. -21.5 -> 'M21.5'."""
    return f"M{abs(mag):.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NeighborResult:
    """Neighbor galaxies around a single bright galaxy.

    Attributes
    ----------
    bright_mag : float
    bright_coord : np.ndarray, shape (3,)
    faint_mags : np.ndarray
    faint_coords : np.ndarray, shape (N, 3)
    distances : np.ndarray
    bright_limit : float
    faint_limit : float
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
# Catalog loaders
# ---------------------------------------------------------------------------

def load_halo_catalog(halo_catalog_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load halo coordinates and log-masses from an HDF5 halo catalog."""
    path = Path(halo_catalog_path)
    with h5py.File(path, "r") as f:
        coords_raw = np.array(f["HaloCatalog"]["OutputFields"]["halo_coords"])
        masses_raw = np.array(f["HaloCatalog"]["OutputFields"]["halo_masses"])

    mask = masses_raw > 0.0
    return coords_raw[mask], np.log10(masses_raw[mask])


def load_muv_catalog(
    muv_catalog_path: str | Path,
    index: int | list[int] | None = 0,
    n_realizations: int | None = None,
) -> np.ndarray:
    """Load UV magnitudes from a model catalog HDF5 file.

    Exactly one of `index` or `n_realizations` should be provided.

    Parameters
    ----------
    index : int or list of int, optional
        Single realization index or an explicit list of indices to load
        and concatenate. Default: 0.
    n_realizations : int, optional
        Load the first N realizations ([:N]) and concatenate.
        Mutually exclusive with `index`.
    """
    if index is not None and n_realizations is not None:
        raise ValueError("Provide either `index` or `n_realizations`, not both.")

    path = Path(muv_catalog_path)
    with h5py.File(path, "r") as f:
        if n_realizations is not None:
            muvs = np.concatenate(f["data"][:n_realizations], axis=0)
        elif isinstance(index, list):
            muvs = np.concatenate([f["data"][i] for i in index], axis=0)
        else:
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
    """Find faint galaxies within a cuboid search box around a bright galaxy."""
    bx, by, bz = bright_coord
    box_mask = (
        (faint_coords[:, 0] >= bx - half_side) & (faint_coords[:, 0] <= bx + half_side) &
        (faint_coords[:, 1] >= by - half_side) & (faint_coords[:, 1] <= by + half_side) &
        (faint_coords[:, 2] >= bz - half_side) & (faint_coords[:, 2] <= bz + half_side) &
        (faint_mags < faint_limit)
    )

    matched_coords = faint_coords[box_mask]
    matched_mags   = faint_mags[box_mask]

    if len(matched_coords) == 0:
        return matched_mags, matched_coords, np.array([])

    dx = matched_coords - bright_coord
    distances = np.sqrt((dx ** 2).sum(axis=1))
    return matched_mags, matched_coords, distances


# ---------------------------------------------------------------------------
# Galaxy model
# ---------------------------------------------------------------------------

class GalaxyModel:
    """Encapsulates a single galaxy model (fiducial or stochastic).

    Parameters
    ----------
    halo_coords : np.ndarray, shape (N, 3)
    muvs : np.ndarray, shape (N,)
    analysis_cfg : AnalysisConfig
    redshift_cfg : RedshiftConfig
    name : str
    """

    def __init__(
        self,
        halo_coords: np.ndarray,
        muvs: np.ndarray,
        analysis_cfg: AnalysisConfig,
        redshift_cfg: RedshiftConfig,
        name: str = "model",
    ):
        self.analysis_cfg = analysis_cfg
        self.redshift_cfg = redshift_cfg
        self.name = name

        bright_cut = max(analysis_cfg.bright_limits)
        faint_cut = analysis_cfg.preselect_faint_limit

        # When multiple realizations are concatenated, muvs is N*n_real long
        # but halo_coords is only N long — tile coords to match.
        n_halos = len(halo_coords)
        n_muvs = len(muvs)
        if n_muvs != n_halos:
            if n_muvs % n_halos != 0:
                raise ValueError(
                    f"MUV array length ({n_muvs}) is not a multiple of "
                    f"halo catalog length ({n_halos}). Check your catalogs."
                )
            halo_coords = np.tile(halo_coords, (n_muvs // n_halos, 1))

        self.bright_coords = halo_coords[muvs < bright_cut]
        self.bright_mags = muvs[muvs < bright_cut]
        faint_mask = (muvs < faint_cut) & (muvs >= bright_cut)
        self.faint_coords = halo_coords[faint_mask]
        self.faint_mags = muvs[faint_mask]

    @classmethod
    def from_hdf5(
        cls,
        muv_catalog_path: str | Path,
        analysis_cfg: AnalysisConfig,
        redshift_cfg: RedshiftConfig,
        muv_index: int | list[int] = 0,
        n_realizations: int | None = None,
        name: str = "model",
    ) -> "GalaxyModel":
        """Construct a GalaxyModel directly from HDF5 files.

        Halo catalog path is taken from redshift_cfg automatically.

        Parameters
        ----------
        muv_index : int or list of int
            Single realization index or explicit list.
            Ignored when n_realizations is set.
        n_realizations : int, optional
            Load the first N realizations and concatenate.
        """
        halo_coords, _ = load_halo_catalog(redshift_cfg.halo_catalog_path)
        muvs = load_muv_catalog(
            muv_catalog_path,
            index=muv_index if n_realizations is None else None,
            n_realizations=n_realizations,
        )
        return cls(halo_coords, muvs, analysis_cfg, redshift_cfg, name=name)

    def run(self) -> dict[str, dict[str, list[NeighborResult]]]:
        """Run the full neighbor search over all (bright, faint) limit pairs."""
        cfg      = self.analysis_cfg
        z_cfg    = self.redshift_cfg
        half_side = cfg.search_box_mpc(z_cfg.redshift)

        results: dict[str, dict[str, list]] = {
            bname: {fname: [] for fname in cfg.faint_names}
            for bname in cfg.bright_names
        }

        for ibri, bright_coord in enumerate(self.bright_coords):
            bright_mag = self.bright_mags[ibri]

            for bright_limit, bright_key in zip(cfg.bright_limits, cfg.bright_names):
                if bright_mag >= bright_limit:
                    continue

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
            f"GalaxyModel(name='{self.name}', z={self.redshift_cfg.redshift}, "
            f"n_bright={len(self.bright_coords)}, n_faint={len(self.faint_coords)})"
        )


# ---------------------------------------------------------------------------
# Convenience: run both models in one call
# ---------------------------------------------------------------------------

def run_neighbor_analysis(
    redshift_cfg: RedshiftConfig,
    analysis_cfg: Optional[AnalysisConfig] = None,
    muv_index: int | list[int] = 0,
    n_realizations: int | None = None,
) -> tuple[dict, dict]:
    """Run the full neighbor search for both fiducial and stochastic models.

    Parameters
    ----------
    redshift_cfg : RedshiftConfig
        Snapshot-specific paths and redshift value.
    analysis_cfg : AnalysisConfig, optional
        Magnitude grids and survey area. Uses defaults if not provided.
    muv_index : int or list of int
        Single realization index or explicit list.
    n_realizations : int, optional
        Load the first N realizations and concatenate.

    Returns
    -------
    results_fid, results_stoc : dicts
    """
    if analysis_cfg is None:
        analysis_cfg = AnalysisConfig()

    kwargs = dict(
        analysis_cfg=analysis_cfg,
        redshift_cfg=redshift_cfg,
        muv_index=muv_index,
        n_realizations=n_realizations,
    )

    fiducial   = GalaxyModel.from_hdf5(redshift_cfg.muv_fiducial_path,   name="fiducial",   **kwargs)
    stochastic = GalaxyModel.from_hdf5(redshift_cfg.muv_stochastic_path, name="stochastic", **kwargs)

    return fiducial.run(), stochastic.run()
