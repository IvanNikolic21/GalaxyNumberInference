import argparse
import os
from pathlib import Path
import numpy as np
import h5py
import tables

import py21cmfast as p21c

from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
import logging
import sys


class Muv_Mh_data:
    def compute_Muv_Mh_dict(self, Muv_Mh_filename):
        Muv_Mh = np.genfromtxt(Muv_Mh_filename, dtype=None, names=True)

        # Make a dictionary with Muv_Mh['logMh'] as keys and Muv_Mh['Muv'], as values
        Muv_Mh_dict = {
            Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))
        }
        return Muv_Mh_dict

    def __init__(self, Muv_Mh_file):
        self.Muv_Mh_file = Muv_Mh_file
        self.Muv_Mh_dict = self.compute_Muv_Mh_dict(self.Muv_Mh_file)

def Muv_from_logMh(logMh, Muv_Mh_dict, use_scatter=True, sigmaUV=0.3, dust=False):
    if use_scatter:
        scatter = np.random.normal(0, sigmaUV)
    else:
        scatter = 0
    if dust:
        return Muv_Mh_dict[logMh][1] + scatter
    return Muv_Mh_dict[logMh][0] + scatter

def pMuv_Mh(Muv, logMh, z, sigmaUV_a=-0.34, sigmaUV_b=0.42, Muv_add=0, return_med_sigma=False, Muv_Mh_dict=None):
    """
    sigmaUV (Mh) = a (log Mh - 12) + b.
    a = −0.34 and b = 0.42,
    """
    sigmaUV = sigmaUV_a * (logMh - 12) + sigmaUV_b

    # Get the Muv-Mh relation for the given redshift
    Muv_Mh_med = np.array(
        [Muv_from_logMh(np.round(np.float64(lM), 1), Muv_Mh_dict, use_scatter=False, dust=True) for lM in
         logMh]) + Muv_add

    if return_med_sigma:
        return Muv_Mh_med, sigmaUV

    else:
        # Calculate the probability density function (PDF) of Muv given logMh
        pdf = (1 / (np.sqrt(2 * np.pi) * sigmaUV[:, None])) * np.exp(
            -0.5 * ((Muv - Muv_Mh_med[:, None]) / sigmaUV[:, None]) ** 2)

        return pdf


def Muv_from_logMh_proper(
        logMh,
        Muv_Mh_dict,
        sigmaUV_a=-0.34,
        sigmaUV_b=0.42,
        Muv_add=0
):
    med, sig = pMuv_Mh(
        None,
        logMh,
        z=10.5,
        sigmaUV_b=sigmaUV_b,
        sigmaUV_a=sigmaUV_a,
        Muv_add=Muv_add,
        return_med_sigma=True,
        Muv_Mh_dict=Muv_Mh_dict
    )
    scatter = np.random.normal(0, sig, len(sig))
    return med + scatter


def save_iterative_h5(
        filename,
        n_iter,
        logmhs,
        Muv_Mh_dict,
        sigmaUV_a=-0.34,
        sigmaUV_b=0.42,
        Muv_add=0
):
    """
    Iteratively compute arrays and append them to an HDF5 file
    without storing them in RAM.
    """

    # Open file
    f = tables.open_file(filename, mode='w')

    # Get shape of a single sample
    test_sample = Muv_from_logMh_proper(logmhs, Muv_Mh_dict)
    sample_shape = test_sample.shape  # e.g. (N,)

    # Create an extendable array (EArray)
    atom = tables.Float64Atom()
    array_c = f.create_earray(
        f.root,
        'data',
        atom,
        shape=(0,) + sample_shape,  # First dimension is unlimited
        expectedrows=n_iter
    )

    # Append data iteratively
    for i in range(n_iter):
        Muv_samples = Muv_from_logMh_proper(
            logmhs, Muv_Mh_dict,
            sigmaUV_b=sigmaUV_b,
            Muv_add=Muv_add,
            sigmaUV_a=sigmaUV_a,
        )
        array_c.append(Muv_samples[np.newaxis, :])  # add new "row"

    f.close()

def _load_catalog(
        fname,
        num_rows = None,
        return_mass_and_position = True,
        fname_21cmfast = '/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421/1952/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5'
):
    with h5py.File(fname, 'r') as f:
        if num_rows is not None:
            catalog_loaded = np.array(f['data'])[:num_rows]
        else:
            catalog_loaded = np.array(f['data'])

    if return_mass_and_position:
        halo_coords_proper, logmhs = _load_21cmfast_catalog(fname = fname_21cmfast)
        return catalog_loaded, halo_coords_proper, logmhs
    return catalog_loaded

def _load_21cmfast_catalog(
        fname = '/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421/1952/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5',
):
    with h5py.File(fname, 'r') as f:

        halo_coords = np.array(f['HaloCatalog']['OutputFields']['halo_coords'])
        halo_masses = np.array(f['HaloCatalog']['OutputFields']['halo_masses'])
    logmhs = np.log10(halo_masses[halo_masses > 0.0])
    halo_coords_proper = halo_coords[halo_masses > 0.0]
    return halo_coords_proper, logmhs

def _coeval_to_catalog(
    fname = '/lustre/astro/ivannik/21cmFAST_cache/d12b21e80b7885d62d31717c2c2d8421/1952/ffa852ccaa39d8f82951cc98ff798ab4/10.5000/HaloCatalog.h5',
    niter = 200,
    return_catalog = False,
    Muv_Mh_file = '/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt',
    catalog_name = '/lustre/astro/ivannik/catalog_fiducial_bigger_new.h5',
    sigmaUV_a=-0.34,
    sigmaUV_b=0.42,
    Muv_add=0
):
    """

    :param fname: Name of the native 21cmfast Halo catalog.
    :param niter: Number of iterations of randomly selected galaxies to perform.
    :param return_catalog: If True, the function also returns the catalog. Otherwise, None is returned
    :param Muv_Mh_file: Filename of the
    :param catalog_name:
    :param sigmaUV_a:
    :param sigmaUV_b:
    :param Muv_add:
    :return:
    """
    halo_coords_proper, logmhs = _load_21cmfast_catalog(fname = fname)

    Muv_Mh = np.genfromtxt(Muv_Mh_file, dtype=None, names=True)

    # Make a dictionary with Muv_Mh['logMh'] as keys and Muv_Mh['Muv'], as values
    Muv_Mh_dict = {Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))}

    save_iterative_h5(
        catalog_name,
        niter,
        logmhs,
        Muv_Mh_dict,
        sigmaUV_a=sigmaUV_a,
        sigmaUV_b=sigmaUV_b,
        Muv_add=Muv_add
    )

    if return_catalog:
        catalog_loaded =  _load_catalog(
            fname = catalog_name,
            return_mass_and_position = False,
        )
        return catalog_loaded, halo_coords_proper, logmhs
    else:
        return None


def get_galaxies(Muv, iters, catalog, halo_coords, logmhs):
    halo_mass_all = []
    positions_all = []
    muvs_all = []

    for i in range(iters):
        halo_mass_all.append(logmhs[catalog[i] < Muv])
        positions_all.append(
            halo_coords[catalog[i] < Muv]
        )
        muvs_all.append(
            catalog[i][catalog[i] < -21.5]
        )

    return halo_mass_all, positions_all, muvs_all

class JWST_pointings():
    """
    This class will carry all of the information about the JWST pointings. The general idea is that it takes redshift,
    area and delta_z as inputs and stores them, compute astropy versions of it, or whatever is necessary to facilitate
    computations and comparisons afterwards. I'll start with the simplest case: n_Lx, and delta_z that mark a setup.
    """

    def __NirSpec_defaults__(self):
        self.NIRSpec_area = 12.24 * u.arcmin**2
        self.NIRSpec_Lx = np.sqrt(self.NIRSpec_area) * cosmo.kpc_comoving_per_arcmin(z=self.z).to(u.Mpc/u.arcmin)
        self.Lz_plus = cosmo.comoving_distance(self.z+self.delta_z/2) - cosmo.comoving_distance(self.z)
        self.Lz_minus = cosmo.comoving_distance(self.z) - cosmo.comoving_distance(self.z - self.delta_z / 2)

    def __init__(self, z, n_sky, delta_z):
        self.z = z
        self.n_sky = n_sky
        self.delta_z = delta_z
        self.__NirSpec_defaults__()



def setup_JWST(
        z = 10.15,
        n_sky = 1,
        delta_z = 0.2
):
    """Setter function for JWST pointings"""
    return JWST_pointings(z, n_sky, delta_z)

def get_cutouts(
        catalog_bright,
        catalog_faint,
        bright_positions,
        faint_positions,
        logmhs_faint,
        jwst_pointing=None,
):
    """Based on bright positions get cutouts with faint ones and calculate quantities."""
    positions_cutouts = []
    coords_cutouts_x = []
    coords_cutouts_y = []
    coords_cutouts_z = []
    mags_cutouts = []
    logmhs_cutouts = []

    if jwst_pointing is None:
        jwst_pointing = setup_JWST()
    l_sky = jwst_pointing.NIRSpec_Lx * jwst_pointing.n_sky

    n_iter = len(catalog_bright)
    for i in range(n_iter):
        for i_bright, bright_position in enumerate(bright_positions[i]):
            mask = (abs(
                faint_positions[i][:,0] - bright_position[0]
            ) < l_sky) & (abs(
                faint_positions[i][:, 1] - bright_position[1]
            ) < l_sky ) & (
                faint_positions[i][:, 2] - bright_position[2] <= jwst_pointing.Lz_plus
            ) & (
                faint_positions[i][:, 2] - bright_position[2] >= jwst_pointing.Lz_minus
            )
            positions_cutouts.append(
                np.sqrt(np.sum(faint_i**2 - bright_position**2)) for faint_i in faint_positions[i][mask]
            )
            mags_cutouts.append(catalog_faint[i][mask])
            logmhs_cutouts.append(logmhs_faint[i][mask])
            coords_cutouts_x.append(faint_positions[i][mask][:,0])
            coords_cutouts_y.append(faint_positions[i][mask][:,1])
            coords_cutouts_z.append(faint_positions[i][mask][:,2])
    return positions_cutouts, mags_cutouts, logmhs_cutouts, coords_cutouts_x,

def radial_distribution(
        positions_cutouts,
        jwst_pointing = None,
        n_bins= 6,
):
    """Get the radial histogram of the neighbors for all of the bright galaxies"""
    if jwst_pointing is None:
        jwst_pointing = setup_JWST()
    bins = np.linspace(0.01, jwst_pointing.NIRSpec_Lx * jwst_pointing.n_sky * np.sqrt(2), n_bins)
    histogram = np.apply_along_axis(
        lambda x: np.histogram(
            x,
            bins=bins
        )[0],
        1,
        positions_cutouts
    )
    return histogram


def cutout_selection(
        position_binned,
        mags_cutout,
        logmhs_cutout,
        coords_cutouts_x,
        coords_cutouts_y,
        coords_cutouts_z,
        num_neighbors=2,
):
    """
    Function selects cutouts based on a certain criterion.
    So far the only criterion is number of neighbors, but that can easily be changed.
    """
    mask = np.cumsum(position_binned, axis=1)[:, -1] >= num_neighbors
    mags_chosen = [magi for i,magi in enumerate(mags_cutout) if mask[i]]
    logmhs_chosen = [mhi for i, mhi in enumerate(logmhs_cutout) if mask[i]]
    coords_chosen_x = [coordxi for i, coordxi in enumerate(coords_cutouts_x) if mask[i]]
    coords_chosen_y = [coordyi for i, coordyi in enumerate(coords_cutouts_y) if mask[i]]
    coords_chosen_z = [coordzi for i, coordzi in enumerate(coords_cutouts_z) if mask[i]]

    return coords_chosen_x, coords_chosen_y, coords_chosen_z, mags_chosen, logmhs_chosen


def triangle_area(a, b, c):
    """
    Compute area of triangle given by 3 points in 3D.

    Parameters:
        a, b, c: array-like, shape (3,)

    Returns:
        float: area of the triangle
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Compute edge vectors
    ab = b - a
    ac = c - a

    # Cross product magnitude = 2 * area of triangle
    cross_prod = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_prod)


def process_cutouts(
        coords_cutouts_x,
        coords_cutouts_y,
        coords_cutouts_z,
        logmhs_cutout,
        mags_cutout,
):
    n = len(mags_cutout)
    d1s = np.zeros(n)
    d2s = np.zeros(n)
    d3s = np.zeros(n)
    d14s = np.zeros(n)
    tot_lengths = np.zeros(n)
    areas = np.zeros(n)
    max_out_of_3s = np.zeros(n)
    dict_stoch = {}
    mhs_brightest_stochier = []
    mag_brightest_stochier_arr = []

    for index, (mhiii, magi, cx, cy, cz) in enumerate(
            zip(logmhs_cutout, mags_cutout, coords_cutouts_x, coords_cutouts_y,
                coords_cutouts_z)
    ):
