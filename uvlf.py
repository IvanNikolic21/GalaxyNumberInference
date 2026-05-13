"""
A module defining model predictions from Mason+15 (and Mason+23, Nikolic+26) model.

The module includes functions that process the input parameters and returns UVLF.

"""

import os

import hmf
import numpy as np


def pMuv_Mh(Muv, logMh, Muv_Mh_dict, sigmaUV_a=-0.34, sigmaUV_b=0.42, Muv_add=0, return_med_sigma=False):
    """
    sigmaUV (Mh) = a (log Mh - 12) + b.
    a = −0.34 and b = 0.42,
    """
    sigmaUV = sigmaUV_a * (logMh - 12) + sigmaUV_b
    # Ensure sigmaUV is strictly positive to define a valid Gaussian PDF
    sigmaUV_floor = 1e-3
    if np.isscalar(sigmaUV):
        sigmaUV = max(sigmaUV, sigmaUV_floor)
    else:
        sigmaUV = np.maximum(sigmaUV, sigmaUV_floor)

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

def Muv_from_logMh(logMh, Muv_Mh_dict, use_scatter=True, sigmaUV=0.3, dust=False):
    if use_scatter:
        scatter = np.random.normal(0, sigmaUV)
    else:
        scatter = 0
    if dust:
        return Muv_Mh_dict[logMh][1] + scatter
    return Muv_Mh_dict[logMh][0] + scatter

def calculate_uvlf(Muv_shift, sigma_UV_a, sigma_UV_b , mf = None, Muv_grid = None, z = None, Muv_Mh_dict = None):
    if z is None:
        z = 10.0
    if mf is None:
        mf = hmf.MassFunction(
            z=z,
            Mmin=8,
            Mmax=14,
            dlog10m=0.05,
        )
    if Muv_grid is None:
        Muv_grid = np.linspace(-25, -13, 100)

    if z==10.0 and Muv_Mh_dict is None :
        Muv_Mh_file = '/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt'
        Muv_Mh = np.genfromtxt(Muv_Mh_file, dtype=None, names=True)
        Muv_Mh_dict = {Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))}


    pmuvmh = pMuv_Mh(Muv_grid, np.log10(mf.m), Muv_Mh_dict, sigmaUV_a=sigma_UV_a, sigmaUV_b=sigma_UV_b, Muv_add=Muv_shift)
    UVLF_stochier = np.trapz(pmuvmh.T * mf.dndm, mf.m, axis=1)
    return UVLF_stochier

class Mason15(object):
    def __init__(self, z=10.0, hm_inst = None):
        self.z = z
        self.mf = hmf.MassFunction(z=z, Mmin=8, Mmax=14, dlog10m=0.05)

        if z==10.0:
            Muv_Mh_file = '/groups/astro/ivannik/notebooks/clustering_project/Muv_Mh_z=10.txt'
            Muv_Mh = np.genfromtxt(Muv_Mh_file, dtype=None, names=True)
            self.Muv_Mh_dict = {Muv_Mh['logMh'][i]: [Muv_Mh['Muv'][i], Muv_Mh['Muv_dust'][i]] for i in range(len(Muv_Mh))}

    def calculate_UVLF(self, Muv_shift, sigma_UV_a, sigma_UV_b, Muv_grid = np.linspace(-25, -13, 100)):
        UVLF_pred = calculate_uvlf(
            Muv_shift, sigma_UV_a, sigma_UV_b, mf=self.mf, Muv_grid = Muv_grid, z=self.z, Muv_Mh_dict = self.Muv_Mh_dict
        )
        return UVLF_pred





class Observations():
    def __init__(self, ang, uvlf):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # if script_dir == "/groups/astro/ivannik/programs/JWST-Inference":
        #     dir_dat = "/groups/astro/ivannik/programs/JWST-Inference/data/Paquereau_2025_clustering/GalClustering_COSMOS-Web_Paquereau2025/clustering_measurements/"
        dir_dat = script_dir + "/data/Paquereau_2025_clustering/GalClustering_COSMOS-Web_Paquereau2025/clustering_measurements/"
        if ang:
            self.cons_theta = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_theta.dat"
                ).readlines()
            ]
            self.cons_wtheta = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wtheta.dat"
                ).readlines()
            ]
            self.cons_wsig = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin8.0-10.5_conservative_wsig.dat"
                ).readlines()
            ]

            self.theta_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_theta.dat"
                ).readlines()
            ]
            self.wtheta_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_wtheta.dat"
                ).readlines()
            ]
            self.wsig_z7 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin6.0-8.0_wsig.dat"
                ).readlines()
            ]

            self.theta_z5_5 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin5.0-6.0_theta.dat"
                ).readlines()
            ]
            self.wtheta_z5_5 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin5.0-6.0_wtheta.dat"
                ).readlines()
            ]
            self.wsig_z5_5 = [
                i.strip().split() for i in open(
                    dir_dat + "clustresults_Paquereau2025_COSMOS-Web_FullSurvey_zbin5.0-6.0_wsig.dat"
                ).readlines()
            ]
        if uvlf:
            pass

    def get_obs_z9_m87(self):
        thethats87_pos = [
            float(i) for i,j in zip(self.cons_theta[1],self.cons_wtheta[1]) if float(j)>0
        ]
        wthethats87_pos = [
            float(i) for i,j in zip(self.cons_wtheta[1],self.cons_wtheta[1]) if float(j)>0
        ]
        wsig87_pos = [
            float(i) for i,j in zip(self.cons_wsig[1],self.cons_wtheta[1]) if float(j)>0
        ]
        return thethats87_pos, wthethats87_pos, wsig87_pos

    def get_obs_z9_m90(self):
        thethats90_pos = [
            float(i) for i,j in zip(self.cons_theta[2],self.cons_wtheta[2]) if float(j)>0
        ]
        wthethats90_pos = [
            float(i) for i,j in zip(self.cons_wtheta[2],self.cons_wtheta[2]) if float(j)>0
        ]
        wsig90_pos = [
            float(i) for i,j in zip(self.cons_wsig[2],self.cons_wtheta[2]) if float(j)>0
        ]
        return thethats90_pos, wthethats90_pos, wsig90_pos

    def get_obs_z7_m87(self):
        thethats87_pos_z7 = [
            float(i) for i,j in zip(self.theta_z7[1],self.wtheta_z7[1]) if float(j)>0
        ]
        wthethats87_pos_z7 = [
            float(i) for i,j in zip(self.wtheta_z7[1],self.wtheta_z7[1]) if float(j)>0
        ]
        wsig87_pos_z7 = [
            float(i) for i,j in zip(self.wsig_z7[1],self.wtheta_z7[1]) if float(j)>0
        ]
        return thethats87_pos_z7, wthethats87_pos_z7, wsig87_pos_z7

    def get_obs_z7_m90(self):
        thethats90_pos_z7 = [
            float(i) for i,j in zip(self.theta_z7[2],self.wtheta_z7[2]) if float(j)>0
        ]
        wthethats90_pos_z7 = [
            float(i) for i,j in zip(self.wtheta_z7[2],self.wtheta_z7[2]) if float(j)>0
        ]
        wsig90_pos_z7 = [
            float(i) for i,j in zip(self.wsig_z7[2],self.wtheta_z7[2]) if float(j)>0
        ]
        return thethats90_pos_z7, wthethats90_pos_z7, wsig90_pos_z7

    def get_obs_z7_m93(self):
        thethats93_pos_z7 = [
            float(i) for i,j in zip(self.theta_z7[3],self.wtheta_z7[3]) if float(j)>0
        ]
        wthethats93_pos_z7 = [
            float(i) for i,j in zip(self.wtheta_z7[3],self.wtheta_z7[3]) if float(j)>0
        ]
        wsig93_pos_z7 = [
            float(i) for i,j in zip(self.wsig_z7[3],self.wtheta_z7[3]) if float(j)>0
        ]
        return thethats93_pos_z7, wthethats93_pos_z7, wsig93_pos_z7

    def get_obs_z5_5_m85(self):
        thethats85_pos_z5_5 = [
            float(i) for i,j in zip(self.theta_z5_5[1],self.wtheta_z5_5[1]) if float(j)>0
        ]
        wthethats85_pos_z5_5 = [
            float(i) for i,j in zip(self.wtheta_z5_5[1],self.wtheta_z5_5[1]) if float(j)>0
        ]
        wsig85_pos_z5_5 = [
            float(i) for i,j in zip(self.wsig_z5_5[1],self.wtheta_z5_5[1]) if float(j)>0
        ]
        return thethats85_pos_z5_5, wthethats85_pos_z5_5, wsig85_pos_z5_5

    def get_obs_z5_5_m90(self):
        thethats90_pos_z5_5 = [
            float(i) for i,j in zip(self.theta_z5_5[2],self.wtheta_z5_5[2]) if float(j)>0
        ]
        wthethats90_pos_z5_5 = [
            float(i) for i,j in zip(self.wtheta_z5_5[2],self.wtheta_z5_5[2]) if float(j)>0
        ]
        wsig90_pos_z5_5 = [
            float(i) for i,j in zip(self.wsig_z5_5[2],self.wtheta_z5_5[2]) if float(j)>0
        ]
        return thethats90_pos_z5_5, wthethats90_pos_z5_5, wsig90_pos_z5_5

    def get_obs_z5_5_m92_5(self):
        thethats92_5_pos_z5_5 = [
            float(i) for i,j in zip(self.theta_z5_5[3],self.wtheta_z5_5[3]) if float(j)>0
        ]
        wthethats92_5_pos_z5_5 = [
            float(i) for i,j in zip(self.wtheta_z5_5[3],self.wtheta_z5_5[3]) if float(j)>0
        ]
        wsig92_5_pos_z5_5 = [
            float(i) for i,j in zip(self.wsig_z5_5[3],self.wtheta_z5_5[3]) if float(j)>0
        ]
        return thethats92_5_pos_z5_5, wthethats92_5_pos_z5_5, wsig92_5_pos_z5_5

    def get_obs_z5_5_m95(self):
        thethats95_pos_z5_5 = [
            float(i) for i,j in zip(self.theta_z5_5[4],self.wtheta_z5_5[4]) if float(j)>0
        ]
        wthethats95_pos_z5_5 = [
            float(i) for i,j in zip(self.wtheta_z5_5[4],self.wtheta_z5_5[4]) if float(j)>0
        ]
        wsig95_pos_z5_5 = [
            float(i) for i,j in zip(self.wsig_z5_5[4],self.wtheta_z5_5[4]) if float(j)>0
        ]
        return thethats95_pos_z5_5, wthethats95_pos_z5_5, wsig95_pos_z5_5

    def get_obs_uvlf_z11_McLeod23(self):
        McL_Muvs = np.array(
            [-22.57,-21.80,-20.80,-20.05,-19.55,-18.85,-18.23]
        )
        McL_uvlf = np.array(
            [0.012,0.128,1.251,3.951,9.713,23.490,63.080]
        )*1e-5
        Mcl_sig = np.array(
            [0.010,0.128,0.424,1.319, 4.170,9.190,28.650]
        )*1e-5

        return McL_Muvs, McL_uvlf, Mcl_sig

    def get_obs_uvlf_z9_Donnan24(self):
        Don_Muvs = np.array(
            [-20.75,-20.25,-19.75,-19.25,-18.55,-18.05,-17.55]
        )
        Don_uvlf = np.array(
            [12,32,144,235,486,1110,1776]
        )*1e-6
        Don_sig_p = np.array(
            [8,13,30,60,157,310,578]
        )*1e-6
        Don_sig_m = np.array(
            [5,10,28,49,139,310,510]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)

    def get_obs_uvlf_z10_Donnan24(self):
        Don_Muvs = np.array(
            [-20.75,-20.25,-19.75,-19.25,-18.55,-18.05,-17.55]
        )
        Don_uvlf = np.array(
            [4,27,92,177,321,686,1278]
        )*1e-6
        Don_sig_p = np.array(
            [10,13,25,53,127,245,486]
        )*1e-6
        Don_sig_m = np.array(
            [4,10,20,45,111,223,432]
        )*1e-6
        return Don_Muvs, Don_uvlf, (Don_sig_p, Don_sig_m)
