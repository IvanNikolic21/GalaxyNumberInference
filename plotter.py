import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

def CDF_all_gal_talk_version(d1, d1s, mag_brightest_fiducial_arr, mag_brightest_stochier_arr):
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0.0, 10, 70)
    bins_mag = np.linspace(-22.5, -18, 70)
    vals0 = np.histogram(d1, bins=bins)
    vals1 = np.histogram(d1s, bins=bins)
    # vals0 = axs.hist(d1, bins=bins,alpha=0.5, density=True,color='red', label='enhances luminosity')
    # vals1 = axs.hist(d1s, bins=bins,alpha=0.5, density=True,color='blue', label='enhances luminosity')

    plt.plot(0.5 * (vals0[1][1:] + vals0[1][:-1]), np.cumsum(vals0[0]) / np.cumsum(vals0[0])[-1], color='#a63603', lw=3)
    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    plt.plot(0.5 * (vals1[1][1:] + vals1[1][:-1]), np.cumsum(vals1[0]) / np.cumsum(vals1[0])[-1], color='#08519c', lw=3)
    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jackys[3], color='blue', alpha=0.5)
    plt.xlabel(r'$d_{12}$ = separation of two brightest galaxies [cMpc]', fontsize=14)
    plt.ylabel(r'CDF (separation)', fontsize=14)
    plt.text(1.65, 0.90, 'intrinsically bright', color='#a63603', rotation=3)
    plt.text(2.5, 0.61, 'increased stochasticity', color='#08519c', rotation=21)

    ax_ad = fig.add_subplot((0.48, 0.25, 0.4, 0.4))
    val = ax_ad.hist(
        np.array(mag_brightest_fiducial_arr)[:, 1], bins=bins_mag, alpha=0.5, density=True, color='#a63603',
        label='brightest galaxy,\nintrinsically bright'
    )
    vals = ax_ad.hist(
        np.array(mag_brightest_stochier_arr)[:, 1], bins=bins_mag, alpha=0.5, density=True, color='#08519c',
        label='brightest galaxy,\nintrinsically bright')

    ax_ad.spines[['left', 'right', 'top']].set_visible(False)
    ax_ad.tick_params(top=False, left=False, right=False, labelleft=False)
    ax_ad.set_xlabel(r'$M_{\rm UV}$ of the 2nd brightest', fontsize=16)
    plt.savefig('/groups/astro/ivannik/projects/Neighbors/CDF_all_gal_talk_version.pdf', bbox_inches='tight') #change to alternative folder

def CDF_all_gal_paper_version(d1, d1s):
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0.0, 10, 70)
    vals0 = np.histogram(d1, bins=bins)
    vals1 = np.histogram(d1s, bins=bins)
    # vals0 = axs.hist(d1, bins=bins,alpha=0.5, density=True,color='red', label='enhances luminosity')
    # vals1 = axs.hist(d1s, bins=bins,alpha=0.5, density=True,color='blue', label='enhances luminosity')

    axs.plot(0.5 * (vals0[1][1:] + vals0[1][:-1]), np.cumsum(vals0[0]) / np.cumsum(vals0[0])[-1], color='#a63603', lw=3,
             label='intrinsically bright')
    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(0.5 * (vals1[1][1:] + vals1[1][:-1]), np.cumsum(vals1[0]) / np.cumsum(vals1[0])[-1], color='#08519c', lw=3,
             label='increased stochasticity')
    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jackys[3], color='blue', alpha=0.5)
    axs.set_xlabel(r'$d_{12}$ = separation of two brightest galaxies [cMpc]', fontsize=14)
    axs.set_ylabel(r'CDF (separation)', fontsize=14)
    plt.legend()
    plt.savefig('/groups/astro/ivannik/projects/Neighbors/CDF_all_gal_paper_version.pdf', bbox_inches='tight')

def CDF_all_gal_paper_version_split_by_N(dict_fid, dict_stoch):
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    bins = np.linspace(0.0, 13, 70)
    bins_mag = np.linspace(-22, -18, 70)
    vals0 = np.histogram(dict_fid['4']['d1s'], bins=bins)
    vals1 = np.histogram(dict_stoch['4']['d1s'], bins=bins)
    colors_fid = ['#d94701', '#fd8d3c', '#fdbe85']
    colors_stoc = ['#2171b5', '#6baed6', '#bdd7e7']
    axs.plot(0.5 * (vals0[1][1:] + vals0[1][:-1]), np.cumsum(vals0[0]) / np.cumsum(vals0[0])[-1], color=colors_fid[2],
             lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(0.5 * (vals1[1][1:] + vals1[1][:-1]), np.cumsum(vals1[0]) / np.cumsum(vals1[0])[-1], color=colors_stoc[2],
             lw=3)

    vals0 = np.histogram(dict_fid['6']['d1s'], bins=bins)
    vals1 = np.histogram(dict_stoch['6']['d1s'], bins=bins)

    axs.plot(0.5 * (vals0[1][1:] + vals0[1][:-1]), np.cumsum(vals0[0]) / np.cumsum(vals0[0])[-1], color=colors_fid[1],
             lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(0.5 * (vals1[1][1:] + vals1[1][:-1]), np.cumsum(vals1[0]) / np.cumsum(vals1[0])[-1], color=colors_stoc[1],
             lw=3)

    vals0 = np.histogram(dict_fid['9']['d1s'], bins=bins)
    vals1 = np.histogram(dict_stoch['9']['d1s'], bins=bins)

    axs.plot(0.5 * (vals0[1][1:] + vals0[1][:-1]), np.cumsum(vals0[0]) / np.cumsum(vals0[0])[-1], color=colors_fid[0],
             lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(0.5 * (vals1[1][1:] + vals1[1][:-1]), np.cumsum(vals1[0]) / np.cumsum(vals1[0])[-1], color=colors_stoc[0],
             lw=3)

    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jackys[3], color='blue', alpha=0.5)
    axs.set_xlabel(r'$d_{12}$ = separation of two brightest galaxies [cMpc]', fontsize=14)
    axs.set_ylabel(r'CDF(separation)', fontsize=14)
    # plt.text(1.65, 0.90,'intrinsically bright', color='red', rotation=3)
    # plt.text(2.5, 0.61,'increased stochasticity', color='blue', rotation=21)

    color_boxes = [
        Line2D([0], [0], marker='s', linestyle='None',
               markersize=8, markerfacecolor=c, markeredgecolor='none')
        for c in colors_fid
    ]
    x0, y0 = 0.3, 0.15
    dx = 0.025
    for i, c in enumerate(colors_fid):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'intrinsically bright', transform=axs.transAxes)
    x0, y0 = 0.3, 0.11
    dx = 0.025
    for i, c in enumerate(colors_stoc):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'increased stochasticity', transform=axs.transAxes)
    # axs.arrow(x0, 0.19, 3 * dx, 0, transform = ax.transAxes, head_width=0.01, length_includes_head=True, facecolor='black')
    axs.text(x0 - 1.5 * dx, 0.06, 'N=', transform=axs.transAxes, fontsize=10)
    axs.text(x0, 0.06, '9', transform=axs.transAxes, fontsize=10)
    axs.text(x0 + dx, 0.06, '6', transform=axs.transAxes, fontsize=10)
    axs.text(x0 + 2 * dx, 0.06, '4', transform=axs.transAxes, fontsize=10)
    plt.savefig('/groups/astro/ivannik/projects/Neighbors/CDF_all_gal_paper_version_split_by_N.pdf',
                bbox_inches='tight')


def UV_properties_brightest():
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    bins = np.linspace(0.0, 13, 70)
    bins_mag = np.linspace(-23.5, -18, 70)

    colors_fid = ['#d94701', '#fd8d3c', '#fdbe85']
    colors_stoc = ['#2171b5', '#6baed6', '#bdd7e7']

    kde_fid = gaussian_kde(np.array(mag_brightest_fiducial_arr)[:, 2])
    kde_stoc = gaussian_kde(np.array(mag_brightest_stochier_arr)[:, 2])

    axs.plot(bins_mag, kde_fid(bins_mag), color=colors_fid[2], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mag, kde_stoc(bins_mag), color=colors_stoc[2], lw=3)

    kde_fid = gaussian_kde(np.array(mag_brightest_fiducial_arr)[:, 1])
    kde_stoc = gaussian_kde(np.array(mag_brightest_stochier_arr)[:, 1])

    axs.plot(bins_mag, kde_fid(bins_mag), color=colors_fid[1], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mag, kde_stoc(bins_mag), color=colors_stoc[1], lw=3)

    kde_fid = gaussian_kde(np.array(mag_brightest_fiducial_arr)[:, 0])
    kde_stoc = gaussian_kde(np.array(mag_brightest_stochier_arr)[:, 0])

    axs.plot(bins_mag, kde_fid(bins_mag), color=colors_fid[0], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mag, kde_stoc(bins_mag), color=colors_stoc[0], lw=3)

    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jackys[3], color='blue', alpha=0.5)
    axs.set_xlabel(r'M_{\rm UV}', fontsize=14)
    axs.set_ylabel(r'PDF', fontsize=14)
    # plt.text(1.65, 0.90,'intrinsically bright', color='red', rotation=3)
    # plt.text(2.5, 0.61,'increased stochasticity', color='blue', rotation=21)

    color_boxes = [
        Line2D([0], [0], marker='s', linestyle='None',
               markersize=8, markerfacecolor=c, markeredgecolor='none')
        for c in colors_fid
    ]
    x0, y0 = 0.47, 0.95
    dx = 0.025
    for i, c in enumerate(colors_fid):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'intrinsically bright', transform=axs.transAxes)
    x0, y0 = 0.47, 0.88
    dx = 0.025
    for i, c in enumerate(colors_stoc):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'increased stochasticity', transform=axs.transAxes)
    axs.arrow(x0, 00.84, 3 * dx, 0, transform=axs.transAxes, head_width=0.01, length_includes_head=True,
              facecolor='black')
    axs.text(x0 - 1.5 * dx, 0.81, 'decreasing rank', transform=axs.transAxes, fontsize=10)

    plt.savefig('/groups/astro/ivannik/projects/Neighbors/UV_properties_brightest.pdf', bbox_inches='tight')


def attempt_at_plotting_density():
    fig, ax = plt.subplots(figsize=(8, 8))
    mappable = plt.imshow(
        np.mean(density[
                    int(min(coordsx_fiducial_chosen[220]) * 0.5): int(max(coordsx_fiducial_chosen[220]) * 0.5),
                    int(min(coordsy_fiducial_chosen[220]) * 0.5): int(max(coordsy_fiducial_chosen[220]) * 0.5),
                    int(min(coordsz_fiducial_chosen[220]) * 0.5): int(max(coordsz_fiducial_chosen[220]) * 0.5),
                ], axis=2).T,
        extent=(2.0 * int(min(coordsx_fiducial_chosen[220]) * 0.5), 2.0 * int(max(coordsx_fiducial_chosen[220]) * 0.5),
                2.0 * int(max(coordsy_fiducial_chosen[220]) * 0.5), 2.0 * int(min(coordsy_fiducial_chosen[220]) * 0.5)),
    )
    plt.colorbar()

    plt.scatter(coordsx_fiducial_chosen[220], coordsy_fiducial_chosen[220])  # , coordsz_fiducial_chosen[50]


def better_2D_density():


    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    index = 20
    index_stoch = 55

    vals_stoch = -np.array(mags_stochier_chosen[index_stoch])  # or mass, or whatever

    vals = -np.array(mags_fiducial_chosen[index])

    from matplotlib.colors import Normalize

    norm_scatter = Normalize(
        vmin=min(vals.min(), vals_stoch.min()),
        vmax=max(vals.max(), vals_stoch.max()),
    )
    coordsx_low = coordsx_fiducial_chosen[index] // 2
    coordsy_low = coordsy_fiducial_chosen[index] // 2
    coordsz_low = coordsz_fiducial_chosen[index] // 2
    z0 = int(np.median(coordsz_low))
    x0 = int(np.median(coordsx_low))
    print(mask)
    y0 = int(np.median(coordsy_low))
    # rho_halo = density[np.array(coordsx_low).astype(int), np.array(coordsy_low).astype(int), np.array(coordsz_low).astype(int)]

    # plt.figure()
    # plt.hist(rho_halo, bins=50)
    # plt.xlabel("Density at halo position")
    proj = np.mean(density[x0 - 5:x0 + 5, y0 - 5:y0 + 5, z0 - 5:z0 + 5], axis=2)
    mapppable = axes[0].imshow(
        proj.T,
        origin="lower",
        cmap="magma",
        extent=(x0 - 4.5, x0 + 5.5, y0 + 5 + 2, y0 - 5 + 2)
    )
    print(logmhs_fiducial_chosen[index])
    sc = axes[0].scatter(
        coordsx_fiducial_chosen[index] / 2,
        coordsy_fiducial_chosen[index] / 2,
        # s=150 * (-np.array(mags_fiducial_chosen[index]) - 18),
        c=vals,
        s=200 * (vals - 18),
        cmap="Greens",
        norm=norm_scatter,
        marker="*",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_title('intrinsically bright')

    coordsx_low = coordsx_stochier_chosen[index_stoch] // 2
    coordsy_low = coordsy_stochier_chosen[index_stoch] // 2
    coordsz_low = coordsz_stochier_chosen[index_stoch] // 2
    z0 = int(np.median(coordsz_low))
    x0 = int(np.median(coordsx_low))

    y0 = int(np.median(coordsy_low))
    vmin, vmax = mappable.get_clim()

    proj = np.mean(density[x0 - 5:x0 + 5, y0 - 5:y0 + 5, z0 - 5:z0 + 5], axis=2)
    axes[1].imshow(
        proj,
        origin="lower",
        cmap="magma",
        extent=(x0 - 5 - 2, x0 + 5 - 2, y0 - 5, y0 + 5),
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].scatter(
        coordsx_stochier_chosen[index_stoch] / 2,
        coordsy_stochier_chosen[index_stoch] / 2,
        # s=200 * (-np.array(mags_stochier_chosen[index_stoch]) - 18),
        c=vals_stoch,
        s=200 * (vals_stoch - 18),
        cmap="Greens",
        norm=norm_scatter,
        marker="*",
        edgecolor="black",
        linewidth=0.3,
    )

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("top", size="5%", pad=0.45)
    cbdelt = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cbdelt.set_label(r'overdensity $\delta$')

    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    divider2 = make_axes_locatable(axes[0])

    cax_scatter = divider2.append_axes("top", size="5%", pad=0.45)

    cb_sc = fig.colorbar(sc, cax=cax_scatter, orientation="horizontal")
    cb_sc.set_label(r"$-M_{\rm UV}$")

    cax_scatter.xaxis.set_ticks_position("top")
    axes[1].set_title('increased stochasticity')

    cax_scatter.xaxis.set_label_position("top")
    # , coordsz_fiducial_chosen[50]

    plt.savefig('/groups/astro/ivannik/projects/Neighbors/visualization_21cmfast_density.pdf', bbox_inches='tight')


def halo_mass_rank(mhs_brightest_fiducial, mhs_brightest_stochier):

    """

    :param mhs_brightest_fiducial: list of arrays of halo masses of brightest candidates (fiducial case)
    :param mhs_brightest_stochier:  list of arrays of halo masses of brightest candidates (stochastic case)
    :return: None, just plot
    """

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    bins = np.linspace(0.0, 13, 70)
    bins_mh = np.linspace(9.5, 11.6, 100)

    def stack_truncated(arr_list):
        min_len = min(len(a) for a in arr_list)
        return np.vstack([a[:min_len] for a in arr_list])

    mhs_brightest_fiducial_trunc = stack_truncated(mhs_brightest_fiducial)
    mhs_brightest_stochier_trunc = stack_truncated(mhs_brightest_stochier)

    colors_fid = ['#d94701', '#fd8d3c', '#fdbe85']
    colors_stoc = ['#2171b5', '#6baed6', '#bdd7e7']

    kde_fid = gaussian_kde(np.array(mhs_brightest_fiducial_trunc)[:, 2], bw_method=0.3)
    kde_stoc = gaussian_kde(np.array(mhs_brightest_stochier_trunc)[:, 2], bw_method=0.3)

    axs.plot(bins_mh, kde_fid(bins_mh), color=colors_fid[2], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mh, kde_stoc(bins_mh), color=colors_stoc[2], lw=3)

    kde_fid = gaussian_kde(np.array(mhs_brightest_fiducial_trunc)[:, 1], bw_method=0.3)
    kde_stoc = gaussian_kde(np.array(mhs_brightest_stochier_trunc)[:, 1], bw_method=0.3)

    axs.plot(bins_mh, kde_fid(bins_mh), color=colors_fid[1], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mh, kde_stoc(bins_mh), color=colors_stoc[1], lw=3)

    kde_fid = gaussian_kde(np.array(mhs_brightest_fiducial_trunc)[:, 0], bw_method=0.4)
    kde_stoc = gaussian_kde(np.array(mhs_brightest_stochier_trunc)[:, 0], bw_method=0.4)

    axs.plot(bins_mh, kde_fid(bins_mh), color=colors_fid[0], lw=3)
    # axs.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jacky[3], color='red', alpha=0.5)
    axs.plot(bins_mh, kde_stoc(bins_mh), color=colors_stoc[0], lw=3)

    # plt.fill_between(0.5*(vals0[1][1:] + vals0[1][:-1]), *jackys[3], color='blue', alpha=0.5)
    axs.set_xlabel(r'$\log{(M_{\rm h} / M_{\odot})}$', fontsize=14)
    axs.set_ylabel(r'PDF', fontsize=14)
    # plt.text(1.65, 0.90,'intrinsically bright', color='red', rotation=3)
    # plt.text(2.5, 0.61,'increased stochasticity', color='blue', rotation=21)

    color_boxes = [
        Line2D([0], [0], marker='s', linestyle='None',
               markersize=8, markerfacecolor=c, markeredgecolor='none')
        for c in colors_fid
    ]
    x0, y0 = 0.07, 0.95
    dx = 0.025
    for i, c in enumerate(colors_fid):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'intrinsically bright', transform=axs.transAxes)
    x0, y0 = 0.07, 0.88
    dx = 0.025
    for i, c in enumerate(colors_stoc):
        rect = mpatches.Rectangle(
            (x0 + i * dx, y0 - 0.01),
            0.015, 0.02,
            transform=axs.transAxes,
            facecolor=c,
            edgecolor='none'
        )
        axs.add_patch(rect)
    axs.text(x0 + 3 * dx, y0 - 0.01, 'increased stochasticity', transform=axs.transAxes)
    axs.arrow(x0, 00.84, 3 * dx, 0, transform=axs.transAxes, head_width=0.01, length_includes_head=True,
              facecolor='black')
    axs.text(x0 - 1.5 * dx, 0.81, 'decreasing rank', transform=axs.transAxes, fontsize=10)

    plt.savefig('/groups/astro/ivannik/projects/Neighbors/halomass_properties_brightest.pdf', bbox_inches='tight')