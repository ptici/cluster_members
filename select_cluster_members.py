'''
script for reading photometry data from GAIA mission and selecting probable star members
from a given stellar cluster
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## reading data
df = pd.read_csv('GAIA_foo.csv')
df.fillna("nan", inplace=True)

G, Gbp, Grp = df['phot_g_mean_mag'], df['phot_bp_mean_mag'], df['phot_rp_mean_mag']
plx = df['parallax']
pmra, pmdec = df['pmra'], df['pmdec']
ra, dec = df['ra'], df['dec']
ruwe = df['ruwe']

## Cleaning dataframe
RA, DEC, PMRA, PMDEC, PLX, g, gbp, grp = [], [], [], [], [], [], [], []

for i in range(len(ra)):
    if (float(G.iloc[i]) < 20) * (float(G.iloc[i]) > 8) * (float(ruwe.iloc[i]) <= 1.4) * (G.iloc[i] != 'nan') * (Gbp.iloc[i] != 'nan') * (Grp.iloc[i] != 'nan') * (plx.iloc[i] != 'nan') * (pmra.iloc[i] != 'nan') * (pmdec.iloc[i] != 'nan') * (ra.iloc[i] != 'nan') * (dec.iloc[i] != 'nan'):
        RA.append(ra.iloc[i]); DEC.append(dec.iloc[i])
        PMRA.append(pmra.iloc[i]); PMDEC.append(pmdec[i])
        PLX.append(plx.iloc[i])
        g.append(G.iloc[i]); gbp.append(Gbp.iloc[i]); grp.append(Grp.iloc[i])

## making arrays
ra_orig, dec_orig = np.array(RA), np.array(DEC)
pmra_orig, pmdec_orig = np.array(PMRA), np.array(PMDEC)
plx_orig = np.array(PLX)
G_orig, Gbp_orig, Grp_orig = np.array(g), np.array(gbp), np.array(grp)

## mask creation to find members
pm_ra_mask = (pmra_orig < 1.2) & (pmra_orig > 0)
pm_dec_mask = (pmdec_orig < 0) & (pmdec_orig > -2)
parallax_mask = (plx_orig < 0.1) & (plx_orig > -0.1)
membro_mask = pm_ra_mask & pm_dec_mask & parallax_mask

## applying mask
ra, dec = ra_orig[membro_mask], dec_orig[membro_mask]
pmra, pmdec = pmra_orig[membro_mask], pmdec_orig[membro_mask]
G, Gbp, Grp = G_orig[membro_mask], Gbp_orig[membro_mask], Grp_orig[membro_mask]
plx = plx_orig[membro_mask]


## saving mags from NGC330 members for test in ML models
data_foo = pd.DataFrame()
dictx = {'RA': ra, 'Dec': dec, 'pmra': pmra, 'pmdec': pmdec, 'parallax': plx}
data_foo = data_foo.append(dictx, ignore_index=True)
data_foo.to_csv('cluster_members_mags_foo.csv', index=False)

## print how many stars were initially, on cleaned DF, and probable members
print(f"O aglomerado tem {len(G)} estrelas membros.")
print(f"Valor original: {len(G_orig)} estrelas")

## proper motion on RA and DEC; blue star = mean position
plt.figure(figsize=(10,9.5))
plt.plot(pmdec_orig, pmra_orig, 'ko', markersize=2)
plt.plot(np.mean(pmdec_orig), np.mean(pmra_orig), 'b*', markersize=15)
plt.xlabel('pm dec', fontsize=18)
plt.ylabel('pm ra', fontsize=18)
plt.xlim(-3, 1.3)
plt.ylim(-1, 2.3)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
plt.clf()

## RA and DEC; blue star = mean position
plt.figure(figsize=(10,9.5))
plt.plot(dec_orig, ra_orig, 'ko', markersize=2)
plt.plot(np.mean(dec_orig), np.mean(ra_orig), 'b*', markersize=15)
plt.xlabel('dec', fontsize=18)
plt.ylabel('ra', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
plt.clf()


## GAIA observations
plt.figure(figsize=(9,9.5))
plt.plot((Gbp - Grp), G, 'ko', markersize=5, label='Probable members')
plt.plot((Gbp_orig - Grp_orig), G_orig, 'ro', markersize=1, alpha=0.35, label='Original data (clean; RUEW <= 1.4)')
plt.xlabel('Gbp - Grp', fontsize=18)
plt.ylabel('G', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().invert_yaxis()
plt.legend(fontsize=13)
plt.show()

