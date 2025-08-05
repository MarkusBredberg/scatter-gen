import astropy

# Read the header from a fits file
from astropy.io import fits
header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G031.93+78.71/PSZ2G031.93+78.71T100kpcSUB.fits')
print("Header for PSZ2G031.93+78.71 (T100kpcSUB):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G031.93+78.71/PSZ2G031.93+78.71T50kpc.fits')
print("Header for PSZ2G031.93+78.71 (T50kpc):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G053.80+36.49/PSZ2G053.80+36.49.fits')
print("Header for PSZ2G053.80+36.49 (RAW):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G149.22+54.18/PSZ2G149.22+54.18.fits')
print("Header for PSZ2G149.22+54.18 (RAW):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G149.22+54.18/PSZ2G149.22+54.18T100kpcSUB.fits')
print("Header for PSZ2G149.22+54.18 (T100kpc):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G149.22+54.18/PSZ2G149.22+54.18T50kpc.fits')
print("Header for PSZ2G149.22+54.18 (T50kpc):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G055.59+31.85/PSZ2G055.59+31.85.fits')
print("Header for PSZ2G055.59+31.85 (RAW):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G101.52-29.98/PSZ2G101.52-29.98.fits')
print("Header for PSZ2G101.52-29.98 (RAW):", header)

header = fits.getheader('/users/mbredber/scratch/data/PSZ2/fits/PSZ2G106.61+66.71/PSZ2G106.61+66.71.fits')
print("Header for PSZ2G106.61+66.71 (RAW):", header)