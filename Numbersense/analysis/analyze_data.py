from typing import Optional
import numpy as np
from numpy import fft
from scipy.stats import weibull_min
from scipy.ndimage import gaussian_gradient_magnitude

# NOTE: DISCARD ALPHA CHANNEL

# ************** Spatial frequency **************

def rotational_average_power_spectrum(image_freqs:np.ndarray, magnitude:bool = False):
    ImgSize = len(image_freqs)
    rAvg = np.zeros(int(ImgSize/2))

    ## Zero-Frequency Value of the Power Spectrum (Magnitude would be without **2)
    value = np.abs(image_freqs[int(ImgSize/2), int(ImgSize/2)])
    rAvg[0] = value if magnitude else np.power(ImgSize, 2)

    # Create meshgrid for x and y
    x, y = np.meshgrid(np.arange(-ImgSize/2, ImgSize/2), np.arange(-ImgSize/2, ImgSize/2))

    # Convert to polar coordinates and round to integers
    _, radius = np.arctan2(y, x), np.round(np.sqrt(x**2 + y**2))

    ## All Remaining Frequency Rotational Average Values
    for r in range(1, int(ImgSize/2)):
        rAvg[r] = np.mean(np.abs(image_freqs[radius == r])) if magnitude else np.mean(np.abs(image_freqs[radius == r])**2)
    
    return rAvg

def image_spatial_frequency(Image: np.ndarray):
    fft_image = fft.fftshift(fft.fft2(Image)) # compute fft2 & shift the zero-frequency component to the center of the spectrum. 
    return rotational_average_power_spectrum(fft_image, magnitude=False)

# ************** Aggregate fourier magnitude **************

def image_aggregate_fourier_magnitude(image:np.ndarray, f1:Optional[int] = None):
    grayscale_image = np.mean(image[:,:,:3], axis=2)
    If = fft.fftshift(fft.fft2(grayscale_image))
    return rotational_average_power_spectrum(If, magnitude=True) if not f1 else np.sum(rotational_average_power_spectrum(If, magnitude=True)[f1:])

# ************** Weibull parameters **************

def estimate_weibull_parameters(image:np.ndarray, sigma:int, max_percent:float = 0.95):
    gamma_opt, beta_opt, area_error, bad_fit = 1, 0, 0, True
    height, width = image.shape[:2]
    Y, X = np.ogrid[:height, :width]
    center = (int(width/2), int(height/2))
    radius = max_percent * center[0]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    try:
        edges = gaussian_gradient_magnitude(image, sigma=sigma) # compute edges
        edges = edges[mask != 0] # mask the outer circle of the edges/contrasts map which are identically zero
        y, x  = np.histogram(edges, bins=256)  # compute density of contrasts
        dx = x[1:]-x[:-1]

        try:
            gamma_opt, mu, beta_opt = weibull_min.fit(edges) # Careful : Do not force floc=np.min(edges) otherwise it will fail to fit !
            fitted_weibull = weibull_min(c=gamma_opt, loc=mu, scale=beta_opt)
            area_error = np.sum(np.abs(y - fitted_weibull.pdf(x[1:]))*dx)
        except ValueError as e: print(f"ValueError: {e}")
    except Exception as e: print(f"Exception: {e}")

    if gamma_opt != 1:
        bad_fit = False
    return gamma_opt, beta_opt, area_error, bad_fit

def image_weibull_parameters(image:np.ndarray, sigma:int = 12):
    image = np.mean(image[:,:,:3], axis=2) # convert to gray scale : (height, width, depth)
    return estimate_weibull_parameters(image, sigma)[:2]
