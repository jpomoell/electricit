# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Functions for various processing of field data
"""

import numpy as np

import scipy.interpolate
import scipy.ndimage

import astropy.units as u


def interpolate_and_smooth(data, source_coordinates, target_coordinates, gaussian_radius, sigma=1.0):

    xs, ys = source_coordinates

    xt, yt = target_coordinates

    # Create the interpolant
    interp = scipy.interpolate.RectBivariateSpline(ys[:, 0], xs[0, :], data, kx=1, ky=1)

    # interpolate
    interpolated_data \
        = interp(yt.ravel(), xt.ravel(), grid=False).reshape(xt.shape)
    
    # Apply smoothing using a Gaussian filter
    smoothed_interpolated_data \
        = scipy.ndimage.gaussian_filter(interpolated_data, sigma=sigma, truncate=gaussian_radius)
    
    return smoothed_interpolated_data


def disambiguate_field(transverse_B_field_CCD, disambiguation_info, weak_field_disambiguation_method):
    """Applies disambiguation of the magnetic field.
    
    Args:
        transverse_B_field_CCD: Sequence of 2D numpy.ndarrays containing the transverse 
            components B_ksi and B_eta of the magnetic field before the disambiguation.
        disambiguation_info: 2D numpy.ndarray that contains information about the disambiguation 
            of the azimuth in each pixel following the format used by the HMI data.
        weak_field_disambiguation_method: String providing the weak field disambiguation method 
            to apply. One of (\'potential_field_acute_angle\', \'random\', \'radial_acute_angle\')").
            Strong field pixels are always disambiguated using the minimum energy method.
            For details see (Hoeksema et al., 2014).

    Returns:
        B_ksi, B_eta: 2D numpy.ndarrays of the B_ksi and B_eta components of the 
            magnetic field after the disambiguation.
                     
    References:
        Hoeksema et al. (2014): https://doi.org/10.1007/s11207-014-0516-8
    """  
    
    B_ksi, B_eta = transverse_B_field_CCD
    
    # The "disambig_arr" segment contains information about the 
    # disambiguation result in the form of a three-bit code. 
    # Each bit encodes whether the azimuth of the B vector in CCD coordinates 
    # should be reversed (1 = yes, 0 = no), with each bit representing the result
    # from a different disambiguation method. See Hoeksema et al. (2014) for details.

    # Get the bit representing the weak field disamiguation method to use
    weak_field_disambiguation_bit \
        = {"potential_field_acute_angle": 0,
           "random": 1,
           "radial_acute_angle": 2
           }[weak_field_disambiguation_method.lower()]


    # Determine which pixels to flip by computing the truth value of
    #  data & 0b001 (returns 1 if bit 0 is set)
    #  data & 0b010 (returns 2 if bit 1 is set)
    #  data & 0b100 (returns 4 if bit 2 is set)    
    pixels_to_flip = np.nonzero(np.bitwise_and(disambiguation_info, 2**weak_field_disambiguation_bit))

    # Reverse transverse field direction
    B_ksi[pixels_to_flip] *= -1
    B_eta[pixels_to_flip] *= -1
    
    return B_ksi, B_eta


def scale_to_Mercator(data, coordinates, center):
    """Scales 2D data multiplying it by cos(lat')**2-term

    Scales the provided data array with a cos(lat')**2-term, where 
    lat' is the latitude with respect to the center of the cutout patch 
    in a non-standard heliographic coordinate system where the center point 
    is at (lon', lat') = (0, 0). 
    
    Args:
        data: 2D numpy.ndarray containing the data to be scaled.
        coordinates: SkyCoord object containing the Heliographic coordinates 
            of the input data.
        center: SkyCoord defining the Heliographic coordinates of the center 
            point of the input data to be scaled.
                      
    Returns:
        A copy of the input data scaled accordingly.    
    """
    
    # Copy of the input data array
    data = data.copy()

    # Heliographic coordinates
    lat = coordinates.lat.to(u.rad).value
    lon = coordinates.lon.to(u.rad).value

    # Center of patch
    lon_c, lat_c = center.lon.to(u.rad).value, center.lat.to(u.rad).value
    
    arg = np.sin(lat)*np.cos(lat_c) - np.cos(lat)*np.sin(lat_c)*np.cos(lon - lon_c)

    scale = np.cos(np.arcsin(arg).real)**2

    return data*scale


def unscale_from_Mercator(data, coordinates, center):
    """Unscales 2D data from Mercator dividing it by cos(lat')**2-term

    Removes cos(lat')**2 scaling from the data (created using the
    "scale_to_Mercator" function), where lat' is the latitude with respect to 
    the center of the cutout patch in a non-standard heliographic coordinate 
    system where the center point is always at (lon', lat') = (0, 0).
    
    Args:
        data: 2D numpy.ndarray containing the data to be scaled.
        coordinates: SkyCoord object containing the Heliographic coordinates 
            of the input data.
        center: SkyCoord defining the Heliographic coordinates of the center 
            point of the input data to be scaled.
                         
    Returns:
        A copy of the input data unscaled accordingly.        
    """
    
    # Copy of the input data array
    data = data.copy()
    
    # Heliographic coordinates
    lat = coordinates.lat.to(u.rad).value
    lon = coordinates.lon.to(u.rad).value

    # Center of patch
    lon_c, lat_c = center.lon.to(u.rad).value, center.lat.to(u.rad).value
    
    arg = np.sin(lat)*np.cos(lat_c) - np.cos(lat)*np.sin(lat_c)*np.cos(lon - lon_c)

    scale = np.cos(np.arcsin(arg).real)**2

    return data/scale

