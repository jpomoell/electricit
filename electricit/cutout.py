# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Cutout creation functions
"""

import numpy as np
import sunpy.coordinates
import sunpy.map

import astropy.coordinates
import astropy.units as u

import electricit.chart
import electricit.transform
import electricit.process


def create_vector_cutout(bottom_left, 
                         top_right, 
                         time, 
                         resolution, 
                         dataset, 
                         projection, 
                         output_basis, 
                         weak_field_disambiguation_method="potential_field_acute_angle",
                         data_loader=None):

    #
    # Check inputs
    #
    if output_basis.lower() not in ("hg", "local cartesian"):
        raise ValueError("Argument \"output_basis\" must be one of (\'HG\', \'Local Cartesian\')")
    
    if projection.lower() not in ("cea", "plate carree", "mercator"):
        raise ValueError("Argument \"projection\" must be one of (\'CEA\', \'Plate Carree\', \'Mercator\')")
    
    if weak_field_disambiguation_method not in ("potential_field_acute_angle", "random", "radial_acute_angle"):
        raise ValueError("The weak field disambiguation method must be one of \
                         (\'potential_field_acute_angle\', \'random\', \'radial_acute_angle\')")

    #
    # Load the vector data
    #
    
    # If no loader function is defined, assume the file can directly be read by sunpy
    if data_loader is None:
        data_loader = lambda f : sunpy.map.Map(f)

    # Load the source vector data
    field = data_loader(dataset.create_filestring(time, "field"))
    azimuth = data_loader(dataset.create_filestring(time, "azimuth"))
    inclination = data_loader(dataset.create_filestring(time, "inclination"))
    disambig = data_loader(dataset.create_filestring(time, "disambig"))

    #
    # Transform target chart coordinates to CCD
    #

    # Create a chart defining the coordinates used in extracting the data from 
    # the observed datasets. The chart is oversampled by a factor of 3 to increase fidelity 
    # in the interpolation. In addition, an extra layer of pixels is added to minimize edge-effects.
    oversampling_ratio = 3.0
    gaussian_radius = 2.0

    wchart = electricit.chart.ChartCoordinates(bottom_left,
                                               top_right,
                                               res=resolution/oversampling_ratio,
                                               border_pixels=1*u.pix)
    
    # Create the chart defining the target/final coordinates of the cutout data
    chart = wchart.remove_border(1*u.pix).create_coarse(coarsen_by=oversampling_ratio)


    # Map projection coordinates to heliographic coordinates
    heliographic_coordinates \
        = electricit.transform.projection_coordinates_to_heliographic(wchart, projection)

    # Same for the target coordinates
    target_heliographic_coordinates \
        = electricit.transform.projection_coordinates_to_heliographic(chart, projection)

    # To convert heliographic coordinate to CCD coordinates, need info on the observer
    # NOTE: for other observers, this step needs to be generalized
    observer = astropy.coordinates.SkyCoord(field.observer_coordinate.data, 
                                            frame=field.observer_coordinate.frame,
                                            observer="earth") 

    # Radius of the Sun in pixels
    r_pix = field.rsun_obs/field.scale.axis1

    # Fixed angular width of 16'1'' for the Sun at 1 AU
    ang_sun = 16.0*u.arcmin + 1.0*u.arcsec

    # Rotation of the solar north (p-angle)
    p_angle = -field.meta['crota2']*u.deg

    # Disk center in CCD pixel coordinates
    x0_pix, y0_pix = field.meta['crpix1'] - 1, field.meta['crpix2'] - 1

    # Convert heliographic coordinates to the instrument-native CCD coordinates
    ksi, eta = electricit.transform.heliographic_coordinates_to_CCD(heliographic_coordinates, observer, p_angle, ang_sun, r_pix)

    # Since the transformation to CCD assumes the disk center to be at (0, 0), 
    # the location of the disk center in pixel coordinates needs to be added
    ksi += x0_pix
    eta += y0_pix

    #
    # Create a local cutout of the data
    #

    # Define the window of the local cutout
    submap_definition = CCD_submap_definition(ksi, eta)

    # Get the cutout data
    cutout_field = field.submap(**submap_definition)
    cutout_azimuth = azimuth.submap(**submap_definition)
    cutout_inclination = inclination.submap(**submap_definition)
    cutout_disambig = disambig.submap(**submap_definition)

    # Coordinate of the submap
    x_CCD_min = min(submap_definition["bottom_left"][0].value, submap_definition["top_right"][0].value)
    y_CCD_min = min(submap_definition["bottom_left"][1].value, submap_definition["top_right"][1].value)

    X_CCD, Y_CCD \
        = np.meshgrid(np.floor(np.min(x_CCD_min)) + np.arange(cutout_field.data.shape[1]), 
                      np.floor(np.min(y_CCD_min)) + np.arange(cutout_field.data.shape[0]))
    
    #
    # Magnetic field vector in the CCD coordinate system
    #
    B_ksi, B_eta, B_los \
        = electricit.transform.obs_field_to_CCD([cutout_field, cutout_inclination, cutout_azimuth])

    # Perform disambiguation of the field
    B_ksi, B_eta \
        = electricit.process.disambiguate_field([B_ksi, B_eta],
                                                cutout_disambig.data,
                                                weak_field_disambiguation_method)

    #
    # Remove "bad" pixels
    #
    B_field = [B_ksi, B_eta, B_los]

    # Set non-finite values to zero
    for B in B_field:
        B[np.where(np.isfinite(B) == False)] = 0.0

    #
    # Interpolate magnetic field vector to the coordinates of the target chart 
    #
    for component in (0, 1, 2):

        # Interpolation is done via oversampling and smoothing
        Bcomponent \
            = electricit.process.interpolate_and_smooth(B_field[component], 
                                                        (X_CCD, Y_CCD), 
                                                        (ksi, eta), 
                                                        gaussian_radius)

        # Get the field values of the target chart by directly selecting
        # the corresponding values from the oversampled chart
        selection = slice(int(gaussian_radius), None, int(oversampling_ratio))

        # NOTE: Re-purpose the list entries to contain the field values at the 
        # coordinates of the target chart
        B_field[component] = Bcomponent[selection, selection]

    
    #
    # Transform vector components from CCD to the heliographic system
    #
    B_field_heliographic \
        = electricit.transform.CCD_vector_to_heliographic(B_field,
                                                          target_heliographic_coordinates,
                                                          observer,
                                                          p_angle)

    # Mercator projection requires additional scaling of the field
    if projection.lower() == "mercator":
        B_field_heliographic[0] \
            = electricit.process.scale_to_Mercator(B_field_heliographic[0], 
                                                   target_heliographic_coordinates, 
                                                   chart.center)


    # Output vectors should be in a local Cartesian (x, y, z) basis
    if output_basis.lower() == "local cartesian":

        [Br, Bt, Bp] \
            = electricit.transform.heliographic_vector_to_patch_centered_frame(B_field_heliographic,
                                                                               target_heliographic_coordinates,
                                                                               chart.center)
        # Transform to local cartesian basis
        output_B_field = {"Bx": Bp, "By": -Bt, "Bz": Br}

    # Output vectors should be in heliographic (r, theta, phi) basis
    elif output_basis.lower() == "hg":
        
        # HG result is already available
        output_B_field = {"Br": B_field_heliographic[0], 
                          "Bt": B_field_heliographic[1], 
                          "Bp": B_field_heliographic[2]}

    #
    # Create the final maps
    #
    meta = create_cutout_metadata(field.meta, B_field[0].shape, chart.center, resolution, projection, observer)

    B_field_maps = list()
    for key, value in output_B_field.items():
        meta['segment'] = key
        
        B_field_maps.append(sunpy.map.Map(value, meta))
        

    return B_field_maps


def CCD_submap_definition(x, y):
    """Defines the submap for the CCD cutout for use with the sunpy.map.Map.submap function
    
    Args:
        x: 2D numpy.ndarray containing the CCD x (ksi) - coordinates of the cutout 
        y: 2D numpy.ndarray containing the CCD y (eta) - coordinates of the cutout 
        
    Returns:
        dictionary containing the submap definition
    """

    # Get the limits of the x and y coordinates
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    # Expand the limits of the cutout to ensure that there is sufficient data to 
    # avoid any extrapolation when interpolating from the CCD cutout to the final
    # grid. 1 extra pixel is added to each side, and +2 for the upper limit, 
    # since the sunpy.map.Map.submap function rounds the input CCD limits down 
    # before taking the cutout, and furthermore, does not include the pixels at 
    # the upper limits 
    x_CCD_min, x_CCD_max = x_min - 1, x_max + 2
    y_CCD_min, y_CCD_max = y_min - 1, y_max + 2

    # Return the information in the format expected by the submap() function
    submap_def \
        = {"bottom_left" : [x_CCD_min, y_CCD_min]*u.pixel,
           "top_right" : [x_CCD_max,y_CCD_max]*u.pixel}

    return submap_def


def create_cutout_metadata(ref_meta, shape, center, resolution, projection, observer):
    """Creates the necessary meta data for the cutout
    
    Creates the meta data required by the FITS standard and sunpy.map.Map objects to
    function correctly.

    Args:
        ref_meta: sunpy.map.header.MapMeta object. Provides some information that
            is copied to the cutout metadata.
        shape: 2-element tuple of integers giving the shape of the output map data
            in pixels (rows, cols).
        center: SkyCoord defining the center of the cutout patch.
        projection: String defining the map projection
        observer: SkyCoord providing the coordinate of the observing telescope.
                                        
    Returns:
        A dictionary containing the meta data
    """

    # Create a new metadata dictionary from scratch
    meta = dict()
    
    # Copy some items from the reference metadata
    for key in ('wavelnth', 'telescop', 'waveunit'):
        if key in ref_meta:
            meta[key] = ref_meta[key]

    meta['wcsaxes'] = 2
    meta['date-obs'] = ref_meta['t_obs']

    # Set observer info 
    obs = observer.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst(obstime=observer.obstime))
    meta['hgln_obs'] = obs.lon.to_value(u.deg)
    meta['hglt_obs'] = obs.lat.to_value(u.deg)
    meta['dsun_obs'] = obs.radius.to_value(u.m)
    meta['rsun_ref'] = obs.rsun.to_value(u.m)
    meta['rsun_obs'] = sunpy.coordinates.sun._angular_radius(obs.rsun, obs.radius).to_value(u.arcsec)

    [M, N] = shape

    meta['crpix1'] = (N + 1.0)/2.0
    meta['crpix2'] = (M + 1.0)/2.0
    meta['crval1'] = center.lon.to_value(u.deg)
    meta['crval2'] = center.lat.to_value(u.deg)
    meta['naxis'] = 2
    meta['naxis1'] = N
    meta['naxis2'] = M

    if isinstance(center.frame, sunpy.coordinates.HeliographicCarrington):
        meta['ctype1'] = 'CRLN_' + projection
        meta['ctype2'] = 'CRLT_' + projection
    elif isinstance(center.frame, sunpy.coordinates.HeliographicStonyhurst):
        meta['ctype1'] = 'HGLN_' + projection
        meta['ctype2'] = 'HGLT_' + projection

    meta['cdelt1'] = resolution.to_value(u.deg/u.pix)
    meta['cdelt2'] = resolution.to_value(u.deg/u.pix)
    meta['cunit1'] = 'deg'
    meta['cunit2'] = 'deg'

    return meta
