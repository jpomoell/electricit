# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Functions for transforming between coordinate systems
"""


import numpy as np

import astropy.units as u
import astropy.coordinates


def projection_coordinates_to_heliographic(chart, projection):
    """Transforms projection coordinates to heliographic coordinates
    
    Transforms the coordinates of the patch provided by the chart in the given
    projection plane to corresponding heliographic coordinates.  

    Args:
        chart: A ChartCoordinates instance defining the coordinates to be transformed
        projection: A string ('CEA'/'Mercator'/'Plate Carree') defining the used projection
        
    Returns:
        astropy.SkyCoord containing the heliographic coordinates 
                
    References:
        Calabretta & Greisen (2002): http://dx.doi.org/10.1051/0004-6361:20021327
        Sun (2013): http://adsabs.harvard.edu/abs/2013arXiv1309.2392S                
    """

    # Make edge case an error
    if chart.lon.size < 2:
        raise ValueError("Size of chart in longitude must be larger than 1")

    # Get chart coordinates relative to the patch center in units of radians
    x, y = np.meshgrid(chart.x_relative_to_center.to(u.rad).value, 
                       chart.y_relative_to_center.to(u.rad).value)
    
    # Center of patch
    lon_c, lat_c = chart.center.lon.to(u.rad).value, chart.center.lat.to(u.rad).value
    
    cos_lat_c = np.cos(lat_c)
    sin_lat_c = np.sin(lat_c)

    if projection.lower() == 'cea':
        
        sqrt_y1 = np.sqrt(1.0 - y**2)

        lat = np.arcsin(cos_lat_c*y + sin_lat_c*sqrt_y1*np.cos(x))
        
        cos_lat = np.cos(lat)
        
        lon = np.arcsin(sqrt_y1*np.sin(x)/cos_lat) + lon_c        

    elif projection.lower() == 'mercator':
        
        phicom = 2.0*np.arctan(np.exp(y)) - np.pi/2.0
        
        sin_phi = np.sin(phicom)
        cos_phi = np.cos(phicom)
        
        lat = np.arcsin(sin_phi*cos_lat_c + cos_phi*np.cos(x)*sin_lat_c)
        
        cos_lat = np.cos(lat)
        
        lon = np.arcsin(cos_phi*np.sin(x)/cos_lat) + lon_c
                
    elif projection.lower() == 'plate carree':
        
        lat = np.arcsin(cos_lat_c*np.sin(y) + sin_lat_c*np.cos(y)*np.cos(x)).real
        
        cos_lat = np.cos(lat)
        
        lon = np.arcsin(np.cos(y)*np.sin(x)/cos_lat).real + lon_c
        
    else:
        raise ValueError("Unknown projection")

    # Correct for pole coordinate if present
    lon[np.where(cos_lat == 0.0)] = lon_c
    
    # Attach units
    lon *= u.rad
    lat *= u.rad

    return astropy.coordinates.SkyCoord(lon=lon, lat=lat,
                                        frame=chart.frame.name, 
                                        observer=chart.center.observer,
                                        obstime=chart.frame.obstime)


def heliographic_coordinates_to_CCD(coordinates, observer, p_angle, g_angle, RSpix):
    """Transform heliographic coordinates to CCD coordinates.

    Transforms the heliographic longitudes and latitudes to the CCD coordinates of 
    the camera observing the Sun. 
    
    Args:
        coordinates: astropy.SkyCoord instance giving the heliographic coordinates.
        observer: astropy.SkyCoord providing the coordinates of the observing telescope.
        p_angle: astropy.Quantity providing the p-angle of the telescope, i.e., the angle 
            between the y axis of the CCD coordinates of the telescope and the solar North
            (computed CCW from the solar north).
        g_angle: astropy.Quantity providing the half angular width of the solar disk as 
            seen from the telescope.
        RSpix: astropy.Quantity giving the radius of the solar disk in pixels.
    
    Returns:
        ksi, eta: 2D numpy.ndarrays containing the output CCD coordinates where 
            ksi corresponds to the x axis and eta the y axis. 
               
    References:
        Calabretta & Greisen (2002): http://dx.doi.org/10.1051/0004-6361:20021327
        Sun (2013): http://adsabs.harvard.edu/abs/2013arXiv1309.2392S
    """

    # Heliographic coordinates
    lat = coordinates.lat.to(u.rad).value
    lon = coordinates.lon.to(u.rad).value

    # Heliographic coordinates of observer
    obs = observer.transform_to(coordinates.frame)
    lat_c = obs.lat.to(u.rad).value
    lon_c = obs.lon.to(u.rad).value
    
    p = p_angle.to(u.rad).value
    g = g_angle.to(u.rad).value
    RS = RSpix.to(u.pix).value

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lat_c = np.cos(lat_c)
    sin_lat_c = np.sin(lat_c)
    sin_dlon = np.sin(lon-lon_c)
    cos_dlon = np.cos(lon-lon_c)
    sin_p = np.sin(p)
    cos_p = np.cos(p)

    # Perform the transformation to the CCD coordinates   
    hem_var = sin_lat*sin_lat_c + cos_lat_c*cos_lat*cos_dlon
    r = RS*np.cos(g)/(1.0 - hem_var*np.sin(g))
    ksi0 = r*cos_lat*sin_dlon
    eta0 = r*(sin_lat*cos_lat_c - cos_lat*sin_lat_c*cos_dlon)

    # Rotate by p angle
    ksi = ksi0*cos_p - eta0*sin_p
    eta = ksi0*sin_p + eta0*cos_p

    # Data points which would be behind the visible solar disk 
    # (the CCD coordinate is unspecified) are filled with NaNs
    if lon.size > 1:
        behind_the_sun = np.where(hem_var < 0)
        ksi[behind_the_sun] = np.NaN
        eta[behind_the_sun] = np.NaN#
    elif hem_var < 0:
        return np.NaN, np.NaN
        
    return ksi, eta


def obs_field_to_CCD(B_vector):
    """Transforms observed magnetic field components to the CCD basis.
    
    Args:
        B_vector: Sequence of three sunpy.Map instances containing the magnitude (B), 
            inclination (gamma) and azimuth (phi) of the magnetic field.

    Returns:
        B_ksi, B_eta, B_los: tuple of numpy.ndarrays containing the 
            three components of the magnetic field in the CCD basis.
    """

    B, gamma, phi = B_vector
        
    gamma_values \
        = gamma.data*astropy.units.Quantity(1.0, gamma.meta["bunit"]).to(u.rad).value

    phi_values \
        = phi.data*astropy.units.Quantity(1.0, phi.meta["bunit"]).to(u.rad).value

    B_ksi = -B.data*np.sin(gamma_values)*np.sin(phi_values)
    B_eta = B.data*np.sin(gamma_values)*np.cos(phi_values)
    B_los = B.data*np.cos(gamma_values)

    return B_ksi, B_eta, B_los


def CCD_vector_to_heliographic(field, coordinates, observer, p_angle):
    """Transforms the vector from the CCD basis to the heliographic basis.

    Rotates the vector field data from the CCD basis, V_field = [V_ksi, V_eta, V_los] 
    to the heliographic basis V_field = [Vr, Vt, Vp].

    Args:
        V_field: Sequence containing 2D numpy.ndarrays of the CCD V_ksi, V_eta and V_los 
            vector components, in this order.
        coordinates: astropy.SkyCoord containing the heliographic coordinates of the 
            vector field
        p_angle: astropy.Quantity giving the rotation angle of the camera with respect 
            to solar north. Positive p means rotation of camera clockwise (looking at 
            the Sun from behind the camera). p = 180 deg means that solar north points 
            down in images.

    Returns:
        [Vr, Vt, Vp] list of 2D numpy.ndarrays of the vector field components 
            in the heliographic basis.
    """
    
    # Components of the vector field
    V_ksi, V_eta, V_los = field
    
    # Heliographic coordinates of V
    lat = coordinates.lat.to(u.rad).value
    lon = coordinates.lon.to(u.rad).value

    # Heliographic coordinates of observer
    obs = observer.transform_to(coordinates.frame)
    lat_c = obs.lat.to(u.rad).value
    lon_c = obs.lon.to(u.rad).value
    
    p = p_angle.to(u.rad).value
    
    # Precompute sin and cos terms 
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    
    cos_lat_c = np.cos(lat_c)
    sin_lat_c = np.sin(lat_c)
    
    sin_dlon = np.sin(lon - lon_c)
    cos_dlon = np.cos(lon - lon_c)

    sin_p = np.sin(p)
    cos_p = np.cos(p)

    a11 =  cos_lat*(sin_lat_c*sin_p*cos_dlon + cos_p*sin_dlon) - sin_lat*cos_lat_c*sin_p
    a12 = -cos_lat*(sin_lat_c*cos_p*cos_dlon - sin_p*sin_dlon) + sin_lat*cos_lat_c*cos_p

    a21 =  sin_lat*(sin_lat_c*sin_p*cos_dlon + cos_p*sin_dlon) + cos_lat*cos_lat_c*sin_p
    a22 = -sin_lat*(sin_lat_c*cos_p*cos_dlon - sin_p*sin_dlon) - cos_lat*cos_lat_c*cos_p
    
    a13 = cos_lat*cos_lat_c*cos_dlon + sin_lat*sin_lat_c    
    a23 = cos_lat_c*sin_lat*cos_dlon - sin_lat_c*cos_lat
    
    a31 = -sin_lat_c*sin_p*sin_dlon + cos_p*cos_dlon
    a32 =  sin_lat_c*cos_p*sin_dlon + sin_p*cos_dlon
    a33 = -cos_lat_c*sin_dlon

    Vr = a11*V_ksi.data + a12*V_eta.data + a13*V_los.data
    Vt = a21*V_ksi.data + a22*V_eta.data + a23*V_los.data
    Vp = a31*V_ksi.data + a32*V_eta.data + a33*V_los.data

    return [Vr, Vt, Vp]


def heliographic_vector_to_patch_centered_frame(V_field, coordinates, center):
    """Transforms vector field from heliographic to patch-centered basis

    Rotates a vector field defined in the standard heliographic basis (V_field = 
    [Vr, Vt, Vp]) to a non-standard heliographic basis in which the center of the 
    cutout patch is at (lon, lat) = (0,0). This requires rotation of the V_field 
    around the r-axis.

    Args:
        V_field: Sequence of 2D numpy.ndarrays. The input vector field components 
            contains the heliographic Vr, Vt and Vp components in this order.
        coordinates: ChartCoordinates instance providing the heliographic coordinates.
        center: astropy.SkyCoord providing the coordinates of the center of the input 
            patch, over which the "V_field" is given, in standard heliographic coordinates.
    
    Returns:
        The vector field rotated to the patch-centered heliographic basis.
    """

    # Heliographic coordinates of V
    lat = coordinates.lat.to(u.rad).value
    lon = coordinates.lon.to(u.rad).value

    # Center of patch
    lon_c, lat_c = center.lon.to(u.rad).value, center.lat.to(u.rad).value

    # Field components    
    Vr, Vt, Vp = V_field

    # Compute the cosine of the latitude in patch-centered spherical coordinates
    sin_lat_centered = np.cos(lat_c)*np.sin(lat)-np.sin(lat_c)*np.cos(lat)*np.cos(lon - lon_c)
    cos_lat_centered = np.sqrt(1.0 - sin_lat_centered**2)
    
    # Rotation angle(s) about the r-axis
    cos_rot_ang \
        = (  np.sin(lat_c)*np.sin(lat)*np.cos(lon - lon_c) \
           + np.cos(lat_c)*np.cos(lat))/cos_lat_centered
    sgn_rot_ang = np.sign(lat_c)*np.sign(lon - lon_c)

    # Transformation matrix which rotates V_field around local r-axes by rot_ang
    a11 = cos_rot_ang
    a12 = -sgn_rot_ang*np.sqrt(1.0 -cos_rot_ang**2).real
    a21 = sgn_rot_ang*np.sqrt(1.0 - cos_rot_ang**2).real
    a22 = cos_rot_ang 

    Vt_rot = a11*Vt + a12*Vp
    Vp_rot = a21*Vt + a22*Vp

    return [Vr, Vt_rot, Vp_rot]
