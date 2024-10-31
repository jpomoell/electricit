# This file is part of ELECTRICIT.
#
# Copyright 2024 ELECTRICIT developers
#
# Use of this source code is governed by a BSD-style license 
# that can be found in the LICENSE.md file.

"""Chart class definitions and related utilities
"""

import numpy as np

from sunpy.map.mapbase import PixelPair, SpatialPair

import astropy
import astropy.units as u
import astropy.coordinates


class ChartCoordinates:
    
    def __init__(self, bottom_left, top_right, res, border_pixels=0*u.pix):
        
        # Check that the frames of the two coordinates are the same
        if not bottom_left.frame.is_equivalent_frame(top_right.frame):
            raise ValueError("The frames of the coorner coordinates must be equivalent")
                
        # Get extent in lon and lat
        lon_extent = np.abs(top_right.lon - bottom_left.lon)
        lat_extent = np.abs(top_right.lat - bottom_left.lat)
        
        # Store the constant resolution
        self.delta = res.to(u.deg/u.pix)
        
        # Frame of the chart
        self.frame = bottom_left.frame
        
        # Center of the chart
        self._center \
            = astropy.coordinates.SkyCoord(0.5*(bottom_left.lon + top_right.lon), 
                                           0.5*(bottom_left.lat + top_right.lat),
                                           observer=bottom_left.observer,
                                           frame=self.frame)
        
        # Construct chart coordinates
        # Internally, the coordinates are stored relative to the center (average
        # of the input coordinates
        self._x = self._coordinates(lon_extent.to(u.deg).value, self.delta.value, extend=2*border_pixels.value)*u.deg
        self._y = self._coordinates(lat_extent.to(u.deg).value, self.delta.value, extend=2*border_pixels.value)*u.deg
        
    @property
    def x(self):
        return self._x + self.center.lon
    
    @property
    def y(self):
        return self._y + self.center.lat
    
    @property
    def x_relative_to_center(self):
        return self._x

    @property
    def y_relative_to_center(self):
        return self._y

    @property
    def lon(self):
        return self.x

    @property
    def lat(self):
        return self.y

    @property
    def dimensions(self):
        return PixelPair(len(self.x)*u.pix, len(self.y)*u.pix)
   
    @property
    def scale(self):
        return SpatialPair(self.delta, self.delta)
    
    @property
    def extent(self):
        return SpatialPair(self.x[-1] - self.x[0] + u.pix*self.delta,
                           self.y[-1] - self.y[0] + u.pix*self.delta)

    @property
    def center(self):
        return self._center

    @property
    def coordinates_relative_to_center(self):
        return (self._x, self._y)
    
    def remove_border(self, pixels):

        # Remove pixels from borders
        selection = slice(int(pixels.to_value(u.pix)), -int(pixels.to_value(u.pix)))
        
        # Return new instance with removed pixels
        return self._create_from_coordinates(self.x[selection], self.y[selection])

    def create_coarse(self, coarsen_by):

        # The coordinate of the coarse chart are exactly copied from the source chart
        selection = slice(None, None, int(coarsen_by))
        
        return self._create_from_coordinates(self.x[selection], self.y[selection])
  
    def create_refined(self, refinement_factor, add_crse_border_pixels=0*u.pix, add_fine_border_pixels=0*u.pix):
        
        # Get edges of the source chart
        xedges = [self.x[0] - 0.5*u.pix*self.delta, self.x[-1] + 0.5*u.pix*self.delta]
        yedges = [self.y[0] - 0.5*u.pix*self.delta, self.y[-1] + 0.5*u.pix*self.delta]
        
        # Add border pixels with the coarse resolution before refinement
        xedges[0] -= add_crse_border_pixels*self.delta
        xedges[1] += add_crse_border_pixels*self.delta

        yedges[0] -= add_crse_border_pixels*self.delta
        yedges[1] += add_crse_border_pixels*self.delta

        # Return new instance
        return ChartCoordinates(astropy.coordinates.SkyCoord(xedges[0], yedges[0], frame=self.frame),
                                astropy.coordinates.SkyCoord(xedges[1], yedges[1], frame=self.frame),
                                self.delta/refinement_factor,
                                border_pixels=add_fine_border_pixels
                                )
         
    def _create_from_coordinates(self, x, y):

        # Check that coordinates are uniform
        dx = (x[-1] - x[0]).to(u.deg)/((len(x) - 1)*u.pix)
        dy = (y[-1] - y[0]).to(u.deg)/((len(y) - 1)*u.pix)
        
        x_center = 0.5*(x[0] + x[-1]).to_value(u.deg)
        y_center = 0.5*(y[0] + y[-1]).to_value(u.deg)

        chart = ChartCoordinates(astropy.coordinates.SkyCoord(x[ 0], y[ 0], frame=self.frame),
                                 astropy.coordinates.SkyCoord(x[-1], y[-1], frame=self.frame),
                                 dx)
        
        # Set coordinates
        chart._x = (np.copy(x.to_value(u.deg)) - x_center)*u.deg
        chart._y = (np.copy(y.to_value(u.deg)) - y_center)*u.deg
        
        return chart
    
    def _coordinates(self, extent, delta, extend=0, edge=False):
        
        # Number of pixels approximately covering the interval.
        # Rounding down consistently produces a region smaller (or equal)
        # to the requested size. The resulting actual extent is N*delta.        
        N = int(np.floor(extent/delta)) + extend

        # Return edge (edge=True) or center (edge=False) coordinates
        return (-0.5*N + 0.5*(1-edge) + np.arange(0.0, N+edge))*delta