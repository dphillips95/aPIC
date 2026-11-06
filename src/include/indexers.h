/*
-- INSERT FILE DESCRIPTION --

Copyright 2026 Finnish Meteorological Institute.

This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with this program. If not, see
<https://www.gnu.org/licenses/>.


Author(s): David Phillips
*/

#ifndef INDEXERS_H_
#define INDEXERS_H_

#include <AMReX_REAL.H>

// Indexers
// Get cell (or node etc.) ID from given cell indices of cell in box
// If the box has ghost cells then index needs to be shifted to accomodate
inline int get_cellID(int x_index, int y_index, int z_index, const amrex::IntVect& len, int nghost = 0) {
   x_index += nghost;
   y_index += nghost;
   z_index += nghost;
   return (z_index*len[1] + y_index)*len[0] + x_index;
}

inline int get_cellID(amrex::IntVect cell_indices, const amrex::IntVect& len, int nghost = 0) {
   cell_indices += nghost;
   return get_cellID(cell_indices[0], cell_indices[1], cell_indices[2], len);
}

template <size_t num>
inline std::array<int,num> get_cellID(const std::array<amrex::IntVect,num>& cell_indices, const amrex::IntVect& len, int nghost = 0) {
   std::array<int,num> ret;
   for (size_t ii=0; ii<num; ++ii) {
      ret[ii] = get_cellID(cell_indices[0], len, nghost);
   }
   return ret;
}

// Get cell indices from given cellID
inline amrex::IntVect get_cell_indices(int cellID, const amrex::IntVect& len, int nghost = 0) {
   int
      x_index = cellID%len[0]          - nghost,
      y_index = (cellID/len[0])%len[1] - nghost,
      z_index = cellID/(len[0]*len[1]) - nghost;
   return amrex::IntVect(x_index,y_index,z_index);
}

// Get box index of closest node/cell at position > x
// Alternatively: Get index of cell/staggered dual cell containing position
inline int get_pos_index(const amrex::Real x, const amrex::Real dx, const amrex::Real dom_min, const int index_type) {
   return static_cast<int>(floor((x - dom_min)/dx - 0.5*(1 - index_type)));
 }

// Particle weight for single dimension at given position
// Weight is 1 - normalised distance to point from x_ind position
// dom_min is location of lower edge of box
inline std::array<amrex::Real,2> CIC_weights_1D(const amrex::Real x, const amrex::Real dx, const int x_ind, const amrex::Real dom_min, const int index_type) {
   amrex::Real w = (x - dom_min)/dx - 0.5*(1 - index_type) - x_ind;
   return {1-w, w};
}

#endif
