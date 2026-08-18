/*
Header for Operators on data arrays for aPIC.

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

#ifndef OPERATORS_H_
#define OPERATORS_H_

#include <constants.h>

#include <AMReX_REAL.H>
#include <AMReX_MultiFab.H>

#include <cmath>

/*
  Face and edge data must be stored in seperate MultiFabs per face or edge direction
  Thus the standard in aPIC is a std::array of three MultiFabs
  However on occasion we may have each direction MultiFab stored separately
  Repackaging these as a new array is a waste of cpu cycles and memory
  Thus each operator which takes face or edge data as an input (potentially mutable)
  is overloaded to accept the inputs in either format
  Return values are newly allocated and hence can be always returned as arrays
  (c++ does not allow multiple object returns anyway)
  
  Each operator is also overloaded with void operators that return in the inputs
  and return operators which return the (newly allocated) data as an output of the function
  Return operators take an extra nghost input which determines the number of ghost cells in the return
  Since void operators already have the output MultiFab(s) this is already given
  
  Thus each operator may have as many as six overloads
*/

// Interpolators
// cell_data should have the same number of components as node_data
void node2cell(amrex::MultiFab& cell_data, const amrex::MultiFab& node_data);

inline amrex::MultiFab node2cell(const amrex::MultiFab& node_data, int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = node_data.n_grow;
   }
   
   int nvar = node_data.nComp();
   
   amrex::MultiFab cell_data(convert(node_data.boxArray(),AMReXConst::btype_c),node_data.distributionMap,nvar,vect_nghost);

   cell_data.setVal(0.0);
   
   node2cell(cell_data, node_data);

   return cell_data;
}

void face2cell(amrex::MultiFab& cell_data, const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data);

inline void face2cell(amrex::MultiFab& cell_data, const std::array<amrex::MultiFab,3>& face_data) {
   face2cell(cell_data, face_data[0], face_data[1], face_data[2]);
}

inline amrex::MultiFab face2cell(const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = xface_data.n_grow;
   }
   
   amrex::MultiFab cell_data(convert(xface_data.boxArray(),AMReXConst::btype_c),xface_data.distributionMap,3,vect_nghost);

   cell_data.setVal(0.0);
   
   face2cell(cell_data, xface_data, yface_data, zface_data);

   return cell_data;
}

inline amrex::MultiFab face2cell(const std::array<amrex::MultiFab,3>& face_data, const int nghost = -1) {
   return face2cell(face_data[0], face_data[1], face_data[2], nghost);
}

void node2edge(amrex::MultiFab& xedge_data, amrex::MultiFab& yedge_data, amrex::MultiFab& zedge_data, const amrex::MultiFab& node_data);

inline void node2edge(std::array<amrex::MultiFab,3>& edge_data, const amrex::MultiFab& node_data) {
   node2edge(edge_data[0], edge_data[1], edge_data[2], node_data);
}

inline std::array<amrex::MultiFab,3> node2edge(const amrex::MultiFab& node_data, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = node_data.n_grow;
   }
   
   std::array<amrex::MultiFab,3> edge_data = {
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ex),node_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ey),node_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ez),node_data.distributionMap, 1, vect_nghost)
   };

   for (int nn=0; nn<3; ++nn) {
      edge_data[nn].setVal(0.0);
   }

   node2edge(edge_data, node_data);

   return edge_data;
}

amrex::Real node2r_scalar(const amrex::Array4<const amrex::Real>& node_array, const amrex::Real xpos, const amrex::Real ypos, const amrex::Real zpos, const amrex::IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min);
std::array<amrex::Real,3> node2r_vector(const amrex::Array4<const amrex::Real>& node_array, const amrex::Real xpos, const amrex::Real ypos, const amrex::Real zpos, const amrex::IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min);
std::array<amrex::Real,3> face2r(const amrex::Array4<const amrex::Real>& xface_array, const amrex::Array4<const amrex::Real>& yface_array, const amrex::Array4<const amrex::Real>& zface_array, const amrex::Real xpos, const amrex::Real ypos, const amrex::Real zpos, const amrex::IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min);

// Diagnostics
// Energy_c should be cell-centred with 3 components
void compute_EM_energy(amrex::MultiFab& Energy_c, const amrex::MultiFab& B_c, const amrex::MultiFab& E_c);

inline amrex::MultiFab compute_EM_energy(const amrex::MultiFab& B_c, const amrex::MultiFab& E_c, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = B_c.n_grow;
   }
   
   amrex::MultiFab Energy_c = amrex::MultiFab(B_c.boxArray(), B_c.distributionMap, 3, vect_nghost);

   Energy_c.setVal(0.0);
   
   compute_EM_energy(Energy_c, B_c, E_c);

   return Energy_c;
}

// Derivative operators
void curl_e2f(amrex::MultiFab& xface_curl, amrex::MultiFab& yface_curl, amrex::MultiFab& zface_curl, const amrex::MultiFab& xedge_data, const amrex::MultiFab& yedge_data, const amrex::MultiFab& zedge_data, const amrex::GpuArray<amrex::Real,3>& dx);

inline void curl_e2f(std::array<amrex::MultiFab,3>& face_curl, const amrex::MultiFab& xedge_data, const amrex::MultiFab& yedge_data, const amrex::MultiFab& zedge_data, const amrex::GpuArray<amrex::Real,3>& dx) {
   curl_e2f(face_curl[0], face_curl[1], face_curl[2], xedge_data, yedge_data, zedge_data, dx);
}

inline void curl_e2f(amrex::MultiFab& xface_curl, amrex::MultiFab& yface_curl, amrex::MultiFab& zface_curl, const std::array<amrex::MultiFab,3>& edge_data, const amrex::GpuArray<amrex::Real,3>& dx) {
   curl_e2f(xface_curl, yface_curl, zface_curl, edge_data[0], edge_data[1], edge_data[2], dx);
}

inline void curl_e2f(std::array<amrex::MultiFab,3>& face_curl, const std::array<amrex::MultiFab,3>& edge_data, const amrex::GpuArray<amrex::Real,3>& dx) {
   curl_e2f(face_curl[0], face_curl[1], face_curl[2], edge_data[0], edge_data[1], edge_data[2], dx);
}

inline std::array<amrex::MultiFab,3> curl_e2f(const amrex::MultiFab& xedge_data, const amrex::MultiFab& yedge_data, const amrex::MultiFab& zedge_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = xedge_data.n_grow;
   }
   
   std::array<amrex::MultiFab,3> face_curl = {
      amrex::MultiFab(convert(xedge_data.boxArray(),AMReXConst::btype_fx),xedge_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(xedge_data.boxArray(),AMReXConst::btype_fy),xedge_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(xedge_data.boxArray(),AMReXConst::btype_fz),xedge_data.distributionMap, 1, vect_nghost)
   };

   for (int nn=0; nn<3; ++nn) {
      face_curl[nn].setVal(0.0);
   }
   
   curl_e2f(face_curl, xedge_data, yedge_data, zedge_data, dx);
   
   return face_curl;
}

inline std::array<amrex::MultiFab,3> curl_e2f(const std::array<amrex::MultiFab,3>& edge_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   return curl_e2f(edge_data[0], edge_data[1], edge_data[2], dx, nghost);
}

void curl_n2f(amrex::MultiFab& xface_curl, amrex::MultiFab& yface_curl, amrex::MultiFab& zface_curl, const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, amrex::Real fact = 1);

inline void curl_n2f(std::array<amrex::MultiFab,3>& face_curl, const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, amrex::Real fact = 1) {
   curl_n2f(face_curl[0], face_curl[1], face_curl[2], node_data, dx, fact);
}

inline std::array<amrex::MultiFab,3> curl_n2f(const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = node_data.n_grow;
   }
   
   std::array<amrex::MultiFab,3> face_curl = {
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_fx),node_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_fy),node_data.distributionMap, 1, vect_nghost),
      amrex::MultiFab(convert(node_data.boxArray(),AMReXConst::btype_fz),node_data.distributionMap, 1, vect_nghost)
   };

   for (int nn=0; nn<3; ++nn) {
      face_curl[nn].setVal(0.0);
   }
   
   curl_n2f(face_curl, node_data, dx);

   return face_curl;
}

void curl_f2n(amrex::MultiFab& node_curl, const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx, amrex::Real fact = 1);

inline void curl_f2n(amrex::MultiFab& node_curl, const std::array<amrex::MultiFab,3>& face_data, const amrex::GpuArray<amrex::Real,3>& dx, amrex::Real fact = 1) {
   curl_f2n(node_curl, face_data[0], face_data[1], face_data[2], dx, fact);
}

inline amrex::MultiFab curl_f2n(const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = xface_data.n_grow;
   }
   
   amrex::MultiFab node_curl(convert(xface_data.boxArray(),AMReXConst::btype_n),xface_data.distributionMap, 3, vect_nghost);

   node_curl.setVal(0.0);
   
   curl_f2n(node_curl, xface_data, yface_data, zface_data, dx);

   return node_curl;
}

inline amrex::MultiFab curl_f2n(const std::array<amrex::MultiFab,3>& face_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   return curl_f2n(face_data[0], face_data[1], face_data[2], dx, nghost);
}

void div_f2c(amrex::MultiFab& cell_div, const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx);

inline void div_f2c(amrex::MultiFab& cell_div, const std::array<amrex::MultiFab,3>& face_data, const amrex::GpuArray<amrex::Real,3>& dx) {
   div_f2c(cell_div, face_data[0], face_data[1], face_data[2], dx);
}

inline amrex::MultiFab div_f2c(const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = xface_data.n_grow;
   }
   
   amrex::MultiFab cell_div(convert(xface_data.boxArray(),AMReXConst::btype_c),xface_data.distributionMap, 1, vect_nghost);

   cell_div.setVal(0.0);
   
   div_f2c(cell_div, xface_data, yface_data, zface_data, dx);

   return cell_div;
}

inline amrex::MultiFab div_f2c(const std::array<amrex::MultiFab,3>& face_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   return div_f2c(face_data[0], face_data[1], face_data[2], dx, nghost);
}

void div_n2c(amrex::MultiFab& cell_div, const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx);

inline amrex::MultiFab div_n2c(const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, const int nghost = -1) {
   // If number of ghost cells is not given (thus set to -1), copy from input MultiFab
   amrex::IntVect vect_nghost(nghost,nghost,nghost);
   if (nghost == -1) {
      vect_nghost = node_data.n_grow;
   }
   
   amrex::MultiFab cell_div(convert(node_data.boxArray(),AMReXConst::btype_c),node_data.distributionMap, 1, vect_nghost);

   cell_div.setVal(0.0);
   
   div_n2c(cell_div, node_data, dx);

   return cell_div;
}

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

// Get (global) cell/face/edge/node indices from true domain position
inline amrex::IntVect get_pos_indices(amrex::Real xpos, amrex::Real ypos, amrex::Real zpos, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min, const amrex::IntVect& index_type = amrex::IntVect::TheZeroVector()) {
   amrex::IntVect ret = {
      int(floor((xpos - dom_min[0])/dx[0] + index_type[0]/2)),
      int(floor((ypos - dom_min[1])/dx[1] + index_type[1]/2)),
      int(floor((zpos - dom_min[2])/dx[2] + index_type[2]/2))
   };
   return ret;
}

// Particle weight of nearest lower neighbour at given location for node-type indexing
inline std::array<amrex::Real,2> CIC_weights_1D(const amrex::Real x, const amrex::Real dx, const int x_ind, const amrex::Real xmin) {
   amrex::Real w = (x - xmin)/dx - x_ind;
   return {1-w, w};
}

// Check symmetry in given direction (i.e. check if constant along given direction(s)
bool sym_test(const amrex::MultiFab& mf, const amrex::IntVect& dir);

#endif
