/*
Operators on data arrays for aPIC.

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

#include <operators.h>
#include <constants.h>
#include <math_functions.h>

#include <AMReX_REAL.H>
#include <AMReX_MultiFab.H>

#include <AMReX_Print.H>

#include <vector>

using namespace amrex;

// Interpolators: Node to Cell - Value in cell centre is calculated as average of value at all 8 nodes
void node2cell(MultiFab& cell_data, const MultiFab& node_data) {
   int nvar = node_data.nComp();
   
   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.validbox();
      const Array4<const Real>& n_array = node_data.const_array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx_c, nvar, [&](int ii, int jj, int kk, int nn) {
         c_array(ii,jj,kk,nn) += n_array(ii,  jj,  kk,  nn);
         c_array(ii,jj,kk,nn) += n_array(ii,  jj,  kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii,  jj+1,kk,  nn);
         c_array(ii,jj,kk,nn) += n_array(ii,  jj+1,kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj,  kk,  nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj,  kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj+1,kk,  nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj+1,kk+1,nn);
         c_array(ii,jj,kk,nn) /= 8;
      });
   }
}

// Interpolators: Face to Cell - Each face stores single component of vector perpendicular to face; cell centre data is calculated as average in each component of the two faces of the cell containing the given component
void face2cell(MultiFab& cell_data, const MultiFab& xface_data, const MultiFab& yface_data, const MultiFab& zface_data) {
   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.validbox();
      const Array4<const Real>&
         f_array_x = xface_data.const_array(mfi),
         f_array_y = yface_data.const_array(mfi),
         f_array_z = zface_data.const_array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         c_array(ii,jj,kk,0) += (f_array_x(ii,  jj,  kk  ) + f_array_x(ii+1,jj,  kk  ))/2;
         c_array(ii,jj,kk,1) += (f_array_y(ii,  jj,  kk  ) + f_array_y(ii,  jj+1,kk  ))/2;
         c_array(ii,jj,kk,2) += (f_array_z(ii,  jj,  kk  ) + f_array_z(ii,  jj,  kk+1))/2;
      });
   }
}

// Interpolators: Node to Edge - Edge data stores single component of vector data parallel to edge; edge data is calculated as average of given component from nodes at either end of edge
void node2edge(MultiFab& xedge_data, MultiFab& yedge_data, MultiFab& zedge_data, const MultiFab& node_data) {
   for (MFIter mfi(xedge_data); mfi.isValid(); ++mfi) {
      const Box&
         bx_ex = mfi.tilebox(AMReXConst::btype_ex),
         bx_ey = mfi.tilebox(AMReXConst::btype_ey),
         bx_ez = mfi.tilebox(AMReXConst::btype_ez);
      const Array4<const Real>& n_array = node_data.const_array(mfi);
      const Array4<Real>&
         e_array_x = xedge_data.array(mfi),
         e_array_y = yedge_data.array(mfi),
         e_array_z = zedge_data.array(mfi);
      
      ParallelFor(bx_ex, [&](int ii, int jj, int kk) {
         e_array_x(ii,jj,kk) += (n_array(ii,  jj,  kk,  0) + n_array(ii+1,jj,  kk,  0))/2;
      });
      
      ParallelFor(bx_ey, [&](int ii, int jj, int kk) {
         e_array_y(ii,jj,kk) += (n_array(ii,  jj,  kk,  1) + n_array(ii,  jj+1,kk,  1))/2;
      });
      
      ParallelFor(bx_ez, [&](int ii, int jj, int kk) {
         e_array_z(ii,jj,kk) += (n_array(ii,  jj,  kk,  2) + n_array(ii,  jj,  kk+1,2))/2;
      });
   }
}

// Interpolate scalar node data to position r; input is Array4 not MultiFab
// so position must be inside box real bounds
Real node2r_scalar(const Array4<const Real>& node_array, const Real xpos, const Real ypos, const Real zpos, const IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min) {
   // Cell indices (i,j,k) containing given position
   // Surrounding nodes are at x = i, i+1; y = j, j+1; z = k, k+1
   
   std::array<Real,2>
      x_weight = CIC_weights_1D(xpos, dx[0], cell_indices[0], dom_min[0]),
      y_weight = CIC_weights_1D(ypos, dx[1], cell_indices[1], dom_min[1]),
      z_weight = CIC_weights_1D(zpos, dx[2], cell_indices[2], dom_min[2]);

   Real ret = 0;
   for (int ii=0; ii<2; ++ii) {
      for (int jj=0; jj<2; ++jj) {
         for (int kk=0; kk<2; ++kk) {
            ret += node_array(cell_indices[0]+ii, cell_indices[1]+jj, cell_indices[2]+kk)*x_weight[ii]*y_weight[jj]*z_weight[kk];
         }
      }
   }

   return ret;
}

// Interpolate vector node data to position r; input is Array4 not MultiFab
// so position must be inside box real bounds
std::array<Real,3> node2r_vector(const Array4<const Real>& node_array, const Real xpos, const Real ypos, const Real zpos, const IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min) {
   // Cell indices (i,j,k) containing given position
   // Surrounding nodes are at x = i, i+1; y = j, j+1; z = k, k+1
   
   std::array<Real,2>
      x_weight = CIC_weights_1D(xpos, dx[0], cell_indices[0], dom_min[0]),
      y_weight = CIC_weights_1D(ypos, dx[1], cell_indices[1], dom_min[1]),
      z_weight = CIC_weights_1D(zpos, dx[2], cell_indices[2], dom_min[2]);

   std::array<Real,3> ret = {0,0,0};
   for (int ii=0; ii<2; ++ii) {
      for (int jj=0; jj<2; ++jj) {
         for (int kk=0; kk<2; ++kk) {
            for (int nn=0; nn<3; ++nn) {
               ret[nn] += node_array(cell_indices[0]+ii, cell_indices[1]+jj, cell_indices[2]+kk, nn)*x_weight[ii]*y_weight[jj]*z_weight[kk];
            }
         }
      }
   }
   
   return ret;
}

// Interpolate vector face data to position r; input is Array4 not MultiFab
// so position must be inside box real bounds
std::array<Real,3> face2r(const Array4<const Real>& xface_array, const Array4<const Real>& yface_array, const Array4<const Real>& zface_array, const Real xpos, const Real ypos, const Real zpos, const IntVect& cell_indices, const amrex::GpuArray<amrex::Real,3>& dx, const amrex::GpuArray<amrex::Real,3>& dom_min) {
   // Cell indices (i,j,k) containing given position
   // Surrounding faces are at x = i, i+1; y = j, j+1; z = k, k+1
   
   const std::array<Real,2>
      x_weight = CIC_weights_1D(xpos, dx[0], cell_indices[0], dom_min[0]),
      y_weight = CIC_weights_1D(ypos, dx[1], cell_indices[1], dom_min[1]),
      z_weight = CIC_weights_1D(zpos, dx[2], cell_indices[2], dom_min[2]);

   std::array<Real,3> ret = {0,0,0};
   for (int nn=0; nn<2; ++nn) {
      ret[0] += xface_array(cell_indices[0]+nn, cell_indices[1], cell_indices[2])*x_weight[nn];
      ret[1] += yface_array(cell_indices[0], cell_indices[1]+nn, cell_indices[2])*y_weight[nn];
      ret[2] += zface_array(cell_indices[0], cell_indices[1], cell_indices[2]+nn)*z_weight[nn];
   }
   
   return ret;
}

// Diagnostics: Compute cell Magnetic and Electric energy - Output is MultiFab with three components - B energy, E energy, Total energy
// TODO: Add Particle kinetic energies after particles added
void compute_EM_energy(MultiFab& Energy_c, const MultiFab& B_c, const MultiFab& E_c) {
   for (MFIter mfi(Energy_c); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.fabbox();
      const Array4<const Real>&
         Bc_array = B_c.const_array(mfi),
         Ec_array = E_c.const_array(mfi);
      const Array4<Real>& Energy_c_array = Energy_c.array(mfi);
      
      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         Real
            B_mag = math::square(Bc_array(ii,jj,kk,0)) + math::square(Bc_array(ii,jj,kk,1)) + math::square(Bc_array(ii,jj,kk,2)),
            E_mag = math::square(Ec_array(ii,jj,kk,0)) + math::square(Ec_array(ii,jj,kk,1)) + math::square(Ec_array(ii,jj,kk,2));
         
         Energy_c_array(ii,jj,kk,0) += B_mag/(PhysConst::mu0*2);
         Energy_c_array(ii,jj,kk,1) += E_mag*PhysConst::eps0/2;
         Energy_c_array(ii,jj,kk,2) += (B_mag/PhysConst::mu0 + E_mag*PhysConst::eps0)/2;
      });
   }
}

// Derivative operators: Curl Edge to Face - Curl is calculated at face centres using edges surrounding face
void curl_e2f(MultiFab& xface_curl, MultiFab& yface_curl, MultiFab& zface_curl, const MultiFab& xedge_data, const MultiFab& yedge_data, const MultiFab& zedge_data, const GpuArray<Real,3>& dx) {
   for (MFIter mfi(xface_curl); mfi.isValid(); ++mfi) {
      const Box&
         bx_fx = mfi.tilebox(AMReXConst::btype_fx),
         bx_fy = mfi.tilebox(AMReXConst::btype_fy),
         bx_fz = mfi.tilebox(AMReXConst::btype_fz);
      const Array4<const Real>&
         e_array_x = xedge_data.const_array(mfi),
         e_array_y = yedge_data.const_array(mfi),
         e_array_z = zedge_data.const_array(mfi);
      const Array4<Real>&
         fc_array_x = xface_curl.array(mfi),
         fc_array_y = yface_curl.array(mfi),
         fc_array_z = zface_curl.array(mfi);
      
      ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
         fc_array_x(ii,jj,kk) += (e_array_z(ii,  jj+1,kk  ) - e_array_z(ii,  jj,  kk  ))/dx[1];
         fc_array_x(ii,jj,kk) -= (e_array_y(ii,  jj,  kk+1) - e_array_y(ii,  jj,  kk  ))/dx[2];
      });
      
      ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
         fc_array_y(ii,jj,kk) += (e_array_x(ii,  jj,  kk+1) - e_array_x(ii,  jj,  kk  ))/dx[2];
         fc_array_y(ii,jj,kk) -= (e_array_z(ii+1,jj,  kk  ) - e_array_z(ii,  jj,  kk  ))/dx[0];
      });
      
      ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
         fc_array_z(ii,jj,kk) += (e_array_y(ii+1,jj,  kk  ) - e_array_y(ii,  jj,  kk  ))/dx[0];
         fc_array_z(ii,jj,kk) -= (e_array_x(ii,  jj+1,kk  ) - e_array_x(ii,  jj,  kk  ))/dx[1];
      });
   }
}

// Derivative operators: Curl Node to Face - Curl is calculated at face centres using edges surrounding face; edge values are averaged from neighbours as in node2edge
// Edge values are skipped, i.e. computes directly from node to face curl
// This should therefore be equivalent to curl_e2f(node2edge(MF),dx)
void curl_n2f(MultiFab& xface_curl, MultiFab& yface_curl, MultiFab& zface_curl, const MultiFab& node_data, const GpuArray<Real,3>& dx, Real fact) {
   for (MFIter mfi(xface_curl); mfi.isValid(); ++mfi) {
      const Box&
         bx_fx = mfi.tilebox(AMReXConst::btype_fx),
         bx_fy = mfi.tilebox(AMReXConst::btype_fy),
         bx_fz = mfi.tilebox(AMReXConst::btype_fz);
      const Array4<const Real>&
         n_array = node_data.const_array(mfi);
      const Array4<Real>&
         fc_array_x = xface_curl.array(mfi),
         fc_array_y = yface_curl.array(mfi),
         fc_array_z = zface_curl.array(mfi);
      
      ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
         fc_array_x(ii,jj,kk) += fact*(n_array(ii,  jj+1,kk,  2) + n_array(ii,  jj+1,kk+1,2))/(2*dx[1]);
         fc_array_x(ii,jj,kk) -= fact*(n_array(ii,  jj,  kk,  2) + n_array(ii,  jj,  kk+1,2))/(2*dx[1]);
         fc_array_x(ii,jj,kk) -= fact*(n_array(ii,  jj,  kk+1,1) + n_array(ii,  jj+1,kk+1,1))/(2*dx[2]);
         fc_array_x(ii,jj,kk) += fact*(n_array(ii,  jj,  kk,  1) + n_array(ii,  jj+1,kk,  1))/(2*dx[2]);
      });
      
      ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
         fc_array_y(ii,jj,kk) += fact*(n_array(ii,  jj,  kk+1,0) + n_array(ii+1,jj,  kk+1,0))/(2*dx[2]);
         fc_array_y(ii,jj,kk) -= fact*(n_array(ii,  jj,  kk,  0) + n_array(ii+1,jj,  kk,  0))/(2*dx[2]);
         fc_array_y(ii,jj,kk) -= fact*(n_array(ii+1,jj,  kk,  2) + n_array(ii+1,jj,  kk+1,2))/(2*dx[0]);
         fc_array_y(ii,jj,kk) += fact*(n_array(ii,  jj,  kk,  2) + n_array(ii,  jj,  kk+1,2))/(2*dx[0]);
      });
      
      ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
         fc_array_z(ii,jj,kk) += fact*(n_array(ii+1,jj,  kk,  1) + n_array(ii+1,jj+1,kk,  1))/(2*dx[0]);
         fc_array_z(ii,jj,kk) -= fact*(n_array(ii,  jj,  kk,  1) + n_array(ii,  jj+1,kk,  1))/(2*dx[0]);
         fc_array_z(ii,jj,kk) -= fact*(n_array(ii,  jj+1,kk,  0) + n_array(ii+1,jj+1,kk,  0))/(2*dx[1]);
         fc_array_z(ii,jj,kk) += fact*(n_array(ii,  jj,  kk,  0) + n_array(ii+1,jj,  kk,  0))/(2*dx[1]);
      });
   }
}

// Derivative operators: Curl Face to Node - Curl is calculated at nodes component-wise using faces adjacent each edge connecting to node
void curl_f2n(MultiFab& node_curl, const MultiFab& xface_data, const MultiFab& yface_data, const MultiFab& zface_data, const GpuArray<Real,3>& dx, Real fact) {
   for (MFIter mfi(node_curl); mfi.isValid(); ++mfi) {
      const Box& bx_n = mfi.validbox();
      const Array4<const Real>&
         f_array_x = xface_data.const_array(mfi),
         f_array_y = yface_data.const_array(mfi),
         f_array_z = zface_data.const_array(mfi);
      const Array4<Real>& nc_array = node_curl.array(mfi);
      
      ParallelFor(bx_n, [&](int ii, int jj, int kk) {
         nc_array(ii,jj,kk,0) -= fact*(f_array_y(ii-1,jj,  kk  ) - f_array_y(ii-1,jj,  kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,0) -= fact*(f_array_y(ii,  jj,  kk  ) - f_array_y(ii,  jj,  kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,0) += fact*(f_array_z(ii-1,jj,  kk  ) - f_array_z(ii-1,jj-1,kk  ))/(2*dx[1]);
         nc_array(ii,jj,kk,0) += fact*(f_array_z(ii,  jj,  kk  ) - f_array_z(ii,  jj-1,kk  ))/(2*dx[1]);

         nc_array(ii,jj,kk,1) -= fact*(f_array_z(ii,  jj-1,kk  ) - f_array_z(ii-1,jj-1,kk  ))/(2*dx[0]);
         nc_array(ii,jj,kk,1) -= fact*(f_array_z(ii,  jj,  kk  ) - f_array_z(ii-1,jj,  kk  ))/(2*dx[0]);
         nc_array(ii,jj,kk,1) += fact*(f_array_x(ii,  jj-1,kk  ) - f_array_x(ii,  jj-1,kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,1) += fact*(f_array_x(ii,  jj,  kk  ) - f_array_x(ii,  jj,  kk-1))/(2*dx[2]);

         nc_array(ii,jj,kk,2) -= fact*(f_array_x(ii,  jj,  kk-1) - f_array_x(ii,  jj-1,kk-1))/(2*dx[1]);
         nc_array(ii,jj,kk,2) -= fact*(f_array_x(ii,  jj,  kk  ) - f_array_x(ii,  jj-1,kk  ))/(2*dx[1]);
         nc_array(ii,jj,kk,2) += fact*(f_array_y(ii,  jj,  kk-1) - f_array_y(ii-1,jj,  kk-1))/(2*dx[0]);
         nc_array(ii,jj,kk,2) += fact*(f_array_y(ii,  jj,  kk  ) - f_array_y(ii-1,jj,  kk  ))/(2*dx[0]);
      });
   }
}

// Derivative operators: Divergence Face to Centre - Divergence is calculated from face-centred data as the change across the cell in each face direction
void div_f2c(MultiFab& cell_div, const MultiFab& xface_data, const MultiFab& yface_data, const MultiFab& zface_data, const GpuArray<Real,3>& dx) {
   for (MFIter mfi(cell_div); mfi.isValid(); ++mfi) {
      const Box& bx_n = mfi.validbox();
      const Array4<const Real>&
         f_array_x = xface_data.const_array(mfi),
         f_array_y = yface_data.const_array(mfi),
         f_array_z = zface_data.const_array(mfi);
      const Array4<Real>& cd_array = cell_div.array(mfi);
      
      ParallelFor(bx_n, [&](int ii, int jj, int kk) {
         cd_array(ii,jj,kk) += (f_array_x(ii+1,jj  ,kk  ) - f_array_x(ii  ,jj  ,kk  ))/dx[0];
         cd_array(ii,jj,kk) += (f_array_y(ii  ,jj+1,kk  ) - f_array_y(ii  ,jj  ,kk  ))/dx[1];
         cd_array(ii,jj,kk) += (f_array_z(ii  ,jj  ,kk+1) - f_array_z(ii  ,jj  ,kk  ))/dx[2];
      });
   }
}

// Derivative operators: Divergence Node to Centre - Divergence is calculated by first averaging node to face data (average of 4 nodes surrounding face) then applying the same rule as div_f2c
void div_n2c(MultiFab& cell_div, const MultiFab& node_data, const GpuArray<Real,3>& dx) {
   for (MFIter mfi(cell_div); mfi.isValid(); ++mfi) {
      const Box& bx_n = mfi.validbox();
      const Array4<const Real>& n_array = node_data.const_array(mfi);
      const Array4<Real>& cd_array = cell_div.array(mfi);
      
      ParallelFor(bx_n, [&](int ii, int jj, int kk) {
         cd_array(ii,jj,kk) += (n_array(ii+1,jj  ,kk  ,0) - n_array(ii  ,jj  ,kk  ,0))/(4*dx[0]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj+1,kk  ,0) - n_array(ii  ,jj+1,kk  ,0))/(4*dx[0]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj  ,kk+1,0) - n_array(ii  ,jj  ,kk+1,0))/(4*dx[0]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj+1,kk+1,0) - n_array(ii  ,jj+1,kk+1,0))/(4*dx[0]);

         cd_array(ii,jj,kk) += (n_array(ii  ,jj+1,kk  ,1) - n_array(ii  ,jj  ,kk  ,1))/(4*dx[1]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj+1,kk  ,1) - n_array(ii+1,jj  ,kk  ,1))/(4*dx[1]);
         cd_array(ii,jj,kk) += (n_array(ii  ,jj+1,kk+1,1) - n_array(ii  ,jj  ,kk+1,1))/(4*dx[1]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj+1,kk+1,1) - n_array(ii+1,jj  ,kk+1,1))/(4*dx[1]);

         cd_array(ii,jj,kk) += (n_array(ii  ,jj  ,kk+1,2) - n_array(ii  ,jj  ,kk  ,2))/(4*dx[2]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj  ,kk+1,2) - n_array(ii+1,jj  ,kk  ,2))/(4*dx[2]);
         cd_array(ii,jj,kk) += (n_array(ii  ,jj+1,kk+1,2) - n_array(ii  ,jj+1,kk  ,2))/(4*dx[2]);
         cd_array(ii,jj,kk) += (n_array(ii+1,jj+1,kk+1,2) - n_array(ii+1,jj+1,kk  ,2))/(4*dx[2]);
      });
   }
}

// Test: Symmetry - Checks multifab data is symmetric in given direction(s); symmetry direction(s) are given by '1' in 3D 'dir' vector.
// If all values of dir are 1 then tests if all values in multifab are constant
// If two values of dir are 1 then tests if data is 1D data, non-constant in only the remaining dimension
// If one value of dir is 1 then tests if data is 2D data, constant only in the given symmetry direction
// If no values of dir are 1 then test is skipped as no symmetry is being tested
bool sym_test(const MultiFab& mf, const IntVect& dir) {

   bool test = true;

   int nvar = mf.nComp();
   
   for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& dat = mf.const_array(mfi);
      const Dim3&
         int_min = lbound(bx),
         int_max = ubound(bx);
      
      for (int nn=0; nn<nvar; ++nn) {
         if (dir[0] == 1) {
            if (dir[1] == 1) {
               if (dir[2] == 1) {
                  Real init = dat(0,0,0,nn);
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                     Real init = dat(0,0,kk,nn);
                     for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                        for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            } else {
               if (dir[2] == 1) {
                  for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                     Real init = dat(0,jj,0,nn);
                     for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                     for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                        Real init = dat(0,jj,kk,nn);
                        for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            }
         } else {
            if (dir[1] == 1) {
               if (dir[2] == 1) {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     Real init = dat(ii,0,0,nn);
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                        Real init = dat(ii,0,kk,nn);
                        for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            } else {
               if (dir[2] == 1) {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        Real init = dat(ii,jj,0,nn);
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  Print() << "Symmetry test has no symmetry directions!" << std::endl;
               }
            }
         }
         if (test == false) {
            break;
         }
      }
      if (test == false) {
         break;
      }
   }

   return test;
}
