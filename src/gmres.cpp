/*
GMRES, containing GMRES solver step for aPIC.

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

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GMRES.H>
#include <AMReX_Print.H>

#include <gmres.h>
#include <constants.h>
#include <operators.h>
#include <math_functions.h>
#include <matrix.h>

using namespace amrex;

// Advance B and E fields by solving gmres system
void gmres_step(std::array<MultiFab,3>& B_f, MultiFab& E_n, GpuArray<Real,3> dx, Real dt, Real theta, Periodicity period, Real rtol, Real atol) {
   BL_PROFILE("gmres_step()");
   
   // To prevent reallocation every step, state vector BE and curl results are pre-allocated static
   static BE
      x(E_n.boxArray(), E_n.distributionMap, E_n.n_grow[0], period),
      b = x.copy_dim(0);
   static MultiFab curl_Bf(convert(E_n.boxArray(),AMReXConst::btype_n),E_n.distributionMap, 3, 0);
   static std::array<MultiFab,3> curl_En = {
      MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fx),E_n.distributionMap, 1, 0),
      MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fy),E_n.distributionMap, 1, 0),
      MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fz),E_n.distributionMap, 1, 0)
   };

   x.setVal(0.0);
   b.setVal(0.0);
   for (int ii=0; ii<3; ++ii) {
      curl_En[ii].setVal(0.0);
   }
   curl_Bf.setVal(0.0);

   // State vector (lhs); initial state is start of time step
   
   x.Copy_Bfx(x, B_f[0]);
   x.Copy_Bfy(x, B_f[1]);
   x.Copy_Bfz(x, B_f[2]);
   x.Copy_En(x, E_n);
   
   // rhs; first includes initial B and E so copy x without ghost cells
   b.Copy(b, x, 0);
   
   // Calculate curls of initial state; no ghost cells needed
   curl_n2f(curl_En, E_n, dx);
   curl_f2n(curl_Bf, B_f, dx);
   
   b.Saxpy_B(b, curl_En, -dt*(1-theta));
   b.Saxpy_En(b, curl_Bf, math::square(PhysConst::c)*dt*(1-theta));
   
   linop gmres_operator(E_n.boxArray(), E_n.distributionMap, E_n.n_grow[0], dx, dt*theta, period);
   
   GMRES<BE,linop> gmres_solver;
   
   gmres_solver.define(gmres_operator);
   gmres_solver.setVerbose(1);
   gmres_solver.solve(x, b, rtol, atol);
   
   int gmres_status = gmres_solver.getStatus();
   
   if (gmres_status > 0) {
      Print() << std::endl << "GMRES failed to converge!" << std::endl
              << "Iteration count: " << gmres_solver.getNumIters() << std::endl
              << "Residual norm: " << gmres_solver.getResidualNorm() << std::endl << std::endl;
   } else {
      Print() << "GMRES Iteration count: " << gmres_solver.getNumIters() << std::endl;
   }
   
   MultiFab::Copy(B_f[0], x.getB_fx(), 0, 0, 1, x.nGrow_Bfx());
   MultiFab::Copy(B_f[1], x.getB_fy(), 0, 0, 1, x.nGrow_Bfy());
   MultiFab::Copy(B_f[2], x.getB_fz(), 0, 0, 1, x.nGrow_Bfz());
   MultiFab::Copy(E_n, x.getE_n(), 0, 0, 3, x.nGrow_En());
}

// Compute the curl operator for a given box from faces to nodes
// This function only needs to be computed once per box so speed is not that important
// NB: The box passed should not contain ghost cells
matrix<Real> get_curl_f2n_operator(const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {
   
   const Box&
      bx_fx = grow(convert(bx,AMReXConst::btype_fx), nghost),
      bx_fy = grow(convert(bx,AMReXConst::btype_fy), nghost),
      bx_fz = grow(convert(bx,AMReXConst::btype_fz), nghost),
      bx_n = convert(bx,AMReXConst::btype_n);
   
   const IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = len_fx[0]*len_fx[1]*len_fx[2],
      total_fy = len_fy[0]*len_fy[1]*len_fy[2],
      total_fz = len_fz[0]*len_fz[1]*len_fz[2],
      total_fxy = total_fx + total_fy,
      total_n = len_n[0]*len_n[1]*len_n[2];
   
   matrix<Real> ret(3*total_n, total_fx + total_fy + total_fz, 0.0);
   
   ParallelFor(bx_n, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000(ii,   jj,   kk  ),
         cell_001(ii,   jj,   kk-1),
         cell_010(ii,   jj-1, kk  ),
         cell_011(ii,   jj-1, kk-1),
         cell_100(ii-1, jj,   kk  ),
         cell_101(ii-1, jj,   kk-1),
         cell_110(ii-1, jj-1, kk  );
      // Find column in matrix for each face
      // Each face box has different dimensions, and
      // Matrix columns are organised as x-faces first then y- and z-
      // Rows are organised instead as all three components at each node together
      const int
         // Node ID
         nID_000 = get_cellID(cell_000, len_n),
         // x-face IDs
         fxID_000 = get_cellID(cell_000, len_fx),
         fxID_001 = get_cellID(cell_001, len_fx),
         fxID_010 = get_cellID(cell_010, len_fx),
         fxID_011 = get_cellID(cell_011, len_fx),
         // y-face IDs
         fyID_000 = get_cellID(cell_000, len_fy) + total_fx,
         fyID_001 = get_cellID(cell_001, len_fy) + total_fx,
         fyID_100 = get_cellID(cell_100, len_fy) + total_fx,
         fyID_101 = get_cellID(cell_101, len_fy) + total_fx,
         // z-face IDs
         fzID_000 = get_cellID(cell_000, len_fz) + total_fxy,
         fzID_010 = get_cellID(cell_010, len_fz) + total_fxy,
         fzID_100 = get_cellID(cell_100, len_fz) + total_fxy,
         fzID_110 = get_cellID(cell_110, len_fz) + total_fxy;
      
      ret(nID_000 + 0, fyID_101) += 1/dx[2];
      ret(nID_000 + 0, fyID_100) -= 1/dx[2];
      ret(nID_000 + 0, fzID_100) += 1/dx[1];
      ret(nID_000 + 0, fzID_110) -= 1/dx[1];
      ret(nID_000 + 0, fyID_001) += 1/dx[2];
      ret(nID_000 + 0, fyID_000) -= 1/dx[2];
      ret(nID_000 + 0, fzID_000) += 1/dx[1];
      ret(nID_000 + 0, fzID_010) -= 1/dx[1];
      
      ret(nID_000 + 1, fzID_110) += 1/dx[0];
      ret(nID_000 + 1, fzID_010) -= 1/dx[0];
      ret(nID_000 + 1, fxID_010) += 1/dx[2];
      ret(nID_000 + 1, fxID_011) -= 1/dx[2];
      ret(nID_000 + 1, fzID_100) += 1/dx[0];
      ret(nID_000 + 1, fzID_000) -= 1/dx[0];
      ret(nID_000 + 1, fxID_000) += 1/dx[2];
      ret(nID_000 + 1, fxID_001) -= 1/dx[2];
      
      ret(nID_000 + 2, fxID_011) += 1/dx[1];
      ret(nID_000 + 2, fxID_001) -= 1/dx[1];
      ret(nID_000 + 2, fyID_001) += 1/dx[0];
      ret(nID_000 + 2, fyID_101) -= 1/dx[0];
      ret(nID_000 + 2, fxID_010) += 1/dx[1];
      ret(nID_000 + 2, fxID_000) -= 1/dx[1];
      ret(nID_000 + 2, fyID_000) += 1/dx[0];
      ret(nID_000 + 2, fyID_100) -= 1/dx[0];
   });
   
   return ret;
}

// Compute the curl operator for a given box from nodes to faces
// This function only needs to be computed once per box so speed is not that important
// NB: The box passed should not contain ghost cells
matrix<Real> get_curl_n2f_operator(const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {
   
   const Box&
      bx_fx = convert(bx,AMReXConst::btype_fx),
      bx_fy = convert(bx,AMReXConst::btype_fy),
      bx_fz = convert(bx,AMReXConst::btype_fz),
      bx_n = grow(convert(bx,AMReXConst::btype_n), nghost);
   
   const IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = len_fx[0]*len_fx[1]*len_fx[2],
      total_fy = len_fy[0]*len_fy[1]*len_fy[2],
      total_fz = len_fz[0]*len_fz[1]*len_fz[2],
      total_fxy = total_fx + total_fy,
      total_n = len_n[0]*len_n[1]*len_n[2];
   
   matrix<Real> ret(total_fx + total_fy + total_fz, 3*total_n, 0.0);
   
   ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000(ii,   jj,   kk  ),
         cell_001(ii,   jj,   kk+1),
         cell_010(ii,   jj+1, kk  ),
         cell_011(ii,   jj+1, kk+1);
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // x-face ID
         fxID_000 = get_cellID(cell_000, len_fx),
         // node IDs
         nID_000 = get_cellID(cell_000, len_n),
         nID_001 = get_cellID(cell_001, len_n),
         nID_010 = get_cellID(cell_010, len_n),
         nID_011 = get_cellID(cell_011, len_n);
      
      ret(fxID_000, nID_000 + 1) += 1/(2*dx[2]);
      ret(fxID_000, nID_010 + 1) += 1/(2*dx[2]);
      ret(fxID_000, nID_010 + 2) += 1/(2*dx[1]);
      ret(fxID_000, nID_011 + 2) += 1/(2*dx[1]);
      ret(fxID_000, nID_011 + 1) -= 1/(2*dx[2]);
      ret(fxID_000, nID_001 + 1) -= 1/(2*dx[2]);
      ret(fxID_000, nID_001 + 2) -= 1/(2*dx[1]);
      ret(fxID_000, nID_000 + 2) -= 1/(2*dx[1]);
   });

   ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000(ii,   jj,   kk  ),
         cell_001(ii,   jj,   kk+1),
         cell_100(ii+1, jj,   kk  ),
         cell_101(ii+1, jj,   kk+1);
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // y-face ID
         fyID_000 = get_cellID(cell_000, len_fy) + total_fx,
         // node IDs
         nID_000 = get_cellID(cell_000, len_n),
         nID_001 = get_cellID(cell_001, len_n),
         nID_100 = get_cellID(cell_100, len_n),
         nID_101 = get_cellID(cell_101, len_n);
      
      ret(fyID_000, nID_000 + 2) += 1/(2*dx[0]);
      ret(fyID_000, nID_001 + 2) += 1/(2*dx[0]);
      ret(fyID_000, nID_001 + 0) += 1/(2*dx[2]);
      ret(fyID_000, nID_101 + 0) += 1/(2*dx[2]);
      ret(fyID_000, nID_101 + 2) -= 1/(2*dx[0]);
      ret(fyID_000, nID_100 + 2) -= 1/(2*dx[0]);
      ret(fyID_000, nID_100 + 0) -= 1/(2*dx[2]);
      ret(fyID_000, nID_000 + 0) -= 1/(2*dx[2]);
   });
   
   ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000(ii,   jj,   kk  ),
         cell_010(ii,   jj+1, kk  ),
         cell_100(ii+1, jj,   kk  ),
         cell_110(ii+1, jj+1, kk  );
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // z-face ID
         fzID_000 = get_cellID(cell_000, len_fz) + total_fxy,
         // node IDs
         nID_000 = get_cellID(cell_000, len_n),
         nID_010 = get_cellID(cell_010, len_n),
         nID_100 = get_cellID(cell_100, len_n),
         nID_110 = get_cellID(cell_110, len_n);
      
      ret(fzID_000, nID_000 + 0) += 1/(2*dx[1]);
      ret(fzID_000, nID_100 + 0) += 1/(2*dx[1]);
      ret(fzID_000, nID_100 + 1) += 1/(2*dx[0]);
      ret(fzID_000, nID_110 + 1) += 1/(2*dx[0]);
      ret(fzID_000, nID_110 + 0) -= 1/(2*dx[1]);
      ret(fzID_000, nID_010 + 0) -= 1/(2*dx[1]);
      ret(fzID_000, nID_010 + 1) -= 1/(2*dx[0]);
      ret(fzID_000, nID_000 + 1) -= 1/(2*dx[0]);
   });
   
   return ret;
}

// Get cell (or node etc.) ID from given cell indices of cell in box
int get_cellID(const int x_index, const int y_index, const int z_index, const IntVect& len) {
   return (z_index*len[1] + y_index)*len[0] + x_index;
}

// Get cell (or node etc.) ID from given cell indices of cell in box
int get_cellID(const IntVect& cell_indices, const IntVect& len) {
   return get_cellID(cell_indices[0], cell_indices[1], cell_indices[2], len);
}

// Get cell (or node etc.) ID from given cell indices of cell in box
int get_cellID(const std::array<int,3>& cell_indices, const IntVect& len) {
   return get_cellID(cell_indices[0], cell_indices[1], cell_indices[2], len);
}

// Get cell (or node etc.) indices of cell in box given cell ID
IntVect get_cell_indices(int cellID, const IntVect& len) {
   int
      x_index = cellID%len[0],
      y_index = (cellID/len[0])%len[1],
      z_index = cellID/(len[0]*len[1]);
   return IntVect(x_index,y_index,z_index);
}
