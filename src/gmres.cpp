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

#include <gmres.h>
#include <constants.h>
#include <operators.h>
#include <particles.h>
#include <math_functions.h>
#include <matrix.h>

#include <AMReX_REAL.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GMRES.H>
#include <AMReX_Print.H>

#include <vector>

using namespace amrex;

// Advance B and E fields by solving gmres system
void gmres_step(BE& x, const MultiFab& jHat, std::vector<myPContainer>& pContainer_list, std::vector<Population>& pop_list, const Real dV_inv, const GpuArray<Real,3>& dx, const Real dt, const Real theta, const Real rtol, const Real atol, const int verbosity, const int max_gmres) {
   BL_PROFILE("gmres_step()");
   
   // To prevent reallocation every step, rhs state vector b is static
   // This may have issues if using omp
   static BE
      b = x.copy_dim(0);

   b.setVal(0.0);
   
   // rhs; first includes initial B and E so copy x without ghost cells
   BE::Copy(b, x, 0);
   
   curl_n2f(b.getB_fx(), b.getB_fy(), b.getB_fz(), x.getE_n_const(), dx, -dt*(1-theta));
   curl_f2n(b.getE_n(), x.getB_fx_const(), x.getB_fy_const(), x.getB_fz_const(), dx, math::square(PhysConst::c)*dt*(1-theta));

   BE::Saxpy_En(b, jHat, -dt/PhysConst::eps0);
   
   linop_direct gmres_operator(x.getBoxArray(), x.getDistributionMap(), x.nghost(), dx, dt, theta, x.getPeriod(), pContainer_list, pop_list, dV_inv);
   
   GMRES<BE,linop_direct> gmres_solver;
   
   gmres_solver.define(gmres_operator);
   gmres_solver.setVerbose(verbosity);
   gmres_solver.setMaxIters(max_gmres);
   gmres_solver.solve(x, b, rtol, atol);
   
   int gmres_status = gmres_solver.getStatus();
   
   if (gmres_status > 0) {
      Print() << std::endl << "GMRES failed to converge!" << std::endl
              << "Iteration count: " << gmres_solver.getNumIters() << std::endl
              << "Residual norm: " << gmres_solver.getResidualNorm() << std::endl << std::endl;
   } else {
      Print() << "GMRES Iteration count: " << gmres_solver.getNumIters() << std::endl;
   }
}

/*
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
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_fxy = total_fx + total_fy,
      total_n = math::product(len_n);
   
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
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_fxy = total_fx + total_fy,
      total_n = math::product(len_n);
   
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
*/

// Compute the curl operator for a given box from faces to nodes
// This function only needs to be computed once per box so speed is not that important
// This function returns three separate matrices for the three components of B_f in the input x of Ax
// NB: The box passed should not contain ghost cells
void get_curl_f2n_operator(matrix<Real>& curl_x, matrix<Real>& curl_y, matrix<Real>& curl_z, const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {
   const Real
      grad_x = 1/(2*dx[0]),
      grad_y = 1/(2*dx[1]),
      grad_z = 1/(2*dx[2]);
   
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
   
   const int total_n = math::product(len_n);

   const IntVect base_index { bx_n.smallEnd() };
   
   ParallelFor(bx_n, [&](int ii, int jj, int kk) {
      // Coords of node and adjacent faces
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index,
         cell_100 = cell_000 - IntVect(1,0,0),
         cell_010 = cell_000 - IntVect(0,1,0),
         cell_110 = cell_000 - IntVect(1,1,0),
         cell_001 = cell_000 - IntVect(0,0,1),
         cell_101 = cell_000 - IntVect(1,0,1),
         cell_011 = cell_000 - IntVect(0,1,1);
      
      // Find column in matrix for each face
      // Each face box has different dimensions, and
      // Matrix columns are organised as x-faces first then y- and z-
      // Rows are organised instead as all three components at each node together
      const int
         // Node ID
         nxID_000 = get_cellID(cell_000, len_n),
         nyID_000 = nxID_000 + total_n,
         nzID_000 = nyID_000 + total_n,
         // x-face IDs
         fxID_000 = get_cellID(cell_000, len_fx, nghost),
         fxID_001 = get_cellID(cell_001, len_fx, nghost),
         fxID_010 = get_cellID(cell_010, len_fx, nghost),
         fxID_011 = get_cellID(cell_011, len_fx, nghost),
         // y-face IDs
         fyID_000 = get_cellID(cell_000, len_fy, nghost),
         fyID_001 = get_cellID(cell_001, len_fy, nghost),
         fyID_100 = get_cellID(cell_100, len_fy, nghost),
         fyID_101 = get_cellID(cell_101, len_fy, nghost),
         // z-face IDs
         fzID_000 = get_cellID(cell_000, len_fz, nghost),
         fzID_010 = get_cellID(cell_010, len_fz, nghost),
         fzID_100 = get_cellID(cell_100, len_fz, nghost),
         fzID_110 = get_cellID(cell_110, len_fz, nghost);
      
      // x-component of curl
      curl_y(nxID_000, fyID_101) += grad_z;
      curl_y(nxID_000, fyID_100) -= grad_z;
      curl_z(nxID_000, fzID_100) += grad_y;
      curl_z(nxID_000, fzID_110) -= grad_y;
      curl_y(nxID_000, fyID_001) += grad_z;
      curl_y(nxID_000, fyID_000) -= grad_z;
      curl_z(nxID_000, fzID_000) += grad_y;
      curl_z(nxID_000, fzID_010) -= grad_y;
      // y-component of curl
      curl_z(nyID_000, fzID_110) += grad_x;
      curl_z(nyID_000, fzID_010) -= grad_x;
      curl_x(nyID_000, fxID_010) += grad_z;
      curl_x(nyID_000, fxID_011) -= grad_z;
      curl_z(nyID_000, fzID_100) += grad_x;
      curl_z(nyID_000, fzID_000) -= grad_x;
      curl_x(nyID_000, fxID_000) += grad_z;
      curl_x(nyID_000, fxID_001) -= grad_z;
      // z-component of curl
      curl_x(nzID_000, fxID_011) += grad_y;
      curl_x(nzID_000, fxID_001) -= grad_y;
      curl_y(nzID_000, fyID_001) += grad_x;
      curl_y(nzID_000, fyID_101) -= grad_x;
      curl_x(nzID_000, fxID_010) += grad_y;
      curl_x(nzID_000, fxID_000) -= grad_y;
      curl_y(nzID_000, fyID_000) += grad_x;
      curl_y(nzID_000, fyID_100) -= grad_x;
   });
}

// Compute the curl operator for a given box from nodes to faces
// This function only needs to be computed once per box so speed is not that important
// NB: The box passed should not contain ghost cells
void get_curl_n2f_operator(matrix<Real>& curl_x, matrix<Real>& curl_y, matrix<Real>& curl_z, const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {
   const Real
      grad_x = 1/(2*dx[0]),
      grad_y = 1/(2*dx[1]),
      grad_z = 1/(2*dx[2]);
   
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
   
   const int total_n = math::product(len_n);

   const IntVect base_index_x { bx_fx.smallEnd() };
   
   ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_x,
         cell_010 = cell_000 + IntVect(0,1,0),
         cell_001 = cell_000 + IntVect(0,0,1),
         cell_011 = cell_000 + IntVect(0,1,1);
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // x-face ID
         fxID_000 = get_cellID(cell_000, len_fx),
         // node IDs
         nyID_000 = get_cellID(cell_000, len_n, nghost) + total_n,
         nyID_001 = get_cellID(cell_001, len_n, nghost) + total_n,
         nyID_010 = get_cellID(cell_010, len_n, nghost) + total_n,
         nyID_011 = get_cellID(cell_011, len_n, nghost) + total_n,
         nzID_000 = nyID_000 + total_n,
         nzID_001 = nyID_001 + total_n,
         nzID_010 = nyID_010 + total_n,
         nzID_011 = nyID_011 + total_n;
         
      curl_x(fxID_000, nyID_000) += grad_z;
      curl_x(fxID_000, nyID_010) += grad_z;
      curl_x(fxID_000, nzID_010) += grad_y;
      curl_x(fxID_000, nzID_011) += grad_y;
      curl_x(fxID_000, nyID_011) -= grad_z;
      curl_x(fxID_000, nyID_001) -= grad_z;
      curl_x(fxID_000, nzID_001) -= grad_y;
      curl_x(fxID_000, nzID_000) -= grad_y;
   });
   
   const IntVect base_index_y { bx_fy.smallEnd() };
   
   ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_y,
         cell_100 = cell_000 + IntVect(1,0,0),
         cell_001 = cell_000 + IntVect(0,0,1),
         cell_101 = cell_000 + IntVect(1,0,1);
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // y-face ID
         fyID_000 = get_cellID(cell_000, len_fy),
         // node IDs
         nxID_000 = get_cellID(cell_000, len_n, nghost),
         nxID_001 = get_cellID(cell_001, len_n, nghost),
         nxID_100 = get_cellID(cell_100, len_n, nghost),
         nxID_101 = get_cellID(cell_101, len_n, nghost),
         nzID_000 = nxID_000 + 2*total_n,
         nzID_001 = nxID_001 + 2*total_n,
         nzID_100 = nxID_100 + 2*total_n,
         nzID_101 = nxID_101 + 2*total_n;
      
      curl_y(fyID_000, nzID_000) += grad_x;
      curl_y(fyID_000, nzID_001) += grad_x;
      curl_y(fyID_000, nxID_001) += grad_z;
      curl_y(fyID_000, nxID_101) += grad_z;
      curl_y(fyID_000, nzID_101) -= grad_x;
      curl_y(fyID_000, nzID_100) -= grad_x;
      curl_y(fyID_000, nxID_100) -= grad_z;
      curl_y(fyID_000, nxID_000) -= grad_z;
   });
   
   const IntVect base_index_z { bx_fz.smallEnd() };
   
   ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_z,
         cell_100 = cell_000 + IntVect(1,0,0),
         cell_010 = cell_000 + IntVect(0,1,0),
         cell_110 = cell_000 + IntVect(1,1,0);
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // z-face ID
         fzID_000 = get_cellID(cell_000, len_fz),
         // node IDs
         nxID_000 = get_cellID(cell_000, len_n, nghost),
         nxID_010 = get_cellID(cell_010, len_n, nghost),
         nxID_100 = get_cellID(cell_100, len_n, nghost),
         nxID_110 = get_cellID(cell_110, len_n, nghost),
         nyID_000 = nxID_000 + total_n,
         nyID_010 = nxID_010 + total_n,
         nyID_100 = nxID_100 + total_n,
         nyID_110 = nxID_110 + total_n;
      
      curl_z(fzID_000, nxID_000) += grad_y;
      curl_z(fzID_000, nxID_100) += grad_y;
      curl_z(fzID_000, nyID_100) += grad_x;
      curl_z(fzID_000, nyID_110) += grad_x;
      curl_z(fzID_000, nxID_110) -= grad_y;
      curl_z(fzID_000, nxID_010) -= grad_y;
      curl_z(fzID_000, nyID_010) -= grad_x;
      curl_z(fzID_000, nyID_000) -= grad_x;
   });
}

// Compute the curl operator for a given box from faces to nodes
// This function only needs to be computed once per box so speed is not that important
// This function returns three separate sparse matrices for the three components of B_f in the input x of Ax
// NB: The box passed should not contain ghost cells
void get_curl_f2n_operator(sp_matrix<Real>& curl_x, sp_matrix<Real>& curl_y, sp_matrix<Real>& curl_z, const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {
   
   const Real
      grad_x = 1/(2*dx[0]),
      grad_y = 1/(2*dx[1]),
      grad_z = 1/(2*dx[2]);
   
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
   
   const int total_n = math::product(len_n);
   
   // *_xy, first dim "x" indicates component of B (thus different matrices), second dim "y" indicates component of E (thus different chunk of matrix)
   // These never match as B_i does not affect E_i
   std::vector<Real>
      dat_xy,
      dat_xz,
      dat_yx,
      dat_yz,
      dat_zx,
      dat_zy;

   std::vector<int>
      row_indices_xy, col_indices_xy,
      row_indices_xz, col_indices_xz,
      row_indices_yx, col_indices_yx,
      row_indices_yz, col_indices_yz,
      row_indices_zx, col_indices_zx,
      row_indices_zy, col_indices_zy;
     
   // 4 Entries per row - i.e. for each dimension pair i,j,
   // 4 values of B_j in neighbouring faces affect the local E_i
   constexpr int cols_per_row = 4;
   
   dat_xy.reserve(cols_per_row*total_n);
   dat_xz.reserve(cols_per_row*total_n);
   dat_yx.reserve(cols_per_row*total_n);
   dat_yz.reserve(cols_per_row*total_n);
   dat_zx.reserve(cols_per_row*total_n);
   dat_zy.reserve(cols_per_row*total_n);
   
   row_indices_xy.reserve(cols_per_row*total_n);
   row_indices_zy.reserve(cols_per_row*total_n);
   row_indices_xz.reserve(cols_per_row*total_n);
   row_indices_yx.reserve(cols_per_row*total_n);
   row_indices_yz.reserve(cols_per_row*total_n);
   row_indices_zx.reserve(cols_per_row*total_n);
   row_indices_zy.reserve(cols_per_row*total_n);
   
   col_indices_xy.reserve(cols_per_row*total_n);
   col_indices_xz.reserve(cols_per_row*total_n);
   col_indices_yx.reserve(cols_per_row*total_n);
   col_indices_yz.reserve(cols_per_row*total_n);
   col_indices_zx.reserve(cols_per_row*total_n);
   col_indices_zy.reserve(cols_per_row*total_n);

   const IntVect base_index { bx_n.smallEnd() };
   
   ParallelFor(bx_n, [&](int ii, int jj, int kk) {
      // Coords of node and adjacent faces
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index,
         cell_100 = cell_000 - IntVect(1,0,0),
         cell_010 = cell_000 - IntVect(0,1,0),
         cell_110 = cell_000 - IntVect(1,1,0),
         cell_001 = cell_000 - IntVect(0,0,1),
         cell_101 = cell_000 - IntVect(1,0,1),
         cell_011 = cell_000 - IntVect(0,1,1);
      
      // Find column in matrix for each face
      // Each face box has different dimensions, and
      // Matrix columns are organised as x-faces first then y- and z-
      // Rows are organised instead as all three components at each node together
      const int
         // Node ID
         nxID_000 = get_cellID(cell_000, len_n),
         nyID_000 = nxID_000 + total_n,
         nzID_000 = nyID_000 + total_n,
         // x-face IDs
         fxID_000 = get_cellID(cell_000, len_fx, nghost),
         fxID_010 = get_cellID(cell_010, len_fx, nghost),
         fxID_001 = get_cellID(cell_001, len_fx, nghost),
         fxID_011 = get_cellID(cell_011, len_fx, nghost),
         // y-face IDs
         fyID_000 = get_cellID(cell_000, len_fy, nghost),
         fyID_100 = get_cellID(cell_100, len_fy, nghost),
         fyID_001 = get_cellID(cell_001, len_fy, nghost),
         fyID_101 = get_cellID(cell_101, len_fy, nghost),
         // z-face IDs
         fzID_000 = get_cellID(cell_000, len_fz, nghost),
         fzID_100 = get_cellID(cell_100, len_fz, nghost),
         fzID_010 = get_cellID(cell_010, len_fz, nghost),
         fzID_110 = get_cellID(cell_110, len_fz, nghost);
      
      // x-component of curl
      dat_yx.push_back(grad_z);
      dat_yx.push_back(grad_z);
      dat_yx.push_back(-grad_z);
      dat_yx.push_back(-grad_z);
      for (int nn=0; nn<4; ++nn) {
         row_indices_yx.push_back(nxID_000);
      }
      col_indices_yx.push_back(fyID_101);
      col_indices_yx.push_back(fyID_001);
      col_indices_yx.push_back(fyID_100);
      col_indices_yx.push_back(fyID_000);

      dat_zx.push_back(-grad_y);
      dat_zx.push_back(-grad_y);
      dat_zx.push_back(grad_y);
      dat_zx.push_back(grad_y);
      for (int nn=0; nn<4; ++nn) {
         row_indices_zx.push_back(nxID_000);
      }
      col_indices_zx.push_back(fzID_110);
      col_indices_zx.push_back(fzID_010);
      col_indices_zx.push_back(fzID_100);
      col_indices_zx.push_back(fzID_000);
      
      // y-component of curl
      dat_zy.push_back(grad_x);
      dat_zy.push_back(-grad_x);
      dat_zy.push_back(grad_x);
      dat_zy.push_back(-grad_x);
      for (int nn=0; nn<4; ++nn) {
         row_indices_zy.push_back(nyID_000);
      }
      col_indices_zy.push_back(fzID_110);
      col_indices_zy.push_back(fzID_010);
      col_indices_zy.push_back(fzID_100);
      col_indices_zy.push_back(fzID_000);

      dat_xy.push_back(-grad_z);
      dat_xy.push_back(-grad_z);
      dat_xy.push_back(grad_z);
      dat_xy.push_back(grad_z);
      for (int nn=0; nn<4; ++nn) {
         row_indices_xy.push_back(nyID_000);
      }
      col_indices_xy.push_back(fxID_011);
      col_indices_xy.push_back(fxID_001);
      col_indices_xy.push_back(fxID_010);
      col_indices_xy.push_back(fxID_000);
      
      // z-component of curl
      dat_xz.push_back(grad_y);
      dat_xz.push_back(-grad_y);
      dat_xz.push_back(grad_y);
      dat_xz.push_back(-grad_y);
      for (int nn=0; nn<4; ++nn) {
         row_indices_xz.push_back(nzID_000);
      }
      col_indices_xz.push_back(fxID_011);
      col_indices_xz.push_back(fxID_001);
      col_indices_xz.push_back(fxID_010);
      col_indices_xz.push_back(fxID_000);

      dat_yz.push_back(-grad_x);
      dat_yz.push_back(grad_x);
      dat_yz.push_back(-grad_x);
      dat_yz.push_back(grad_x);
      for (int nn=0; nn<4; ++nn) {
         row_indices_yz.push_back(nzID_000);
      }
      col_indices_yz.push_back(fyID_101);
      col_indices_yz.push_back(fyID_001);
      col_indices_yz.push_back(fyID_100);
      col_indices_yz.push_back(fyID_000);
   });
   
   curl_x.add_chunk(dat_xy, row_indices_xy, col_indices_xy);
   curl_x.add_chunk(dat_xz, row_indices_xz, col_indices_xz);
   curl_x.finalise();
   
   curl_y.add_chunk(dat_yx, row_indices_yx, col_indices_yx);
   curl_y.add_chunk(dat_yz, row_indices_yz, col_indices_yz);
   curl_y.finalise();
   
   curl_z.add_chunk(dat_zx, row_indices_zx, col_indices_zx);
   curl_z.add_chunk(dat_zy, row_indices_zy, col_indices_zy);
   curl_z.finalise();
}

// Compute the curl operator for a given box from nodes to faces
// This function only needs to be computed once per box so speed is not that important
// NB: The box passed should not contain ghost cells
void get_curl_n2f_operator(sp_matrix<Real>& curl_x, sp_matrix<Real>& curl_y, sp_matrix<Real>& curl_z, const Box& bx, const int nghost, const GpuArray<Real,3>& dx) {

   const Real
      grad_x = 1/(2*dx[0]),
      grad_y = 1/(2*dx[1]),
      grad_z = 1/(2*dx[2]);
   
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
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_n = math::product(len_n);

   // 8 Entries per row - i.e. for each dimension pair i,j,
   // 8 values of E_j in neighbouring nodes affect the local B_i
   constexpr int cols_per_row = 8;
   
   std::vector<Real>
      dat_x,
      dat_y,
      dat_z;
   std::vector<int>
      row_indices_x, col_indices_x,
      row_indices_y, col_indices_y,
      row_indices_z, col_indices_z;
   
   dat_x.reserve(cols_per_row*total_fx);
   dat_y.reserve(cols_per_row*total_fy);
   dat_z.reserve(cols_per_row*total_fz);
   
   row_indices_x.reserve(cols_per_row*total_fx);
   row_indices_y.reserve(cols_per_row*total_fy);
   row_indices_z.reserve(cols_per_row*total_fz);
   
   col_indices_x.reserve(cols_per_row*total_fx);
   col_indices_y.reserve(cols_per_row*total_fy);
   col_indices_z.reserve(cols_per_row*total_fz);

   const IntVect base_index_x { bx_fx.smallEnd() };
   
   ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_x,
         cell_010 = cell_000 + IntVect(0,1,0),
         cell_001 = cell_000 + IntVect(0,0,1),
         cell_011 = cell_000 + IntVect(0,1,1);
      
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // x-face ID
         fxID_000 = get_cellID(cell_000, len_fx),
         // node IDs
         nyID_000 = get_cellID(cell_000, len_n, nghost) + total_n,
         nyID_010 = get_cellID(cell_010, len_n, nghost) + total_n,
         nyID_001 = get_cellID(cell_001, len_n, nghost) + total_n,
         nyID_011 = get_cellID(cell_011, len_n, nghost) + total_n,
         nzID_000 = nyID_000 + total_n,
         nzID_010 = nyID_010 + total_n,
         nzID_001 = nyID_001 + total_n,
         nzID_011 = nyID_011 + total_n;

      dat_x.push_back(grad_z);
      dat_x.push_back(grad_z);
      dat_x.push_back(-grad_z);
      dat_x.push_back(-grad_z);
      dat_x.push_back(-grad_y);
      dat_x.push_back(grad_y);
      dat_x.push_back(-grad_y);
      dat_x.push_back(grad_y);
      for (int nn=0; nn<8; ++nn) {
         row_indices_x.push_back(fxID_000);
      }
      col_indices_x.push_back(nyID_000);
      col_indices_x.push_back(nyID_010);
      col_indices_x.push_back(nyID_001);
      col_indices_x.push_back(nyID_011);
      col_indices_x.push_back(nzID_000);
      col_indices_x.push_back(nzID_010);
      col_indices_x.push_back(nzID_001);
      col_indices_x.push_back(nzID_011);
   });

   const IntVect base_index_y { bx_fy.smallEnd() };
   
   ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_y,
         cell_100 = cell_000 + IntVect(1,0,0),
         cell_001 = cell_000 + IntVect(0,0,1),
         cell_101 = cell_000 + IntVect(1,0,1);
      
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // y-face ID
         fyID_000 = get_cellID(cell_000, len_fy),
         // node IDs
         nxID_000 = get_cellID(cell_000, len_n, nghost),
         nxID_100 = get_cellID(cell_100, len_n, nghost),
         nxID_001 = get_cellID(cell_001, len_n, nghost),
         nxID_101 = get_cellID(cell_101, len_n, nghost),
         nzID_000 = nxID_000 + 2*total_n,
         nzID_100 = nxID_100 + 2*total_n,
         nzID_001 = nxID_001 + 2*total_n,
         nzID_101 = nxID_101 + 2*total_n;

      dat_y.push_back(-grad_z);
      dat_y.push_back(-grad_z);
      dat_y.push_back(grad_z);
      dat_y.push_back(grad_z);
      dat_y.push_back(grad_x);
      dat_y.push_back(-grad_x);
      dat_y.push_back(grad_x);
      dat_y.push_back(-grad_x);
      for (int nn=0; nn<8; ++nn) {
         row_indices_y.push_back(fyID_000);
      }
      col_indices_y.push_back(nxID_000);
      col_indices_y.push_back(nxID_100);
      col_indices_y.push_back(nxID_001);
      col_indices_y.push_back(nxID_101);
      col_indices_y.push_back(nzID_000);
      col_indices_y.push_back(nzID_100);
      col_indices_y.push_back(nzID_001);
      col_indices_y.push_back(nzID_101);
   });

   const IntVect base_index_z { bx_fz.smallEnd() };
   
   ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
      const IntVect
         cell_000 = IntVect(ii, jj, kk) - base_index_z,
         cell_100 = cell_000 + IntVect(1,0,0),
         cell_010 = cell_000 + IntVect(0,1,0),
         cell_110 = cell_000 + IntVect(1,1,0);
      
      // Find column in matrix for each node
      // Each face box has different dimensions, and
      // Matrix rows are organised as x-faces first then y- and z-
      // Columns are organised instead as all three components at each node together
      const int
         // z-face ID
         fzID_000 = get_cellID(cell_000, len_fz),
         // node IDs
         nxID_000 = get_cellID(cell_000, len_n, nghost),
         nxID_100 = get_cellID(cell_100, len_n, nghost),
         nxID_010 = get_cellID(cell_010, len_n, nghost),
         nxID_110 = get_cellID(cell_110, len_n, nghost),
         nyID_000 = nxID_000 + total_n,
         nyID_100 = nxID_100 + total_n,
         nyID_010 = nxID_010 + total_n,
         nyID_110 = nxID_110 + total_n;

      dat_z.push_back(grad_y);
      dat_z.push_back(grad_y);
      dat_z.push_back(-grad_y);
      dat_z.push_back(-grad_y);
      dat_z.push_back(-grad_x);
      dat_z.push_back(grad_x);
      dat_z.push_back(-grad_x);
      dat_z.push_back(grad_x);
      for (int nn=0; nn<8; ++nn) {
         row_indices_z.push_back(fzID_000);
      }
      col_indices_z.push_back(nxID_000);
      col_indices_z.push_back(nxID_100);
      col_indices_z.push_back(nxID_010);
      col_indices_z.push_back(nxID_110);
      col_indices_z.push_back(nyID_000);
      col_indices_z.push_back(nyID_100);
      col_indices_z.push_back(nyID_010);
      col_indices_z.push_back(nyID_110);
   });

   curl_x.add_chunk(dat_x, row_indices_x, col_indices_x);
   curl_y.add_chunk(dat_y, row_indices_y, col_indices_y);
   curl_z.add_chunk(dat_z, row_indices_z, col_indices_z);
}

/*
// Apply given matrix to E data in x to give B data in Ax (current)
// Note: Data is summed not overwritten
void BE::apply_matrix_E2E(BE& Ax, const BE& x, const LayoutData<matrix<Real>>& matA_E2E) {
   
}
*/
