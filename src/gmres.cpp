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

using namespace amrex;

// Advance B and E fields by solving gmres system
void gmres_step(std::array<MultiFab,3>& B_f, MultiFab& E_n, GpuArray<Real,3> dx, Real dt, Real theta, Periodicity period, Real rtol, Real atol) {
   // State vector (lhs); initial state is start of time step
   BE x(E_n.boxArray(), E_n.distributionMap, E_n.n_grow[0], period);
   x.Copy_Bfx(x, B_f[0]);
   x.Copy_Bfy(x, B_f[1]);
   x.Copy_Bfz(x, B_f[2]);
   x.Copy_En(x, E_n);
   
   // rhs; first includes initial B and E so copy x without ghost cells
   BE b = x.copy_dim(0);
   b.Copy(b, x, 0);
   
   // Calculate curls of initial state; no ghost cells needed
   std::array<MultiFab,3> curl_En = curl_n2f(E_n, dx, 0);
   MultiFab curl_Bf = curl_f2n(B_f[0], B_f[1], B_f[2], dx, 0);
   
   b.Saxpy_B(b, curl_En, -dt*(1-theta));
   b.Saxpy_En(b, curl_Bf, square(PhysConst::c)*dt*(1-theta));
   
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
   }
   
   MultiFab::Copy(B_f[0], x.getB_fx(), 0, 0, 1, x.nGrow_Bfx());
   MultiFab::Copy(B_f[1], x.getB_fy(), 0, 0, 1, x.nGrow_Bfy());
   MultiFab::Copy(B_f[2], x.getB_fz(), 0, 0, 1, x.nGrow_Bfz());
   MultiFab::Copy(E_n, x.getE_n(), 0, 0, 3, x.nGrow_En());
}
