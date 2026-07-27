/*
Header for GMRES, containing linear operator and vector classes for aPIC.

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

#ifndef GMRES_H_
#define GMRES_H_

#include <AMReX_REAL.H>
#include <AMReX_Print.H>

#include <cmath>

#include <matrix.h>
#include <constants.h>
#include <operators.h>
#include <math_functions.h>

// GMRES linear operator class
// The "vector" is a class BE consisting of MultiFabs of B and E, and ahandful of class methods for norms etc.
// They cannot be combined into a single MultiFab as they do not share the same grid

// GMRES "vector" class; consists of B and E MultiFabs
// Due to staggered grid, face B multiFabs are separate while node E is kept together
class BE {
private:
   amrex::BoxArray m_ba;
   amrex::DistributionMapping m_dm;
   amrex::MultiFab m_B_fx;
   amrex::MultiFab m_B_fy;
   amrex::MultiFab m_B_fz;
   amrex::MultiFab m_E_n;
   amrex::Periodicity m_period;
   int m_nghost;
public:
   BE() {
      
   }
   
   BE(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int nghost, const amrex::Periodicity period) {
      m_B_fx = amrex::MultiFab(convert(ba,AMReXConst::btype_fx),dm,1,nghost);
      m_B_fy = amrex::MultiFab(convert(ba,AMReXConst::btype_fy),dm,1,nghost);
      m_B_fz = amrex::MultiFab(convert(ba,AMReXConst::btype_fz),dm,1,nghost);
      m_E_n = amrex::MultiFab(convert(ba,AMReXConst::btype_n),dm,3,nghost);
      this->setVal(0.0);

      m_ba = convert(ba,AMReXConst::btype_c);
      m_dm = dm;
      m_nghost = nghost;
      m_period = period;
   }
   
   BE copy_dim(int nghost = -1) const {
      // If number of ghost cells is not given (thus set to -1), copy from BE
      if (nghost == -1) {
         nghost = m_nghost;
      }
      
      BE new_BE(m_ba, m_dm, nghost, m_period);
      
      return new_BE;
   }
   
   const amrex::MultiFab& getB_fx() const {
      return m_B_fx;
   }
   const amrex::MultiFab& getB_fy() const {
      return m_B_fy;
   }
   const amrex::MultiFab& getB_fz() const {
      return m_B_fz;
   }
   const amrex::MultiFab& getE_n() const {
      return m_E_n;
   }

   // Return number of ghost cells
   int nGrow_Bfx() const { return m_B_fx.nGrow(); }
   int nGrow_Bfy() const { return m_B_fy.nGrow(); }
   int nGrow_Bfz() const { return m_B_fz.nGrow(); }
   int nGrow_En() const { return m_E_n.nGrow(); }
   
   static void Copy_Bfx(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.m_B_fx, rhs, 0, 0, 1, rhs.nGrow());
   }
   static void Copy_Bfy(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.m_B_fy, rhs, 0, 0, 1, rhs.nGrow());
   }
   static void Copy_Bfz(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.m_B_fz, rhs, 0, 0, 1, rhs.nGrow());
   }
   static void Copy_En(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.m_E_n, rhs, 0, 0, 3, rhs.nGrow());
   }

   static void Copy_Bfx(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.m_B_fx, rhs, 0, 0, 1, nghost);
   }
   static void Copy_Bfy(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.m_B_fy, rhs, 0, 0, 1, nghost);
   }
   static void Copy_Bfz(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.m_B_fz, rhs, 0, 0, 1, nghost);
   }
   static void Copy_En(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.m_E_n, rhs, 0, 0, 3, nghost);
   }
   
   static void Copy(BE& lhs, const BE& rhs) {
      BE::Copy_Bfx(lhs, rhs.m_B_fx);
      BE::Copy_Bfy(lhs, rhs.m_B_fy);
      BE::Copy_Bfz(lhs, rhs.m_B_fz);
      BE::Copy_En(lhs, rhs.m_E_n);
   }
   
   static void Copy(BE& lhs, const BE& rhs, int nghost) {
      BE::Copy_Bfx(lhs, rhs.m_B_fx, nghost);
      BE::Copy_Bfy(lhs, rhs.m_B_fy, nghost);
      BE::Copy_Bfz(lhs, rhs.m_B_fz, nghost);
      BE::Copy_En(lhs, rhs.m_E_n, nghost);
   }

   void apply_BCs() {
      m_B_fx.FillBoundary(m_period);
      m_B_fy.FillBoundary(m_period);
      m_B_fz.FillBoundary(m_period);
      m_E_n.FillBoundary(m_period);
   }
   
   static amrex::Real dotProduct(const BE& v1, const BE& v2) {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(v1.m_B_fx, 0, v2.m_B_fx, 0, 1, v1.nGrow_Bfx());
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(v1.m_B_fy, 0, v2.m_B_fy, 0, 1, v1.nGrow_Bfy());
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(v1.m_B_fz, 0, v2.m_B_fz, 0, 1, v1.nGrow_Bfz());
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.m_E_n, 0, v2.m_E_n, 0, 3, v1.nGrow_En());
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bfx + dot_Bfy + dot_Bfz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   static amrex::Real dotProduct(const BE& v1, const BE& v2, int nghost) {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(v1.m_B_fx, 0, v2.m_B_fx, 0, 1, nghost);
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(v1.m_B_fy, 0, v2.m_B_fy, 0, 1, nghost);
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(v1.m_B_fz, 0, v2.m_B_fz, 0, 1, nghost);
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.m_E_n, 0, v2.m_E_n, 0, 3, nghost);
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bfx + dot_Bfy + dot_Bfz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   static void Saxpy_Bfx(BE& lhs, const amrex::MultiFab& rhs_Bfx, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.m_B_fx, a, rhs_Bfx, 0, 0, 1, rhs_Bfx.nGrow());
   }
   static void Saxpy_Bfy(BE& lhs, const amrex::MultiFab& rhs_Bfy, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.m_B_fy, a, rhs_Bfy, 0, 0, 1, rhs_Bfy.nGrow());
   }
   static void Saxpy_Bfz(BE& lhs, const amrex::MultiFab& rhs_Bfz, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.m_B_fz, a, rhs_Bfz, 0, 0, 1, rhs_Bfz.nGrow());
   }
   static void Saxpy_En(BE& lhs, const amrex::MultiFab& rhs_En, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.m_E_n, a, rhs_En, 0, 0, 3, rhs_En.nGrow());
   }
   // Add all B fields at once
   static void Saxpy_B(BE& lhs, const std::array<amrex::MultiFab,3>& B_f, amrex::Real a = 1) {
      BE::Saxpy_Bfx(lhs, B_f[0], a);
      BE::Saxpy_Bfy(lhs, B_f[1], a);
      BE::Saxpy_Bfz(lhs, B_f[2], a);
   }
   
   static void Saxpy(BE& lhs, const BE& rhs, amrex::Real a = 1) {
      BE::Saxpy_Bfx(lhs, rhs.m_B_fx, a);
      BE::Saxpy_Bfy(lhs, rhs.m_B_fy, a);
      BE::Saxpy_Bfz(lhs, rhs.m_B_fz, a);
      BE::Saxpy_En(lhs, rhs.m_E_n, a);
   }

   static void linComb_Bfx(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfx, amrex::Real b, const amrex::MultiFab& rhs_b_Bfx) {
      amrex::MultiFab::LinComb(lhs.m_B_fx, a, rhs_a_Bfx, 0, b, rhs_b_Bfx, 0, 0, 1, rhs_a_Bfx.nGrow());
   }
   static void linComb_Bfy(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfy, amrex::Real b, const amrex::MultiFab& rhs_b_Bfy) {
      amrex::MultiFab::LinComb(lhs.m_B_fy, a, rhs_a_Bfy, 0, b, rhs_b_Bfy, 0, 0, 1, rhs_a_Bfy.nGrow());
   }
   static void linComb_Bfz(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfz, amrex::Real b, const amrex::MultiFab& rhs_b_Bfz) {
      amrex::MultiFab::LinComb(lhs.m_B_fz, a, rhs_a_Bfz, 0, b, rhs_b_Bfz, 0, 0, 1, rhs_a_Bfz.nGrow());
   }
   static void linComb_En(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_En, amrex::Real b, const amrex::MultiFab& rhs_b_En) {
      amrex::MultiFab::LinComb(lhs.m_E_n, a, rhs_a_En, 0, b, rhs_b_En, 0, 0, 3, rhs_a_En.nGrow());
   }
   
   static void linComb(BE& lhs, amrex::Real a, const BE& rhs_a, amrex::Real b, const BE& rhs_b) {
      BE::linComb_Bfx(lhs, a, rhs_a.m_B_fx, b, rhs_b.m_B_fx);
      BE::linComb_Bfy(lhs, a, rhs_a.m_B_fy, b, rhs_b.m_B_fy);
      BE::linComb_Bfz(lhs, a, rhs_a.m_B_fz, b, rhs_b.m_B_fz);
      BE::linComb_En(lhs, a, rhs_a.m_E_n, b, rhs_b.m_E_n);
   }

   amrex::Real norm2() const {
      return std::sqrt(this->dotProduct((*this),(*this)));
   }

   void mult(amrex::Real fac) {
      m_B_fx.mult(fac);
      m_B_fy.mult(fac);
      m_B_fz.mult(fac);
      m_E_n.mult(fac);
   }
   
   void setVal(amrex::Real val) {
      m_B_fx.setVal(val);
      m_B_fy.setVal(val);
      m_B_fz.setVal(val);
      m_E_n.setVal(val);
   }
};

class linop {
private:
   // BoxArray here has no ghost cells
   amrex::BoxArray m_ba;
   amrex::DistributionMapping m_dm;
   int m_nghost;
   amrex::GpuArray<amrex::Real,3> m_dx;
   amrex::Real m_tFactor; // time step factor, dt*theta
   amrex::Periodicity m_period;
public:
   using RT = amrex::Real;
   
   linop(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int nghost, const amrex::GpuArray<RT,3>& dx, RT tFactor, const amrex::Periodicity& period) {
      m_ba = convert(ba,AMReXConst::btype_c);
      m_dm = dm;
      m_nghost = nghost;
      m_dx = dx;
      m_tFactor = tFactor;
      m_period = period;
   }
   
   void setBoxArray(const amrex::BoxArray& ba) {
      m_ba = convert(ba,AMReXConst::btype_c);
   }
   void setDistributionMapping(const amrex::DistributionMapping& dm) {
      m_dm = dm;
   }
   void setNGhost(int nghost) {
      m_nghost = nghost;
   }
   void setDx(const amrex::GpuArray<RT,3>& dx) {
      m_dx = dx;
   }
   void setTFactor(RT tFactor) {
      m_tFactor = tFactor;
   }
   
   // Actual operator matrix product, i.e. x input, Ax output
   void apply(BE& Ax, const BE& x) {
      BL_PROFILE("gmres_apply()");
      static amrex::MultiFab curl_Bf(convert(x.getE_n().boxArray(),AMReXConst::btype_n),x.getE_n().distributionMap, 3, 0);
      static std::array<amrex::MultiFab,3> curl_En = {
         amrex::MultiFab(convert(x.getE_n().boxArray(),AMReXConst::btype_fx),x.getE_n().distributionMap, 1, 0),
         amrex::MultiFab(convert(x.getE_n().boxArray(),AMReXConst::btype_fy),x.getE_n().distributionMap, 1, 0),
         amrex::MultiFab(convert(x.getE_n().boxArray(),AMReXConst::btype_fz),x.getE_n().distributionMap, 1, 0)
      };
      
      BE::Copy(Ax,x,0);

      BL_PROFILE_VAR("gmres_curl_En()",TIMER_curl_En);
      curl_n2f(curl_En, x.getE_n(), m_dx);
      BL_PROFILE_VAR_STOP(TIMER_curl_En);
      BL_PROFILE_VAR("gmres_curl_Bf()",TIMER_curl_Bf);
      curl_f2n(curl_Bf, x.getB_fx(), x.getB_fy(), x.getB_fz(), m_dx);
      BL_PROFILE_VAR_STOP(TIMER_curl_Bf);
      
      Ax.Saxpy_B(Ax, curl_En, m_tFactor);
      Ax.Saxpy_En(Ax, curl_Bf, -m_tFactor*math::square(PhysConst::c));
   }
   
   // Assign lhs = rhs
   static void assign(BE& lhs, const BE& rhs) {
      BE::Copy(lhs,rhs);
      lhs.apply_BCs();
   }
   
   // Dot product of v1 and v2
   static RT dotProduct(const BE& v1, const BE& v2) {
      return BE::dotProduct(v1,v2);
   }
   
   // lhs += a*rhs
   static void increment(BE& lhs, const BE& rhs, RT a) {
      lhs.Saxpy(lhs,rhs,a);
      lhs.apply_BCs();
   }
   
   // lhs = a*rhs_a + b*rhs_b
   static void linComb(BE& lhs, RT a, const BE& rhs_a, RT b, const BE& rhs_b) {
      BE::linComb(lhs,a,rhs_a,b,rhs_b);
      lhs.apply_BCs();
   }
   
   // Return new vector suitable for rhs of Ax = b (i.e. b)
   BE makeVecRHS() { return BE(m_ba,m_dm,0,m_period); }
   
   // Return new vector suitable for lhs of Ax = b (i.e. x)
   BE makeVecLHS() { return BE(m_ba,m_dm,m_nghost,m_period); }
   
   // 2-norm of v
   static RT norm2(const BE& v) { return v.norm2(); }
   
   // Apply right preconditioning, i.e. solve P(lhs) = rhs for preconditioning P
   // P should be an approximation for A
   // for now we use identity, could use A without particles
   static void precond(BE& lhs, const BE& rhs) {
      BE::Copy(lhs,rhs,0);
      lhs.apply_BCs();
   }
   
   // Multiply vector v by factor fac
   static void scale(BE& v, RT fac) { v.mult(fac); }
   
   // Set vector to zero
   static void setToZero(BE& v) { v.setVal(0.0); }
   
};

void gmres_step(std::array<amrex::MultiFab,3>& B_f, amrex::MultiFab& E_n, amrex::GpuArray<amrex::Real,3> dx, amrex::Real dt, amrex::Real theta, amrex::Periodicity period, amrex::Real rtol, amrex::Real atol);

matrix<amrex::Real> get_curl_f2n_operator(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);
matrix<amrex::Real> get_curl_n2f_operator(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);

int get_cellID(const amrex::IntVect& cell_indices, const amrex::IntVect& len);
int get_cellID(const std::array<int,3>& cell_indices, const amrex::IntVect& len);
amrex::IntVect get_cell_indices(int cellID, const amrex::IntVect& len);

#endif
