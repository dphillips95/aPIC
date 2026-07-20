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
   amrex::MultiFab B_fx;
   amrex::MultiFab B_fy;
   amrex::MultiFab B_fz;
   amrex::MultiFab E_n;
   amrex::Periodicity period;
public:
   BE() {
      
   }
   
   BE(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int nghost, const amrex::Periodicity input_period) {
      B_fx = amrex::MultiFab(convert(ba,AMReXConst::btype_fx),dm,1,nghost);
      B_fy = amrex::MultiFab(convert(ba,AMReXConst::btype_fy),dm,1,nghost);
      B_fz = amrex::MultiFab(convert(ba,AMReXConst::btype_fz),dm,1,nghost);
      E_n = amrex::MultiFab(convert(ba,AMReXConst::btype_n),dm,3,nghost);
      this->setVal(0.0);
      
      period = input_period;
   }
   
   BE copy_dim(int nghost = -1) const {
      // If number of ghost cells is not given (thus set to -1), copy from BE
      if (nghost == -1) {
         nghost = E_n.n_grow[0];
      }
      
      BE new_BE(E_n.boxArray(), E_n.distributionMap, nghost, period);
      
      return new_BE;
   }
   
   const amrex::MultiFab& getB_fx() const {
      return B_fx;
   }
   const amrex::MultiFab& getB_fy() const {
      return B_fy;
   }
   const amrex::MultiFab& getB_fz() const {
      return B_fz;
   }
   const amrex::MultiFab& getE_n() const {
      return E_n;
   }

   // Return number of ghost cells
   int nGrow_Bfx() const { return B_fx.nGrow(); }
   int nGrow_Bfy() const { return B_fy.nGrow(); }
   int nGrow_Bfz() const { return B_fz.nGrow(); }
   int nGrow_En() const { return E_n.nGrow(); }
   
   void Copy_Bfx(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_fx, rhs, 0, 0, 1, rhs.nGrow());
   }
   void Copy_Bfy(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_fy, rhs, 0, 0, 1, rhs.nGrow());
   }
   void Copy_Bfz(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_fz, rhs, 0, 0, 1, rhs.nGrow());
   }
   void Copy_En(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.E_n, rhs, 0, 0, 3, rhs.nGrow());
   }

   void Copy_Bfx(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_fx, rhs, 0, 0, 1, nghost);
   }
   void Copy_Bfy(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_fy, rhs, 0, 0, 1, nghost);
   }
   void Copy_Bfz(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_fz, rhs, 0, 0, 1, nghost);
   }
   void Copy_En(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.E_n, rhs, 0, 0, 3, nghost);
   }
   
   void Copy(BE& lhs, const BE& rhs) {
      BE::Copy_Bfx(lhs, rhs.B_fx);
      BE::Copy_Bfy(lhs, rhs.B_fy);
      BE::Copy_Bfz(lhs, rhs.B_fz);
      BE::Copy_En(lhs, rhs.E_n);
   }
   
   void Copy(BE& lhs, const BE& rhs, int nghost) {
      BE::Copy_Bfx(lhs, rhs.B_fx, nghost);
      BE::Copy_Bfy(lhs, rhs.B_fy, nghost);
      BE::Copy_Bfz(lhs, rhs.B_fz, nghost);
      BE::Copy_En(lhs, rhs.E_n, nghost);
   }

   void apply_BCs() {
      B_fx.FillBoundary(period);
      B_fy.FillBoundary(period);
      B_fz.FillBoundary(period);
      E_n.FillBoundary(period);
   }
   
   amrex::Real dotProduct(const BE& v1, const BE& v2) const {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(v1.B_fx, 0, v2.B_fx, 0, 1, v1.nGrow_Bfx());
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(v1.B_fy, 0, v2.B_fy, 0, 1, v1.nGrow_Bfy());
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(v1.B_fz, 0, v2.B_fz, 0, 1, v1.nGrow_Bfz());
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.E_n, 0, v2.E_n, 0, 3, v1.nGrow_En());
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bfx + dot_Bfy + dot_Bfz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   amrex::Real dotProduct(const BE& v1, const BE& v2, int nghost) const {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(v1.B_fx, 0, v2.B_fx, 0, 1, nghost);
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(v1.B_fy, 0, v2.B_fy, 0, 1, nghost);
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(v1.B_fz, 0, v2.B_fz, 0, 1, nghost);
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.E_n, 0, v2.E_n, 0, 3, nghost);
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bfx + dot_Bfy + dot_Bfz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   void Saxpy_Bfx(BE& lhs, const amrex::MultiFab& rhs_Bfx, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_fx, a, rhs_Bfx, 0, 0, 1, rhs_Bfx.nGrow());
   }
   void Saxpy_Bfy(BE& lhs, const amrex::MultiFab& rhs_Bfy, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_fy, a, rhs_Bfy, 0, 0, 1, rhs_Bfy.nGrow());
   }
   void Saxpy_Bfz(BE& lhs, const amrex::MultiFab& rhs_Bfz, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_fz, a, rhs_Bfz, 0, 0, 1, rhs_Bfz.nGrow());
   }
   void Saxpy_En(BE& lhs, const amrex::MultiFab& rhs_En, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.E_n, a, rhs_En, 0, 0, 3, rhs_En.nGrow());
   }
   // Add all B fields at once
   void Saxpy_B(BE& lhs, const std::array<amrex::MultiFab,3>& rhs_B, amrex::Real a = 1) {
      BE::Saxpy_Bfx(lhs, rhs_B[0], a);
      BE::Saxpy_Bfy(lhs, rhs_B[1], a);
      BE::Saxpy_Bfz(lhs, rhs_B[2], a);
   }
   
   void Saxpy(BE& lhs, const BE& rhs, amrex::Real a = 1) {
      BE::Saxpy_Bfx(lhs, rhs.B_fx, a);
      BE::Saxpy_Bfy(lhs, rhs.B_fy, a);
      BE::Saxpy_Bfz(lhs, rhs.B_fz, a);
      BE::Saxpy_En(lhs, rhs.E_n, a);
   }

   void linComb_Bfx(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfx, amrex::Real b, const amrex::MultiFab& rhs_b_Bfx) {
      amrex::MultiFab::LinComb(lhs.B_fx, a, rhs_a_Bfx, 0, b, rhs_b_Bfx, 0, 0, 1, rhs_a_Bfx.nGrow());
   }
   void linComb_Bfy(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfy, amrex::Real b, const amrex::MultiFab& rhs_b_Bfy) {
      amrex::MultiFab::LinComb(lhs.B_fy, a, rhs_a_Bfy, 0, b, rhs_b_Bfy, 0, 0, 1, rhs_a_Bfy.nGrow());
   }
   void linComb_Bfz(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bfz, amrex::Real b, const amrex::MultiFab& rhs_b_Bfz) {
      amrex::MultiFab::LinComb(lhs.B_fz, a, rhs_a_Bfz, 0, b, rhs_b_Bfz, 0, 0, 1, rhs_a_Bfz.nGrow());
   }
   void linComb_En(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_En, amrex::Real b, const amrex::MultiFab& rhs_b_En) {
      amrex::MultiFab::LinComb(lhs.E_n, a, rhs_a_En, 0, b, rhs_b_En, 0, 0, 3, rhs_a_En.nGrow());
   }
   
   void linComb(BE& lhs, amrex::Real a, const BE& rhs_a, amrex::Real b, const BE& rhs_b) {
      BE::linComb_Bfx(lhs, a, rhs_a.B_fx, b, rhs_b.B_fx);
      BE::linComb_Bfy(lhs, a, rhs_a.B_fy, b, rhs_b.B_fy);
      BE::linComb_Bfz(lhs, a, rhs_a.B_fz, b, rhs_b.B_fz);
      BE::linComb_En(lhs, a, rhs_a.E_n, b, rhs_b.E_n);
   }

   amrex::Real norm2() const {
      return std::sqrt(this->dotProduct((*this),(*this)));
   }

   void mult(amrex::Real fac) {
      B_fx.mult(fac);
      B_fy.mult(fac);
      B_fz.mult(fac);
      E_n.mult(fac);
   }
   
   void setVal(amrex::Real val) {
      B_fx.setVal(val);
      B_fy.setVal(val);
      B_fz.setVal(val);
      E_n.setVal(val);
   }
};

class linop {
private:
   // BoxArray here has no ghost cells
   amrex::BoxArray ba;
   amrex::DistributionMapping dm;
   int nghost;
   amrex::GpuArray<amrex::Real,3> dx;
   amrex::Real tFactor; // time step factor, dt*theta
   amrex::Periodicity period;
public:
   using RT = amrex::Real;
   
   linop(const amrex::BoxArray& input_ba, const amrex::DistributionMapping& input_dm, int input_nghost, const amrex::GpuArray<RT,3>& input_dx, RT input_tFactor, const amrex::Periodicity& input_period) {
      ba = input_ba;
      dm = input_dm;
      nghost = input_nghost;
      dx = input_dx;
      tFactor = input_tFactor;
      period = input_period;
   }
   
   void setBoxArray(const amrex::BoxArray& input_ba) {
      ba = input_ba;
   }
   void setDistributionMapping(const amrex::DistributionMapping& input_dm) {
      dm = input_dm;
   }
   void setNGhost(int input_nghost) {
      nghost = input_nghost;
   }
   void setDx(const amrex::GpuArray<RT,3>& input_dx) {
      dx = input_dx;
   }
   void setTFactor(RT input_tFactor) {
      tFactor = input_tFactor;
   }
   
   // Actual operator matrix product, i.e. x input, Ax output
   void apply(BE& Ax, const BE& x) {
      Ax.Copy(Ax,x,0);
      Ax.Saxpy_B(Ax, curl_n2f(x.getE_n(), this->dx, 0), this->tFactor);
      Ax.Saxpy_En(Ax, curl_f2n(x.getB_fx(), x.getB_fy(), x.getB_fz(), this->dx, 0), -this->tFactor*square(PhysConst::c));
   }
   
   // Assign lhs = rhs
   void assign(BE& lhs, const BE& rhs) {
      lhs.Copy(lhs,rhs);
      lhs.apply_BCs();
   }
   
   // Dot product of v1 and v2
   RT dotProduct(const BE& v1, const BE& v2) {
      return v1.dotProduct(v1,v2);
   }
   
   // lhs += a*rhs
   void increment(BE& lhs, const BE& rhs, RT a) {
      lhs.Saxpy(lhs,rhs,a);
      lhs.apply_BCs();
   }
   
   // lhs = a*rhs_a + b*rhs_b
   void linComb(BE& lhs, RT a, const BE& rhs_a, RT b, const BE& rhs_b) {
      lhs.linComb(lhs,a,rhs_a,b,rhs_b);
      lhs.apply_BCs();
   }
   
   // Return new vector suitable for rhs of Ax = b (i.e. b)
   BE makeVecRHS() { return BE(ba,dm,0,period); }
   
   // Return new vector suitable for lhs of Ax = b (i.e. x)
   BE makeVecLHS() { return BE(ba,dm,nghost,period); }
   
   // 2-norm of v
   RT norm2(const BE& v) { return v.norm2(); }
   
   // Apply right preconditioning, i.e. solve P(lhs) = rhs for preconditioning P
   // P should be an approximation for A
   // for now we use identity, could use A without particles
   void precond(BE& lhs, const BE& rhs) {
      lhs.Copy(lhs,rhs,0);
      lhs.apply_BCs();
   }
   
   // Multiply vector v by factor fac
   void scale(BE& v, RT fac) { v.mult(fac); }
   
   // Set vector to zero
   void setToZero(BE& v) { v.setVal(0.0); }
   
};

void gmres_step(std::array<amrex::MultiFab,3>& B_f, amrex::MultiFab& E_n, amrex::GpuArray<amrex::Real,3> dx, amrex::Real dt, amrex::Real theta, amrex::Periodicity period, amrex::Real rtol, amrex::Real atol);

#endif
