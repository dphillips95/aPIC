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
#include <AMReX_GMRES.H>
#include <AMReX_iMultiFab.H>

#include <cmath>

#include <matrix.h>
#include <constants.h>
#include <operators.h>
#include <math_functions.h>

// GMRES "vector" class; consists of B and E MultiFabs
// Due to staggered grid, face B multiFabs are separated while node E is kept together
class BE {
private:
   amrex::DistributionMapping m_dm;
   int m_nghost;
   amrex::Periodicity m_period;
   amrex::BoxArray m_ba;
   amrex::MultiFab m_B_fx;
   amrex::MultiFab m_B_fy;
   amrex::MultiFab m_B_fz;
   amrex::MultiFab m_E_n;
   std::unique_ptr<amrex::iMultiFab> m_omask_Bfx;
   std::unique_ptr<amrex::iMultiFab> m_omask_Bfy;
   std::unique_ptr<amrex::iMultiFab> m_omask_Bfz;
   std::unique_ptr<amrex::iMultiFab> m_omask_En;
public:
   BE() {}
   
   BE(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int nghost, const amrex::Periodicity period) : m_dm(dm), m_nghost(nghost), m_period(period) {
      m_B_fx = amrex::MultiFab(convert(ba,AMReXConst::btype_fx),dm,1,nghost);
      m_B_fy = amrex::MultiFab(convert(ba,AMReXConst::btype_fy),dm,1,nghost);
      m_B_fz = amrex::MultiFab(convert(ba,AMReXConst::btype_fz),dm,1,nghost);
      m_E_n = amrex::MultiFab(convert(ba,AMReXConst::btype_n),dm,3,nghost);
      this->setVal(0.0);

      m_ba = convert(ba,AMReXConst::btype_c);

      m_omask_Bfx = amrex::OwnerMask(m_B_fx, m_period, this->vectghost());
      m_omask_Bfy = amrex::OwnerMask(m_B_fy, m_period, this->vectghost());
      m_omask_Bfz = amrex::OwnerMask(m_B_fz, m_period, this->vectghost());
      m_omask_En = amrex::OwnerMask(m_E_n, m_period, this->vectghost());
   }

   BE(const BE&) = delete;
   BE& operator=(const BE&) = delete;

   BE(BE&&) = default;
   BE& operator=(BE&&) = default;

   // Create new vector with matching structure
   BE copy_dim() const {
      BE new_BE(m_ba, m_dm, m_nghost, m_period);

      return new_BE;
   }

   // Create new vector with matching structure, with different number of ghost cells
   BE copy_dim(int nghost) const {
      BE new_BE(m_ba, m_dm, nghost, m_period);
      
      return new_BE;
   }
   
   const amrex::MultiFab& getB_fx_const() const {
      return m_B_fx;
   }
   const amrex::MultiFab& getB_fy_const() const {
      return m_B_fy;
   }
   const amrex::MultiFab& getB_fz_const() const {
      return m_B_fz;
   }
   const amrex::MultiFab& getE_n_const() const {
      return m_E_n;
   }

   amrex::MultiFab& getB_fx() {
      return m_B_fx;
   }
   amrex::MultiFab& getB_fy() {
      return m_B_fy;
   }
   amrex::MultiFab& getB_fz() {
      return m_B_fz;
   }
   amrex::MultiFab& getE_n() {
      return m_E_n;
   }
   
   // Return number of ghost cells
   int nghost() const { return m_nghost; }

   // Return ghost cells as IntVect
   amrex::IntVect vectghost() const {
      return amrex::IntVect{m_nghost, m_nghost, m_nghost};
   }
   
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

      // Match "valid" nodal data shared across boxes
      m_B_fx.OverrideSync(m_period);
      m_B_fy.OverrideSync(m_period);
      m_B_fz.OverrideSync(m_period);
      m_E_n.OverrideSync(m_period);
   }
   
   static amrex::Real dotProduct(const BE& v1, const BE& v2) {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(*v1.m_omask_Bfx, v1.m_B_fx, 0, v2.m_B_fx, 0, 1, v1.nghost());
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(*v1.m_omask_Bfy, v1.m_B_fy, 0, v2.m_B_fy, 0, 1, v1.nghost());
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(*v1.m_omask_Bfz, v1.m_B_fz, 0, v2.m_B_fz, 0, 1, v1.nghost());
      amrex::Real dot_En = amrex::MultiFab::Dot(*v1.m_omask_En, v1.m_E_n, 0, v2.m_E_n, 0, 3, v1.nghost());
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bfx + dot_Bfy + dot_Bfz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   static amrex::Real dotProduct(const BE& v1, const BE& v2, int nghost) {
      amrex::Real dot_Bfx = amrex::MultiFab::Dot(*v1.m_omask_Bfx, v1.m_B_fx, 0, v2.m_B_fx, 0, 1, nghost);
      amrex::Real dot_Bfy = amrex::MultiFab::Dot(*v1.m_omask_Bfy, v1.m_B_fy, 0, v2.m_B_fy, 0, 1, nghost);
      amrex::Real dot_Bfz = amrex::MultiFab::Dot(*v1.m_omask_Bfz, v1.m_B_fz, 0, v2.m_B_fz, 0, 1, nghost);
      amrex::Real dot_En = amrex::MultiFab::Dot(*v1.m_omask_En, v1.m_E_n, 0, v2.m_E_n, 0, 3, nghost);
      
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

   // Apply given matrix to B data in x to give E data in Ax (curl B)
   // Note: Data is summed not overwritten
   template <typename T>
   static void apply_matrix_B2E(BE& Ax, const BE& x, const std::array<amrex::LayoutData<T>,3>& matA_B2E, amrex::Real fact) {
      amrex::MultiFab&       Ax_En { Ax.getE_n()       };
      const amrex::MultiFab& x_Bfx { x.getB_fx_const() };
      const amrex::MultiFab& x_Bfy { x.getB_fy_const() };
      const amrex::MultiFab& x_Bfz { x.getB_fz_const() };
      
      for (amrex::MFIter mfi(Ax_En); mfi.isValid(); ++mfi) {
         const T& matA_Bx2E_mf { matA_B2E[0][mfi] };
         const T& matA_By2E_mf { matA_B2E[1][mfi] };
         const T& matA_Bz2E_mf { matA_B2E[2][mfi] };
         
         amrex::FArrayBox&       Ax_En_data { Ax_En[mfi] };
         const amrex::FArrayBox& x_Bfx_data { x_Bfx[mfi] };
         const amrex::FArrayBox& x_Bfy_data { x_Bfy[mfi] };
         const amrex::FArrayBox& x_Bfz_data { x_Bfz[mfi] };
         
         const amrex::IntVect
            len_Ax_En = Ax_En_data.length(),
            len_x_Bfx = x_Bfx_data.length(),
            len_x_Bfy = x_Bfy_data.length(),
            len_x_Bfz = x_Bfz_data.length();
         
         const size_t
            total_Ax_En = math::product(len_Ax_En),
            total_x_Bfx = math::product(len_x_Bfx),
            total_x_Bfy = math::product(len_x_Bfy),
            total_x_Bfz = math::product(len_x_Bfz);
         
         std::span<amrex::Real> Ax_En_span(Ax_En_data.dataPtr(0), 3*total_Ax_En);
         const std::span<const amrex::Real>
            x_Bfx_span(x_Bfx_data.dataPtr(0), total_x_Bfx),
            x_Bfy_span(x_Bfy_data.dataPtr(0), total_x_Bfy),
            x_Bfz_span(x_Bfz_data.dataPtr(0), total_x_Bfz);
         
         // BL_PROFILE_VAR("apply_matrix_B2E::mmult()", TIMER_mmult);
         // std::vector<Real>
         //    tmp_x = matA_Bx2E_mf.mmult(x_Bfx_span),
         //    tmp_y = matA_By2E_mf.mmult(x_Bfy_span),
         //    tmp_z = matA_Bz2E_mf.mmult(x_Bfz_span);
         // BL_PROFILE_VAR_STOP(TIMER_mmult);
         
         // for (size_t ii=0; ii<total_Ax_En; ++ii) {
         //    Ax_Enx_span[ii] += tmp_x[ii]*fact;
         //    Ax_Eny_span[ii] += tmp_y[ii]*fact;
         //    Ax_Enz_span[ii] += tmp_z[ii]*fact;
         // }
         
         BL_PROFILE_VAR("apply_matrix_B2E::mmult_add()", TIMER_mmult_add);
         matA_Bx2E_mf.mmult_add(Ax_En_span, x_Bfx_span, fact);
         matA_By2E_mf.mmult_add(Ax_En_span, x_Bfy_span, fact);
         matA_Bz2E_mf.mmult_add(Ax_En_span, x_Bfz_span, fact);
         BL_PROFILE_VAR_STOP(TIMER_mmult_add);
      }
   }
   
   // Apply given matrix to E data in x to give B data in Ax (curl E)
   // Note: Data is summed not overwritten
   template <typename T>
   static void apply_matrix_E2B(BE& Ax, const BE& x, const std::array<amrex::LayoutData<T>,3>& matA_E2B, amrex::Real fact) {
      amrex::MultiFab&       Ax_Bfx { Ax.getB_fx()     };
      amrex::MultiFab&       Ax_Bfy { Ax.getB_fy()     };
      amrex::MultiFab&       Ax_Bfz { Ax.getB_fz()     };
      const amrex::MultiFab& x_En   { x.getE_n_const() };
      
      for (amrex::MFIter mfi(x_En); mfi.isValid(); ++mfi) {
         const T& matA_E2Bx_mf { matA_E2B[0][mfi] };
         const T& matA_E2By_mf { matA_E2B[1][mfi] };
         const T& matA_E2Bz_mf { matA_E2B[2][mfi] };
         
         amrex::FArrayBox&       Ax_Bfx_data { Ax_Bfx[mfi] };
         amrex::FArrayBox&       Ax_Bfy_data { Ax_Bfy[mfi] };
         amrex::FArrayBox&       Ax_Bfz_data { Ax_Bfz[mfi] };
         const amrex::FArrayBox& x_En_data   { x_En[mfi]   };
         
         const amrex::IntVect
            len_Ax_Bfx = Ax_Bfx_data.length(),
            len_Ax_Bfy = Ax_Bfy_data.length(),
            len_Ax_Bfz = Ax_Bfz_data.length(),
            len_x_En = x_En_data.length();
         
         const size_t
            total_Ax_Bfx = math::product(len_Ax_Bfx),
            total_Ax_Bfy = math::product(len_Ax_Bfy),
            total_Ax_Bfz = math::product(len_Ax_Bfz),
            total_x_En = math::product(len_x_En);
         
         std::span<amrex::Real> Ax_Bfx_span(Ax_Bfx_data.dataPtr(0), total_Ax_Bfx);
         std::span<amrex::Real> Ax_Bfy_span(Ax_Bfy_data.dataPtr(0), total_Ax_Bfy);
         std::span<amrex::Real> Ax_Bfz_span(Ax_Bfz_data.dataPtr(0), total_Ax_Bfz);
         const std::span<const amrex::Real>
            x_En_span(x_En_data.dataPtr(0), 3*total_x_En);
         
         // BL_PROFILE_VAR("apply_matrix_E2B::mmult()", TIMER_mmult);
         // std::vector<Real>
         //    tmp_x = matA_E2Bx_mf.mmult(x_En_span),
         //    tmp_y = matA_E2By_mf.mmult(x_En_span),
         //    tmp_z = matA_E2Bz_mf.mmult(x_En_span);
         // BL_PROFILE_VAR_STOP(TIMER_mmult);
         
         // for (size_t ii=0; ii<total_Ax_Bfx; ++ii) {
         //    Ax_Bfx_span[ii] += tmp_x[ii]*fact;
         // }
         // for (size_t ii=0; ii<total_Ax_Bfx; ++ii) {
         //    Ax_Bfy_span[ii] += tmp_y[ii]*fact;
         // }
         // for (size_t ii=0; ii<total_Ax_Bfx; ++ii) {
         //    Ax_Bfz_span[ii] += tmp_z[ii]*fact;
         // }
         
         BL_PROFILE_VAR("apply_matrix_E2B::mmult_add()", TIMER_mmult_add);
         matA_E2Bx_mf.mmult_add(Ax_Bfx_span, x_En_span, fact);
         matA_E2By_mf.mmult_add(Ax_Bfy_span, x_En_span, fact);
         matA_E2Bz_mf.mmult_add(Ax_Bfz_span, x_En_span, fact);
         BL_PROFILE_VAR_STOP(TIMER_mmult_add);
      }
   }
   
   // template <typename T>
   // static void apply_matrix_E2E(BE& Ax, const BE& x, const amrex::LayoutData<T>& matA_E2E);
};

// GMRES linear operator class
// The "vector" is a class BE consisting of MultiFabs of B and E, and ahandful of class methods for norms etc.
// They cannot be combined into a single MultiFab as they do not share the same grid
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
   
   linop(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int nghost, const amrex::GpuArray<RT,3>& dx, RT tFactor, const amrex::Periodicity& period) : m_ba(convert(ba,AMReXConst::btype_c)), m_dm(dm), m_nghost(nghost), m_dx(dx), m_tFactor(tFactor), m_period(period) {}

   linop(const linop&) = delete;
   linop& operator=(const linop&) = delete;

   linop(linop&&) = default;
   linop& operator=(linop&&) = default;
   
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
      static amrex::MultiFab curl_Bf(convert(m_ba,AMReXConst::btype_n),m_dm, 3, 0);
      static std::array<amrex::MultiFab,3> curl_En = {
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fx),m_dm, 1, 0),
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fy),m_dm, 1, 0),
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fz),m_dm, 1, 0)
      };
      
      BE::Copy(Ax,x,0);

      curl_Bf.setVal(0.0);
      for (int nn=0; nn<3; ++nn) {
         curl_En[nn].setVal(0.0);
      }
      
      BL_PROFILE_VAR("gmres_curl_En()",TIMER_curl_En);
      curl_n2f(curl_En, x.getE_n_const(), m_dx);
      BL_PROFILE_VAR_STOP(TIMER_curl_En);
      
      BL_PROFILE_VAR("gmres_curl_Bf()",TIMER_curl_Bf);
      curl_f2n(curl_Bf, x.getB_fx_const(), x.getB_fy_const(), x.getB_fz_const(), m_dx);
      BL_PROFILE_VAR_STOP(TIMER_curl_Bf);
      
      BE::Saxpy_B(Ax, curl_En, m_tFactor);
      BE::Saxpy_En(Ax, curl_Bf, -m_tFactor*math::square(PhysConst::c));
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
      BE::Saxpy(lhs,rhs,a);
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

// GMRES linear operator class, using matrix representations of operators
// The "vector" is a class BE consisting of MultiFabs of B and E, and ahandful of class methods for norms etc.
// They cannot be combined into a single MultiFab as they do not share the same grid
template <typename T>
class linop_matrix {
private:
   amrex::BoxArray m_ba; // BoxArray here has no ghost cells
   amrex::DistributionMapping m_dm;
   int m_nghost;
   amrex::GpuArray<amrex::Real,3> m_dx;
   amrex::Real m_tFactor; // time step factor, dt*theta
   amrex::Periodicity m_period;
   std::array<amrex::LayoutData<T>,3> m_matA_B2E; // magnetic effect on electric (curl B), each component of B input is separated into different matrices - only changes when BoxArray changes
   std::array<amrex::LayoutData<T>,3> m_matA_E2B; // electric effect on magnetic (curl E), each component of B output is separated into different matrices - only changes when BoxArray changes
   amrex::LayoutData<T> m_matA_E2E; // electric self-interaction, current only (remainder is identity matrix) - this is the only matrix component that changes every time step
public:
   using RT = amrex::Real;
   
   linop_matrix(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int nghost, const amrex::GpuArray<RT,3>& dx, RT tFactor, const amrex::Periodicity& period, const std::array<amrex::LayoutData<T>,3>& matA_B2E, const std::array<amrex::LayoutData<T>,3>& matA_E2B, const amrex::LayoutData<T> matA_E2E) : m_ba(convert(ba,AMReXConst::btype_c)), m_dm(dm), m_nghost(nghost), m_dx(dx), m_tFactor(tFactor), m_period(period), m_matA_B2E(matA_B2E), m_matA_E2B(matA_E2B), m_matA_E2E(matA_E2E) {}

   linop_matrix(const linop_matrix&) = delete;
   linop_matrix& operator=(const linop_matrix&) = delete;

   linop_matrix(linop_matrix&&) = default;
   linop_matrix& operator=(linop_matrix&&) = default;
   
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
      static amrex::MultiFab curl_Bf(convert(m_ba,AMReXConst::btype_n),m_dm, 3, 0);
      static std::array<amrex::MultiFab,3> curl_En = {
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fx),m_dm, 1, 0),
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fy),m_dm, 1, 0),
         amrex::MultiFab(convert(m_ba,AMReXConst::btype_fz),m_dm, 1, 0)
      };
      
      BE::Copy(Ax,x,0);
      
      BL_PROFILE_VAR("apply_matrix_E2B()",TIMER_curl_En);
      BE::apply_matrix_E2B(Ax, x, m_matA_E2B, m_tFactor);
      BL_PROFILE_VAR_STOP(TIMER_curl_En);
      
      BL_PROFILE_VAR("apply_matrix_B2E()",TIMER_curl_Bf);
      BE::apply_matrix_B2E(Ax, x, m_matA_B2E, -m_tFactor*math::square(PhysConst::c));
      BL_PROFILE_VAR_STOP(TIMER_curl_Bf);
      
      // BL_PROFILE_VAR("gmres_current",TIMER_current);
      // apply_matrix_E2E(Ax, x, m_matA_E2E);
      // BL_PROFILE_VAR_STOP(TIMER_current);
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

// Advance B and E fields by solving gmres system using matrices
template <typename T>
void gmres_step_matrix(std::array<amrex::MultiFab,3>& B_f, amrex::MultiFab& E_n, std::array<amrex::LayoutData<T>,3>& matA_B2E, std::array<amrex::LayoutData<T>,3>& matA_E2B, amrex::LayoutData<T>& matA_E2E, amrex::GpuArray<amrex::Real,3> dx, amrex::Real dt, amrex::Real theta, amrex::Periodicity period, amrex::Real rtol, amrex::Real atol, int verbosity) {
   BL_PROFILE("gmres_step()");
   
   // To prevent reallocation every step, state vector BE and curl results are pre-allocated static
   static BE
      x(E_n.boxArray(), E_n.distributionMap, E_n.n_grow[0], period),
      // Ax = x.copy_dim(0),
      b = x.copy_dim(0);
   static amrex::MultiFab curl_Bf(convert(E_n.boxArray(),AMReXConst::btype_n),E_n.distributionMap, 3, 0);
   static std::array<amrex::MultiFab,3> curl_En = {
      amrex::MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fx),E_n.distributionMap, 1, 0),
      amrex::MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fy),E_n.distributionMap, 1, 0),
      amrex::MultiFab(convert(E_n.boxArray(),AMReXConst::btype_fz),E_n.distributionMap, 1, 0)
   };

   x.setVal(0.0);
   // Ax.setVal(0.0);
   b.setVal(0.0);
   for (int ii=0; ii<3; ++ii) {
      curl_En[ii].setVal(0.0);
   }
   curl_Bf.setVal(0.0);

   // State vector (lhs); initial state is start of time step
   
   BE::Copy_Bfx(x, B_f[0]);
   BE::Copy_Bfy(x, B_f[1]);
   BE::Copy_Bfz(x, B_f[2]);
   BE::Copy_En(x, E_n);
   
   // rhs; first includes initial B and E so copy x without ghost cells
   BE::Copy(b, x, 0);
   
   // Calculate curls of initial state; no ghost cells needed
   // curl_n2f(curl_En, E_n, dx);
   // curl_f2n(curl_Bf, B_f, dx);
   
   // BE::Saxpy_B(b, curl_En, -dt*(1-theta));
   // BE::Saxpy_En(b, curl_Bf, math::square(PhysConst::c)*dt*(1-theta));
   
   curl_n2f(b.getB_fx(), b.getB_fy(), b.getB_fz(), E_n, dx, -dt*(1-theta));
   curl_f2n(b.getE_n(), B_f, dx, math::square(PhysConst::c)*dt*(1-theta));
   
   linop_matrix<T> gmres_operator(E_n.boxArray(), E_n.distributionMap, E_n.n_grow[0], dx, dt*theta, period, matA_B2E, matA_E2B, matA_E2E);
   
   amrex::GMRES<BE,linop_matrix<T>> gmres_solver;

   /*
   BE::apply_matrix_E2B(Ax, x, matA_E2B, 1);
   BE::apply_matrix_B2E(Ax, x, matA_B2E, 1);
   
   const amrex::MultiFab& x_En   { x.getE_n_const()   };
   const amrex::MultiFab& x_Bfx  { x.getB_fx_const()  };
   const amrex::MultiFab& x_Bfy  { x.getB_fy_const()  };
   const amrex::MultiFab& x_Bfz  { x.getB_fz_const()  };
   
   const amrex::MultiFab& Ax_En  { Ax.getE_n_const()  };
   const amrex::MultiFab& Ax_Bfx { Ax.getB_fx_const() };
   const amrex::MultiFab& Ax_Bfy { Ax.getB_fy_const() };
   const amrex::MultiFab& Ax_Bfz { Ax.getB_fz_const() };
   
   for (amrex::MFIter mfi(Ax.getE_n_const()); mfi.isValid(); ++mfi) {
      const amrex::FArrayBox& x_En_data   { x_En[mfi]   };
      const amrex::FArrayBox& x_Bfx_data  { x_Bfx[mfi]  };
      const amrex::FArrayBox& x_Bfy_data  { x_Bfy[mfi]  };
      const amrex::FArrayBox& x_Bfz_data  { x_Bfz[mfi]  };
      
      const amrex::FArrayBox& Ax_En_data  { Ax_En[mfi]  };
      const amrex::FArrayBox& Ax_Bfx_data { Ax_Bfx[mfi] };
      const amrex::FArrayBox& Ax_Bfy_data { Ax_Bfy[mfi] };
      const amrex::FArrayBox& Ax_Bfz_data { Ax_Bfz[mfi] };
      
      const sp_matrix<amrex::Real>& matA_Bx2E { matA_B2E[0][mfi] };
      const sp_matrix<amrex::Real>& matA_By2E { matA_B2E[1][mfi] };
      const sp_matrix<amrex::Real>& matA_Bz2E { matA_B2E[2][mfi] };
      const sp_matrix<amrex::Real>& matA_E2Bx { matA_E2B[0][mfi] };
      const sp_matrix<amrex::Real>& matA_E2By { matA_E2B[1][mfi] };
      const sp_matrix<amrex::Real>& matA_E2Bz { matA_E2B[2][mfi] };
      
      matA_Bx2E.print_rowsums("Bx2E");
      matA_By2E.print_rowsums("By2E");
      matA_Bz2E.print_rowsums("Bz2E");
      matA_E2Bx.print_rowsums("E2Bx");
      matA_E2By.print_rowsums("E2By");
      matA_E2Bz.print_rowsums("E2Bz");
      
      const amrex::IntVect
         len_x_En = x_En_data.length(),
         len_x_Bfx = x_Bfx_data.length(),
         len_x_Bfy = x_Bfy_data.length(),
         len_x_Bfz = x_Bfz_data.length(),
         len_Ax_En = Ax_En_data.length(),
         len_Ax_Bfx = Ax_Bfx_data.length(),
         len_Ax_Bfy = Ax_Bfy_data.length(),
         len_Ax_Bfz = Ax_Bfz_data.length();
      
      const size_t
         total_x_En = math::product(len_x_En),
         total_x_Bfx = math::product(len_x_Bfx),
         total_x_Bfy = math::product(len_x_Bfy),
         total_x_Bfz = math::product(len_x_Bfz),
         total_Ax_En = math::product(len_Ax_En),
         total_Ax_Bfx = math::product(len_Ax_Bfx),
         total_Ax_Bfy = math::product(len_Ax_Bfy),
         total_Ax_Bfz = math::product(len_Ax_Bfz);
      
      const std::span<const amrex::Real>
         x_Enx_span(x_En_data.dataPtr(0), total_x_En),
         x_Eny_span(x_En_data.dataPtr(1), total_x_En),
         x_Enz_span(x_En_data.dataPtr(2), total_x_En),
         x_Bfx_span(x_Bfx_data.dataPtr(0), total_x_Bfx),
         x_Bfy_span(x_Bfy_data.dataPtr(0), total_x_Bfy),
         x_Bfz_span(x_Bfz_data.dataPtr(0), total_x_Bfz),
         Ax_Enx_span(Ax_En_data.dataPtr(0), total_Ax_En),
         Ax_Eny_span(Ax_En_data.dataPtr(1), total_Ax_En),
         Ax_Enz_span(Ax_En_data.dataPtr(2), total_Ax_En),
         Ax_Bfx_span(Ax_Bfx_data.dataPtr(0), total_Ax_Bfx),
         Ax_Bfy_span(Ax_Bfy_data.dataPtr(0), total_Ax_Bfy),
         Ax_Bfz_span(Ax_Bfz_data.dataPtr(0), total_Ax_Bfz);
      
      amrex::Print() << Ax_Enx_span[0] << std::endl;
   }
   */
   
   gmres_solver.define(gmres_operator);
   gmres_solver.setVerbose(verbosity);
   gmres_solver.solve(x, b, rtol, atol);
   
   int gmres_status = gmres_solver.getStatus();
   
   if (gmres_status > 0) {
      amrex::Print() << std::endl << "GMRES failed to converge!" << std::endl
                     << "Iteration count: " << gmres_solver.getNumIters() << std::endl
                     << "Residual norm: " << gmres_solver.getResidualNorm() << std::endl << std::endl;
   } else {
      amrex::Print() << "GMRES Iteration count: " << gmres_solver.getNumIters() << std::endl;
   }
   
   amrex::MultiFab::Copy(B_f[0], x.getB_fx(), 0, 0, 1, x.nghost());
   amrex::MultiFab::Copy(B_f[1], x.getB_fy(), 0, 0, 1, x.nghost());
   amrex::MultiFab::Copy(B_f[2], x.getB_fz(), 0, 0, 1, x.nghost());
   amrex::MultiFab::Copy(E_n, x.getE_n(), 0, 0, 3, x.nghost());
}

void gmres_step(std::array<amrex::MultiFab,3>& B_f, amrex::MultiFab& E_n, amrex::GpuArray<amrex::Real,3> dx, amrex::Real dt, amrex::Real theta, amrex::Periodicity period, amrex::Real rtol, amrex::Real atol, int verbosity = 0);

void get_curl_f2n_operator(matrix<amrex::Real>& curl_x, matrix<amrex::Real>& curl_y, matrix<amrex::Real>& curl_z, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);
                           
inline void get_curl_f2n_operator(std::array<matrix<amrex::Real>,3>& curl, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   get_curl_f2n_operator(curl[0], curl[1], curl[2], bx, nghost, dx);
}

inline std::array<matrix<amrex::Real>,3> get_curl_f2n_operator(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   const amrex::Box&
      bx_fx = grow(convert(bx,AMReXConst::btype_fx), nghost),
      bx_fy = grow(convert(bx,AMReXConst::btype_fy), nghost),
      bx_fz = grow(convert(bx,AMReXConst::btype_fz), nghost),
      bx_n = convert(bx,AMReXConst::btype_n);
   
   const amrex::IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_n = math::product(len_n);
   
   std::array<matrix<amrex::Real>,3> ret = {
      matrix<amrex::Real>(3*total_n, total_fx, 0.0),
      matrix<amrex::Real>(3*total_n, total_fy, 0.0),
      matrix<amrex::Real>(3*total_n, total_fz, 0.0)
   };
   
   get_curl_f2n_operator(ret, bx, nghost, dx);
   
   return ret;
}

void get_curl_n2f_operator(matrix<amrex::Real>& curl_x, matrix<amrex::Real>& curl_y, matrix<amrex::Real>& curl_z, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);

inline void get_curl_n2f_operator(std::array<matrix<amrex::Real>,3>& curl, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   get_curl_n2f_operator(curl[0], curl[1], curl[2], bx, nghost, dx);
}

inline std::array<matrix<amrex::Real>,3> get_curl_n2f_operator(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   const amrex::Box&
      bx_fx = convert(bx,AMReXConst::btype_fx),
      bx_fy = convert(bx,AMReXConst::btype_fy),
      bx_fz = convert(bx,AMReXConst::btype_fz),
      bx_n = grow(convert(bx,AMReXConst::btype_n), nghost);
   
   const amrex::IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_n = math::product(len_n);
   
   std::array<matrix<amrex::Real>,3> ret = {
      matrix<amrex::Real>(total_fx, 3*total_n, 0.0),
      matrix<amrex::Real>(total_fy, 3*total_n, 0.0),
      matrix<amrex::Real>(total_fz, 3*total_n, 0.0)
   };
   
   get_curl_n2f_operator(ret, bx, nghost, dx);
   
   return ret;
}

void get_curl_f2n_operator_sparse(sp_matrix<amrex::Real>& curl_x, sp_matrix<amrex::Real>& curl_y, sp_matrix<amrex::Real>& curl_z, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);

inline void get_curl_f2n_operator_sparse(std::array<sp_matrix<amrex::Real>,3>& curl, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   get_curl_f2n_operator_sparse(curl[0], curl[1], curl[2], bx, nghost, dx);
}

inline std::array<sp_matrix<amrex::Real>,3> get_curl_f2n_operator_sparse(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   const amrex::Box&
      bx_fx = grow(convert(bx,AMReXConst::btype_fx), nghost),
      bx_fy = grow(convert(bx,AMReXConst::btype_fy), nghost),
      bx_fz = grow(convert(bx,AMReXConst::btype_fz), nghost),
      bx_n = convert(bx,AMReXConst::btype_n);
   
   const amrex::IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_n = math::product(len_n);

   constexpr int cols_per_row = 4;
   
   std::array<sp_matrix<amrex::Real>,3> ret = {
      sp_matrix<amrex::Real>(2*cols_per_row*total_n, 3*total_n, total_fx),
      sp_matrix<amrex::Real>(2*cols_per_row*total_n, 3*total_n, total_fy),
      sp_matrix<amrex::Real>(2*cols_per_row*total_n, 3*total_n, total_fz)
   };

   get_curl_f2n_operator_sparse(ret, bx, nghost, dx);
   
   return ret;
}

void get_curl_n2f_operator_sparse(sp_matrix<amrex::Real>& curl_x, sp_matrix<amrex::Real>& curl_y, sp_matrix<amrex::Real>& curl_z, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx);

inline void get_curl_n2f_operator_sparse(std::array<sp_matrix<amrex::Real>,3>& curl, const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   get_curl_n2f_operator_sparse(curl[0], curl[1], curl[2], bx, nghost, dx);
}

inline std::array<sp_matrix<amrex::Real>,3> get_curl_n2f_operator_sparse(const amrex::Box& bx, const int nghost, const amrex::GpuArray<amrex::Real,3>& dx) {
   const amrex::Box&
      bx_fx = convert(bx,AMReXConst::btype_fx),
      bx_fy = convert(bx,AMReXConst::btype_fy),
      bx_fz = convert(bx,AMReXConst::btype_fz),
      bx_n = grow(convert(bx,AMReXConst::btype_n), nghost);
   
   const amrex::IntVect
      len_fx = bx_fx.length(),
      len_fy = bx_fy.length(),
      len_fz = bx_fz.length(),
      len_n = bx_n.length();
   
   const int
      total_fx = math::product(len_fx),
      total_fy = math::product(len_fy),
      total_fz = math::product(len_fz),
      total_n = math::product(len_n);

   constexpr int cols_per_row = 8;
   
   std::array<sp_matrix<amrex::Real>,3> ret = {
      sp_matrix<amrex::Real>(cols_per_row*total_fx, total_fx, 3*total_n),
      sp_matrix<amrex::Real>(cols_per_row*total_fy, total_fy, 3*total_n),
      sp_matrix<amrex::Real>(cols_per_row*total_fz, total_fz, 3*total_n)
   };

   get_curl_n2f_operator_sparse(ret, bx, nghost, dx);
   
   return ret;
}

// Get cell (or node etc.) ID from given cell indices of cell in box
// If the box has ghost cells then index needs to be shifted to accomodate
inline int get_cellID(int x_index, int y_index, int z_index, const amrex::IntVect& len, int nghost = 0) {
   x_index += nghost;
   y_index += nghost;
   z_index += nghost;
   return (z_index*len[1] + y_index)*len[0] + x_index;
}
// Get cell (or node etc.) ID from given cell indices of cell in box
// If the box has ghost cells then index needs to be shifted to accomodate
inline int get_cellID(amrex::IntVect cell_indices, const amrex::IntVect& len, int nghost = 0) {
   cell_indices += nghost;
   return get_cellID(cell_indices[0], cell_indices[1], cell_indices[2], len);
}

inline amrex::IntVect get_cell_indices(int cellID, const amrex::IntVect& len, int nghost = 0) {
   int
      x_index = cellID%len[0]          - nghost,
      y_index = (cellID/len[0])%len[1] - nghost,
      z_index = cellID/(len[0]*len[1]) - nghost;
   return amrex::IntVect(x_index,y_index,z_index);
}

#endif
