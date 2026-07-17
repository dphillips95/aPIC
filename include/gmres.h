#ifndef GMRES_H_
#define GMRES_H_

#include <AMReX_REAL.H>

#include <cmath>

#include <constants.h>
#include <operators.h>
#include <math.h>

// GMRES linear operator class
// The "vector" is a class BE consisting of MultiFabs of B and E, and ahandful of class methods for norms etc.
// They cannot be combined into a single MultiFab as they do not share the same grid

// GMRES "vector" class; consists of B and E MultiFabs
// Due to staggered grid, face B multiFabs are separate while node E is kept together
class BE {
private:
   amrex::MultiFab B_x;
   amrex::MultiFab B_y;
   amrex::MultiFab B_z;
   amrex::MultiFab E_n;
public:
   BE(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const int nghost) {
      B_x = amrex::MultiFab(convert(ba,AMReXConst::btype_fx),dm,1,nghost);
      B_y = amrex::MultiFab(convert(ba,AMReXConst::btype_fy),dm,1,nghost);
      B_z = amrex::MultiFab(convert(ba,AMReXConst::btype_fz),dm,1,nghost);
      E_n = amrex::MultiFab(convert(ba,AMReXConst::btype_n),dm,3,nghost);
      this->setVal(0.0);
   }
   
   const amrex::MultiFab& getB_x() const {
      return B_x;
   }
   const amrex::MultiFab& getB_y() const {
      return B_y;
   }
   const amrex::MultiFab& getB_z() const {
      return B_z;
   }
   const amrex::MultiFab& getE_n() const {
      return E_n;
   }

   // Return number of ghost cells
   int nGrow_Bx() const { return B_x.nGrow(); }
   int nGrow_By() const { return B_y.nGrow(); }
   int nGrow_Bz() const { return B_z.nGrow(); }
   int nGrow_En() const { return E_n.nGrow(); }
   
   void Copy_Bx(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_x, rhs, 0, 0, 1, lhs.nGrow_Bx());
   }
   void Copy_By(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_y, rhs, 0, 0, 1, lhs.nGrow_By());
   }
   void Copy_Bz(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.B_z, rhs, 0, 0, 1, lhs.nGrow_Bz());
   }
   void Copy_En(BE& lhs, const amrex::MultiFab& rhs) {
      amrex::MultiFab::Copy(lhs.E_n, rhs, 0, 0, 3, lhs.nGrow_En());
   }

   void Copy_Bx(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_x, rhs, 0, 0, 1, nghost);
   }
   void Copy_By(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_y, rhs, 0, 0, 1, nghost);
   }
   void Copy_Bz(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.B_z, rhs, 0, 0, 1, nghost);
   }
   void Copy_En(BE& lhs, const amrex::MultiFab& rhs, int nghost) {
      amrex::MultiFab::Copy(lhs.E_n, rhs, 0, 0, 3, nghost);
   }
   
   void Copy(BE& lhs, const BE& rhs) {
      BE::Copy_Bx(lhs, rhs.B_x);
      BE::Copy_By(lhs, rhs.B_y);
      BE::Copy_Bz(lhs, rhs.B_z);
      BE::Copy_En(lhs, rhs.E_n);
   }
   
   void Copy(BE& lhs, const BE& rhs, int nghost) {
      BE::Copy_Bx(lhs, rhs.B_x, nghost);
      BE::Copy_By(lhs, rhs.B_y, nghost);
      BE::Copy_Bz(lhs, rhs.B_z, nghost);
      BE::Copy_En(lhs, rhs.E_n, nghost);
   }

   amrex::Real dotProduct(const BE& v1, const BE& v2) const {
      amrex::Real dot_Bx = amrex::MultiFab::Dot(v1.B_x, 0, v2.B_x, 0, 1, v1.nGrow_Bx());
      amrex::Real dot_By = amrex::MultiFab::Dot(v1.B_y, 0, v2.B_y, 0, 1, v1.nGrow_By());
      amrex::Real dot_Bz = amrex::MultiFab::Dot(v1.B_z, 0, v2.B_z, 0, 1, v1.nGrow_Bz());
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.E_n, 0, v2.E_n, 0, 3, v1.nGrow_En());
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bx + dot_By + dot_Bz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   amrex::Real dotProduct(const BE& v1, const BE& v2, int nghost) const {
      amrex::Real dot_Bx = amrex::MultiFab::Dot(v1.B_x, 0, v2.B_x, 0, 1, nghost);
      amrex::Real dot_By = amrex::MultiFab::Dot(v1.B_y, 0, v2.B_y, 0, 1, nghost);
      amrex::Real dot_Bz = amrex::MultiFab::Dot(v1.B_z, 0, v2.B_z, 0, 1, nghost);
      amrex::Real dot_En = amrex::MultiFab::Dot(v1.E_n, 0, v2.E_n, 0, 3, nghost);
      
      // Rescale B and E with mu0 and eps0 to avoid bias
      return (dot_Bx + dot_By + dot_Bz)/PhysConst::mu0 + dot_En*PhysConst::eps0;
   }

   void Saxpy_Bx(BE& lhs, const amrex::MultiFab& rhs_Bx, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_x, a, rhs_Bx, 0, 0, 1, lhs.nGrow_Bx());
   }
   void Saxpy_By(BE& lhs, const amrex::MultiFab& rhs_By, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_y, a, rhs_By, 0, 0, 1, lhs.nGrow_By());
   }
   void Saxpy_Bz(BE& lhs, const amrex::MultiFab& rhs_Bz, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.B_z, a, rhs_Bz, 0, 0, 1, lhs.nGrow_Bz());
   }
   void Saxpy_En(BE& lhs, const amrex::MultiFab& rhs_En, amrex::Real a = 1) {
      amrex::MultiFab::Saxpy(lhs.E_n, a, rhs_En, 0, 0, 1, lhs.nGrow_En());
   }
   // Add all B fields at once
   void Saxpy_B(BE& lhs, const std::array<amrex::MultiFab,3>& rhs_B, amrex::Real a = 1) {
      BE::Saxpy_Bx(lhs, rhs_B[0], a);
      BE::Saxpy_By(lhs, rhs_B[1], a);
      BE::Saxpy_Bz(lhs, rhs_B[2], a);
   }
   
   void Saxpy(BE& lhs, const BE& rhs, amrex::Real a = 1) {
      BE::Saxpy_Bx(lhs, rhs.B_x, a);
      BE::Saxpy_By(lhs, rhs.B_y, a);
      BE::Saxpy_Bz(lhs, rhs.B_z, a);
      BE::Saxpy_En(lhs, rhs.E_n, a);
   }

   void linComb_Bx(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bx, amrex::Real b, const amrex::MultiFab& rhs_b_Bx) {
      amrex::MultiFab::LinComb(lhs.B_x, a, rhs_a_Bx, 0, b, rhs_b_Bx, 0, 0, 1, lhs.nGrow_Bx());
   }
   void linComb_By(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_By, amrex::Real b, const amrex::MultiFab& rhs_b_By) {
      amrex::MultiFab::LinComb(lhs.B_y, a, rhs_a_By, 0, b, rhs_b_By, 0, 0, 1, lhs.nGrow_By());
   }
   void linComb_Bz(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_Bz, amrex::Real b, const amrex::MultiFab& rhs_b_Bz) {
      amrex::MultiFab::LinComb(lhs.B_z, a, rhs_a_Bz, 0, b, rhs_b_Bz, 0, 0, 1, lhs.nGrow_Bz());
   }
   void linComb_En(BE& lhs, amrex::Real a, const amrex::MultiFab& rhs_a_En, amrex::Real b, const amrex::MultiFab& rhs_b_En) {
      amrex::MultiFab::LinComb(lhs.E_n, a, rhs_a_En, 0, b, rhs_b_En, 0, 0, 1, lhs.nGrow_En());
   }
   
   void linComb(BE& lhs, amrex::Real a, const BE& rhs_a, amrex::Real b, const BE& rhs_b) {
      BE::linComb_Bx(lhs, a, rhs_a.B_x, b, rhs_b.B_x);
      BE::linComb_By(lhs, a, rhs_a.B_y, b, rhs_b.B_y);
      BE::linComb_Bz(lhs, a, rhs_a.B_z, b, rhs_b.B_z);
      BE::linComb_En(lhs, a, rhs_a.E_n, b, rhs_b.E_n);
   }

   amrex::Real norm2() const {
      return std::sqrt(this->dotProduct((*this),(*this)));
   }

   void mult(amrex::Real fac) {
      B_x.mult(fac);
      B_y.mult(fac);
      B_z.mult(fac);
      E_n.mult(fac);
   }
   
   void setVal(amrex::Real val) {
      B_x.setVal(val);
      B_y.setVal(val);
      B_z.setVal(val);
      E_n.setVal(val);
   }
};

class linop {
private:
   // BoxArray here has no ghost cells
   amrex::BoxArray* ba;
   amrex::DistributionMapping* dm;
   int nghost;
   amrex::GpuArray<amrex::Real,3> dx;
   amrex::Real tFactor;
public:
   linop(amrex::BoxArray& input_ba, amrex::DistributionMapping input_dm, int input_nghost, const amrex::GpuArray<amrex::Real,3>& input_dx, amrex::Real input_tFactor) {
      ba = &input_ba;
      dm = &input_dm;
      nghost = input_nghost;
      dx = input_dx;
      tFactor = input_tFactor;
   }

   void setBoxArray(amrex::BoxArray& input_ba) {
      ba = &input_ba;
   }
   void setDistributionMapping(amrex::DistributionMapping& input_dm) {
      dm = &input_dm;
   }
   void setNGhost(int input_nghost) {
      nghost = input_nghost;
   }
   void setDx(amrex::GpuArray<amrex::Real,3>& input_dx) {
      dx = input_dx;
   }
   void setTFactor(amrex::Real input_tFactor) {
      tFactor = input_tFactor;
   }
   
   // Actual operator matrix product, i.e. x input, Ax output
   void apply(BE& Ax, const BE& x) {
      Ax.Copy(Ax,x,0);
      Ax.Saxpy_B(Ax, curl_n2f(x.getE_n(), this->dx, 0), this->tFactor);
      Ax.Saxpy_En(Ax, curl_f2n(x.getB_x(), x.getB_y(), x.getB_z(), this->dx, 0), -this->tFactor*square(PhysConst::c));
   }
   
   // Assign lhs = rhs
   void assign(BE& lhs, const BE& rhs) {
      lhs.Copy(lhs,rhs);
   }

   // Dot product of v1 and v2
   amrex::Real dotProduct(const BE& v1, const BE& v2) {
      return v1.dotProduct(v1,v2);
   }
   
   // lhs += a*rhs
   void increment(BE& lhs, const BE& rhs, amrex::Real a) {
      lhs.Saxpy(lhs,rhs,a);
   }

   // lhs = a*rhs_a + b*rhs_b
   void linComb(BE& lhs, amrex::Real a, const BE& rhs_a, amrex::Real b, const BE& rhs_b) {
      lhs.linComb(lhs,a,rhs_a,b,rhs_b);
   }
   
   // Return new vector suitable for rhs of Ax = b (i.e. b)
   BE makeVecRHS() { return BE(*ba,*dm,0); }

   // Return new vector suitable for rhs of Ax = b (i.e. x)
   BE makeVecLHS() { return BE(*ba,*dm,nghost); }

   // 2-norm of v
   amrex::Real norm2(const BE& v) { return v.norm2(); }

   // Apply right preconditioning, i.e. solve P(lhs) = rhs for preconditioning P
   // P should be an approximation for A
   // for now we use identity, could use A without particles
   void precond(BE& lhs, const BE& rhs) { lhs.Copy(lhs,rhs); }
   
   // Multiply vector v by factor fac
   void scale(BE& v, amrex::Real fac) { v.mult(fac); }

   // Set vector to zero
   void setToZero(BE& v) { v.setVal(0.0); }

};

#endif
