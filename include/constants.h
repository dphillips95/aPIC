#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>

namespace PhysConst {
   static constexpr amrex::Real c = 299792458.;
   static constexpr amrex::Real ep0 = 8.854187817e-12;
   static constexpr amrex::Real mu0 = 1.2566370614359173e-06;
   static constexpr amrex::Real q_e = 1.6021764620000001e-19;
   static constexpr amrex::Real m_e = 9.10938291e-31;
   static constexpr amrex::Real m_p = 1.6726231000000001e-27;
}

namespace AMReXConst {
   static constexpr amrex::IntVect btype_c(0,0,0);
   static constexpr amrex::IntVect btype_n(1,1,1);
   static constexpr amrex::IntVect btype_fx(1,0,0);
   static constexpr amrex::IntVect btype_fy(0,1,0);
   static constexpr amrex::IntVect btype_fz(0,0,1);
   static constexpr amrex::IntVect btype_ex(0,1,1);
   static constexpr amrex::IntVect btype_ey(1,0,1);
   static constexpr amrex::IntVect btype_ez(1,1,0);
}
   
#endif
