/*
Physical and Internal constants for aPIC.

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

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>

namespace PhysConst {
   static constexpr amrex::Real c = 299792458.;
   static constexpr amrex::Real mu0 = 1.2566370614359173e-06;
   static constexpr amrex::Real eps0 = 1/(mu0*c*c); // 8.854187817e-12;
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

namespace Log {
   static constexpr int datWidth = 24;
   static constexpr int stepWidth = 8;
   static constexpr int datPrecision = 16;
   static const std::string fieldlog_filename = "fields.log";
}

#endif
