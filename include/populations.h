/*
Populations header, containing population class for aPIC.

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

#ifndef POPULATIONS_H_
#define POPULATIONS_H_

#include <AMReX_REAL.H>

struct Population {
   amrex::Real mass = 0.0;
   amrex::Real charge = 0.0;
   amrex::Real temperature = 0.0;
   std::vector<amrex::Real> velocity = {0.0,0.0,0.0};
   amrex::Real density = 0.0;
   amrex::Real macro = 0.0;
};

#endif
