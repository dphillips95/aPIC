/*
Output header, containing output methods for aPIC.

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
#include <particles.h>

#include <AMReX_REAL.H>

#ifndef OUTPUT_H_
#define OUTPUT_H_

void initialise_datalog(std::ofstream& datalog);

void saveState(int step, amrex::Real time, const BE& EM_state, const std::vector<std::unique_ptr<myPContainer>>& pContainer_list, const std::vector<Population>& pop_list, const amrex::Geometry& geom, std::ofstream& datalog);

#endif
