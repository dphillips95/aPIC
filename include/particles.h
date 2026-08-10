/*
Particles header, containing particle methods etc. for aPIC.

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


Author(s): David Phillips, Ilja Honkonen
*/

#ifndef PARTICLES_H_
#define PARTICLES_H_

#include <AMReX_REAL.H>
#include <AMReX_Particles.H>

struct Population {
   amrex::Real mass = 0.0; // Mass per particle
   amrex::Real charge = 0.0; // Charge per particle
   amrex::Real temperature = 0.0;
   amrex::Real vth = 0.0; // 1D thermal velocity
   std::vector<amrex::Real> velocity = {0.0,0.0,0.0};
   amrex::Real density = 0.0; // Number density
   int macro = 0; // Macroparticles per cell (on initialisation)
};

enum struct Particle_real_extra {
   vx, vy, vz, // Velocity
   weight,     // Statistical Weight / mass
   size
};

namespace pExtra_real_ind {
   constexpr int
      vx_i = static_cast<int>(Particle_real_extra::vx),
      vy_i = static_cast<int>(Particle_real_extra::vy),
      vz_i = static_cast<int>(Particle_real_extra::vz),
      weight_i = static_cast<int>(Particle_real_extra::weight);
};

enum struct Particle_int_extra {
   size
};

using myPContainer = amrex::ParticleContainer<
   static_cast<int>(Particle_real_extra::size), static_cast<int>(Particle_int_extra::size), 0, 0>;
using myPType = myPContainer::ParticleType;
using myPIter = myPContainer::ParIterType;
using myPTile = myPContainer::ParticleTileType;
using myPLevel = myPContainer::ParticleLevel;

void uniform_injector(myPContainer& pContainer, const Population& pop);

void fill_particles_cell(myPTile& parts, const size_t count, const amrex::Real vth, const std::vector<amrex::Real> velocity, const amrex::Real weight, amrex::GpuArray<amrex::Real,3> r_min, amrex::GpuArray<amrex::Real,3> r_max);

#endif
