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

#include <math_functions.h>
#include <matrix.h>

#include <AMReX_REAL.H>
#include <AMReX_Particles.H>

#include <vector>

struct Population {
   std::string name; // Population name
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
using myPIterConst = myPContainer::ParConstIterType;
using myPTile = myPContainer::ParticleTileType;
using myPLevel = myPContainer::ParticleLevel;
using myAoS = myPContainer::AoS;
using mySoA = myPContainer::SoA;

inline matrix<amrex::Real> compute_alpha(const amrex::Real Bp_x, const amrex::Real Bp_y, const amrex::Real Bp_z) {
   matrix<amrex::Real> ret(3,3);

   ret(0,0) = Bp_x*Bp_x + 1;
   ret(0,1) = Bp_x*Bp_y + Bp_z;
   ret(0,2) = Bp_x*Bp_z - Bp_y;
   ret(1,0) = Bp_y*Bp_x - Bp_z;
   ret(1,1) = Bp_y*Bp_y + 1;
   ret(1,2) = Bp_y*Bp_z + Bp_x;
   ret(2,0) = Bp_z*Bp_x + Bp_y;
   ret(2,1) = Bp_z*Bp_y - Bp_x;
   ret(2,2) = Bp_z*Bp_z + 1;
   
   ret.scale(1/(1 + Bp_x*Bp_x + Bp_y*Bp_y + Bp_z*Bp_z));
   
   return ret;   
}

inline matrix<amrex::Real> compute_alpha(const std::array<amrex::Real,3>& B_p) {
   return compute_alpha(B_p[0], B_p[1], B_p[2]);
}

void uniform_injector(myPContainer& pContainer, const Population& pop);

void fill_particles_cell(myPTile& parts, const size_t count, const amrex::Real vth, const std::vector<amrex::Real> velocity, const amrex::Real weight, amrex::GpuArray<amrex::Real,3> r_min, amrex::GpuArray<amrex::Real,3> r_max);

void particlePusher(myPContainer& pContainer, const amrex::Real dt);

// Push all particle populations
inline void particlePusher_all(std::vector<std::unique_ptr<myPContainer>>& pContainer_list, const amrex::Real dt) {
   for (auto& pContainer : pContainer_list) {
      particlePusher(*pContainer, dt);
   }
}

void compute_jHat(amrex::MultiFab& jHat, const amrex::Real beta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const myPContainer& pContainer);

// Add contributions to jHat from all populations
inline void compute_jHat_all(amrex::MultiFab& jHat, const amrex::Real dt_theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const std::vector<std::unique_ptr<myPContainer>>& pContainer_list, const std::vector<Population>& pop_list, const amrex::GpuArray<amrex::Real,3>& dx) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      amrex::Real beta = dt_theta*pop_list[nn].charge/pop_list[nn].mass;
      compute_jHat(jHat, beta, B_fx, B_fy, B_fz, *pContainer_list[nn]);
   }
   
   jHat.mult(1/math::product(dx));
}

void accumulateDensityCurrentKE(amrex::MultiFab& density, amrex::MultiFab& current, amrex::MultiFab& KE_Energy, const myPContainer& pContainer, const Population& pop);
void accumulateTemperature(amrex::MultiFab& temperature, const amrex::MultiFab& velocity, const amrex::MultiFab& density, const myPContainer& pContainer, const Population& pop);

#endif
