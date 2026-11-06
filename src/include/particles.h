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

#include <AMReX_Print.H>

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

// Compute alpha (magnetic rotation matrix)
inline matrix<amrex::Real,3,3> compute_alpha(const amrex::Real Bp_x, const amrex::Real Bp_y, const amrex::Real Bp_z) {
   matrix<amrex::Real,3,3> ret;
   
   ret(0,0) = Bp_x*Bp_x + 1;
   ret(0,1) = Bp_x*Bp_y + Bp_z;
   ret(0,2) = Bp_x*Bp_z - Bp_y;
   ret(1,0) = Bp_y*Bp_x - Bp_z;
   ret(1,1) = Bp_y*Bp_y + 1;
   ret(1,2) = Bp_y*Bp_z + Bp_x;
   ret(2,0) = Bp_z*Bp_x + Bp_y;
   ret(2,1) = Bp_z*Bp_y - Bp_x;
   ret(2,2) = Bp_z*Bp_z + 1;

   amrex::Real scale_factor = 1/(1 + math::square(Bp_x, Bp_y, Bp_z));

   ret.scale(scale_factor);

   return ret;
}

inline matrix<amrex::Real,3,3> compute_alpha(const std::array<amrex::Real,3>& B_p) {
   return compute_alpha(B_p[0], B_p[1], B_p[2]);
}

// Compute alpha as flat array
inline std::array<amrex::Real,9> compute_alpha_arr(const amrex::Real Bp_x, const amrex::Real Bp_y, const amrex::Real Bp_z) {
   std::array<amrex::Real,9> ret;

   ret[0] = Bp_x*Bp_x + 1;
   ret[1] = Bp_x*Bp_y + Bp_z;
   ret[2] = Bp_x*Bp_z - Bp_y;
   ret[3] = Bp_y*Bp_x - Bp_z;
   ret[4] = Bp_y*Bp_y + 1;
   ret[5] = Bp_y*Bp_z + Bp_x;
   ret[6] = Bp_z*Bp_x + Bp_y;
   ret[7] = Bp_z*Bp_y - Bp_x;
   ret[8] = Bp_z*Bp_z + 1;

   amrex::Real scale_factor = 1/(1 + math::square(Bp_x, Bp_y, Bp_z));

   for (int ii=0; ii<9; ++ii) { ret[ii] *= scale_factor; }
   
   return ret;   
}

inline std::array<amrex::Real,9> compute_alpha_arr(const std::array<amrex::Real,3>& B_p) {
   return compute_alpha_arr(B_p[0], B_p[1], B_p[2]);
}

void uniform_injector(myPContainer& pContainer, const Population& pop);

void fill_particles_cell(myPTile& parts, const size_t count, const amrex::Real vth, const std::vector<amrex::Real> velocity, const amrex::Real weight, amrex::GpuArray<amrex::Real,3> r_min, amrex::GpuArray<amrex::Real,3> r_max);

void particleMover(myPContainer& pContainer, const amrex::Real dt);

// Move all particle populations
inline void particleMover_all(std::vector<myPContainer>& pContainer_list, const amrex::Real dt) {
   for (myPContainer& pContainer : pContainer_list) {
      particleMover(pContainer, dt);
   }
}

void particleAccelerator(myPContainer& pContainer, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, const amrex::Real theta, const amrex::Real beta);

inline void particleAccelerator_all(std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, const amrex::Real dt, const amrex::Real theta) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      const auto pop = pop_list[nn];
      amrex::Real beta = dt*theta*pop.charge/pop.mass;
      particleAccelerator(pContainer_list[nn], B_fx, B_fy, B_fz, E_n, theta, beta);
   }
}

// compute jHat; rotated current due only to initial particle velocities
void compute_jHat(amrex::MultiFab& jHat, const amrex::Real beta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const myPContainer& pContainer, amrex::Real factor);

// Add contributions to jHat from all populations
inline void compute_jHat_all(amrex::MultiFab& jHat, const amrex::Real dt_theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const amrex::Real factor) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      const auto pop = pop_list[nn];
      amrex::Real beta = dt_theta*pop.charge/pop.mass;
      compute_jHat(jHat, beta, B_fx, B_fy, B_fz, pContainer_list[nn], pop.charge*factor);
   }
}

// compute jHat_star, rotated current due to initial velocity plus acceleration due to initial electric field
void compute_jHat_star(amrex::MultiFab& jHat_star, const amrex::Real beta, const amrex::Real theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, const myPContainer& pContainer, const amrex::Real factor);

// Add contributions to jHat_star from all populations
inline void compute_jHat_star_all(amrex::MultiFab& jHat_star, const amrex::Real dt, const amrex::Real theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const amrex::Real factor) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      const auto pop = pop_list[nn];
      amrex::Real beta = dt*theta*pop.charge/pop.mass;
      compute_jHat_star(jHat_star, beta, theta, B_fx, B_fy, B_fz, E_n, pContainer_list[nn], pop.charge*factor);
   }
}

// compute j_E, rotated current due to acceleration by electric field
void compute_j_E(amrex::MultiFab& j_E, const amrex::Real beta, const amrex::Real theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, const myPContainer& pContainer, const amrex::Real factor);

// Add contributions to j_E from all populations
inline void compute_j_E_all(amrex::MultiFab& j_E, const amrex::Real dt, const amrex::Real theta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const amrex::MultiFab& E_n, std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const amrex::Real factor) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      const auto pop = pop_list[nn];
      amrex::Real beta = dt*theta*pop.charge/pop.mass;
      // amrex::Print() << pop.name << " Beta: " << beta
      //                << ", theta: " << theta
      //                << ", charge: " << pop.charge
      //                << ", mass: " << pop.mass << std::endl;
      compute_j_E(j_E, beta, theta, B_fx, B_fy, B_fz, E_n, pContainer_list[nn], pop.charge*factor);
      // amrex::Print() << "sub-current due to E_{n+1} + E_{n+1}: ";
      // view_MultiFab(j_E);
   }
}

void accumulateDensityCurrentKE(amrex::MultiFab& density, amrex::MultiFab& current, amrex::MultiFab& KE_Energy, const myPContainer& pContainer, const Population& pop);
void accumulateTemperature(amrex::MultiFab& temperature, const amrex::MultiFab& velocity, const amrex::MultiFab& density, const myPContainer& pContainer, const Population& pop);

void constructEmptyMassMatrix(sp_matrix<amrex::Real>& mat_mass, const amrex::Box& bx, const int nghost);

void fillMassMatrices(amrex::LayoutData<sp_matrix<amrex::Real>>& mat_mass, const int nghost, const amrex::Real beta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const myPContainer& pContainer, const Population& pop);
void fillMassMatrices(amrex::LayoutData<matrix<amrex::Real>>& mat_mass, const int nghost, const amrex::Real beta, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, const myPContainer& pContainer, const Population& pop);

template <typename T>
inline void fillMassMatrices_all(amrex::LayoutData<T>& mat_mass, const int nghost, const amrex::Real dt, const amrex::MultiFab& B_fx, const amrex::MultiFab& B_fy, const amrex::MultiFab& B_fz, std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const amrex::Real dV_inv) {
   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      const auto pop = pop_list[nn];
      amrex::Real beta = dt*pop.charge/(2*pop.mass);
      fillMassMatrices(mat_mass, nghost, beta, B_fx, B_fy, B_fz, pContainer_list[nn], pop);
   }

   for (amrex::MFIter mfi(mat_mass); mfi.isValid(); ++mfi) {
      mat_mass[mfi].scale(dV_inv);
   }
}
// Construct empty mass matrices for entire BoxArray
void constructEmptyMassMatrices_ba(amrex::LayoutData<matrix<amrex::Real>>& mat_mass, const int nghost, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);
void constructEmptyMassMatrices_ba(amrex::LayoutData<sp_matrix<amrex::Real>>& mat_mass, const int nghost, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm);

inline amrex::Real max_spd(const myPContainer& pContainer) {
   constexpr int lev = 0;
   
   amrex::Real max_spd = 0.0;
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const myAoS& particles = pti.GetArrayOfStructs();
      
      for (const myPType& p : particles) {
         amrex::Real spd = std::sqrt(
                                     math::square(p.rdata(pExtra_real_ind::vx_i)) + 
                                     math::square(p.rdata(pExtra_real_ind::vy_i)) + 
                                     math::square(p.rdata(pExtra_real_ind::vz_i)));
         max_spd = std::max(max_spd, spd);
      }
   }
   
   return max_spd;
}

#endif
