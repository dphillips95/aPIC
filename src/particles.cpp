/*
Particle methods e.g. injectors, accumulators etc. for aPIC.

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

#include <operators.h>
#include <particles.h>
#include <math_functions.h>
#include <constants.h>

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_Particles.H>

#include <AMReX_Print.H>

using namespace amrex;

// Each cell filled with macro pulls from uniform distribution in position
// Thus slightly different from macro*N_cells pull over whole domain
// i.e. particle positions not actually identically distributed
void uniform_injector(myPContainer& pContainer, const Population& pop) {
   // AMR level; currently only single levelled
   int lev = 0;
   
   const Geometry& geom = pContainer.Geom(lev);
   
   const GpuArray<Real,3>
      dom_min = geom.ProbLoArray(),
      dx = geom.CellSizeArray();

   const Real weight = pop.density*math::product(dx)/pop.macro;
   
   for (MFIter mfi = pContainer.MakeMFIter(lev); mfi.isValid(); ++mfi) {
      myPTile& parts = pContainer.GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
      
      const Box& bx_c = mfi.tilebox(AMReXConst::btype_c);
      
      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         const GpuArray<Real,3> r_min = {
            ii*dx[0] + dom_min[0],
            jj*dx[1] + dom_min[1],
            kk*dx[2] + dom_min[2]
         };
         const GpuArray<Real,3> r_max = {
            (ii+1)*dx[0] + dom_min[0],
            (jj+1)*dx[1] + dom_min[1],
            (kk+1)*dx[2] + dom_min[2]         
         };
         
         fill_particles_cell(parts, pop.macro, pop.vth, pop.velocity, weight, r_min, r_max);
            
      });
   }
   pContainer.Redistribute();
}

// Add count particles with given parameters within range
void fill_particles_cell(myPTile& parts, const size_t count, const Real vth, const std::vector<Real> velocity, const Real weight, GpuArray<Real,3> r_min, GpuArray<Real,3> r_max) {
   for (size_t ii=0; ii<count; ++ii) {
      myPType particle;
      particle.id() = myPType::NextID();
      particle.cpu() = amrex::ParallelDescriptor::MyProc();
      for (int nn=0; nn<3; ++nn) {
         particle.pos(nn) = Random()*(r_max[nn] - r_min[nn]) + r_min[nn];
         particle.rdata(pExtra_real_ind::vx_i + nn) = RandomNormal(velocity[nn], vth);
      }
      particle.rdata(pExtra_real_ind::weight_i) = weight;

      parts.push_back(particle);
   }
}

// Push all particle populations
void particlePusher_all(std::vector<std::unique_ptr<myPContainer>>& pContainer_list, const Real dt) {
   for (auto& pContainer : pContainer_list) {
      particlePusher(*pContainer, dt);
   }
}

// Push particles according to velocity
void particlePusher(myPContainer& pContainer, const Real dt) {
   constexpr int lev = 0;
   
   for (myPIter pti(pContainer, lev); pti.isValid(); ++pti) {
      auto& particles = pti.GetArrayOfStructs();

      for (auto& p : particles) {
         p.pos(0) += dt*p.rdata(pExtra_real_ind::vx_i);
         p.pos(1) += dt*p.rdata(pExtra_real_ind::vy_i);
         p.pos(2) += dt*p.rdata(pExtra_real_ind::vz_i);
      }
   }
   
   pContainer.Redistribute();
}

// Accumulates number density, current and kinetic energy simultaneously
// Note: Density and current should be set to zero beforehand
// Also Note: This and accumulateTemperature assume that particle and multifab boxes line up
void accumulateDensityCurrentKE(MultiFab& density, MultiFab& current, MultiFab& KE_Energy, const myPContainer& pContainer, const Population& pop) {
   // AMR level, currently no amr so = 0
   constexpr int lev = 0;

   const GpuArray<Real,3>
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer, lev); pti.isValid(); ++pti) {
      const Array4<Real>& density_array = density.array(pti);
      const Array4<Real>& current_array = current.array(pti);
      const Array4<Real>& KE_Energy_array = KE_Energy.array(pti);
      
      const auto& particles = pti.GetArrayOfStructs();

      for (const auto& p : particles) {
         const IntVect p_indices { get_pos_indices(p.pos(0), p.pos(1), p.pos(2), dx, dom_min, AMReXConst::btype_c) };
         
         density_array(p_indices[0],p_indices[1],p_indices[2]) += p.rdata(pExtra_real_ind::weight_i);
         
         current_array(p_indices[0],p_indices[1],p_indices[2],0) += p.rdata(pExtra_real_ind::vx_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;
         current_array(p_indices[0],p_indices[1],p_indices[2],1) += p.rdata(pExtra_real_ind::vy_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;
         current_array(p_indices[0],p_indices[1],p_indices[2],2) += p.rdata(pExtra_real_ind::vz_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;

         KE_Energy_array(p_indices[0],p_indices[1],p_indices[2]) += pop.mass*p.rdata(pExtra_real_ind::weight_i)*(math::square(p.rdata(pExtra_real_ind::vx_i)) + math::square(p.rdata(pExtra_real_ind::vy_i)) + math::square(p.rdata(pExtra_real_ind::vz_i)));

         // Print() << density_array(p_indices[0],p_indices[1],p_indices[2]) << std::endl;
      }
   }
}

// Accumulates temperature. Note: should be set to zero beforehand
void accumulateTemperature(MultiFab& temperature, const MultiFab& velocity, const myPContainer& pContainer, const Population& pop) {
   constexpr int lev = 0;

   const GpuArray<Real,3>
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer, lev); pti.isValid(); ++pti) {
      const Array4<Real>& temperature_array = temperature.array(pti);
      const Array4<const Real>& velocity_array = velocity.const_array(pti);
      
      auto& particles = pti.GetArrayOfStructs();
      
      for (auto& p : particles) {
         const IntVect p_indices { get_pos_indices(p.pos(0), p.pos(1), p.pos(2), dx, dom_min, AMReXConst::btype_c) };

         Real vth
            = math::square(p.rdata(pExtra_real_ind::vx_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],0))
            + math::square(p.rdata(pExtra_real_ind::vy_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],1))
            + math::square(p.rdata(pExtra_real_ind::vz_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],2));

         vth *= p.rdata(pExtra_real_ind::weight_i)*pop.mass/PhysConst::k;
         
         temperature_array(p_indices[0],p_indices[1],p_indices[2]) += vth;
      }
   }
            
}
