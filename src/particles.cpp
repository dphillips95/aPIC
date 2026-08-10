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

#include <particles.h>
#include <math_functions.h>
#include <constants.h>

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_Particles.H>

using namespace amrex;

// Each cell filled with macro pulls from uniform distribution in position
// Thus slightly different from macro*N_cells pull over whole domain
// i.e. particle positions not actually identically distributed
void uniform_injector(myPContainer& pContainer, const Population& pop) {
   // AMR level; currently only single levelled
   int lev = 0;
   
   const Geometry& geom = pContainer.Geom(lev);
   
   GpuArray<Real,3>
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
