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

#include <matrix.h>
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
            (kk+1)*dx[2] + dom_min[2],
         };
         
         fill_particles_cell(parts, pop.macro, pop.vth, pop.velocity, weight, r_min, r_max);
            
      });
   }
   pContainer.Redistribute();
}

// Add n = count particles with given parameters within range
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

// Move particles according to velocity
void particleMover(myPContainer& pContainer, const Real dt) {
   constexpr int lev = 0;
   
   for (myPIter pti(pContainer,lev); pti.isValid(); ++pti) {
      myAoS& particles = pti.GetArrayOfStructs();

      for (myPType& p : particles) {
         p.pos(0) += dt*p.rdata(pExtra_real_ind::vx_i);
         p.pos(1) += dt*p.rdata(pExtra_real_ind::vy_i);
         p.pos(2) += dt*p.rdata(pExtra_real_ind::vz_i);
      }
   }
   
   pContainer.Redistribute();
}

// Accelerate particles by Lorentz force, according to ECSIM method
void particleAccelerator(myPContainer& pContainer, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const MultiFab& E_n, const Real theta, const Real beta) {
   constexpr int lev = 0;
   
   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIter pti(pContainer,lev); pti.isValid(); ++pti) {
      myAoS& particles = pti.GetArrayOfStructs();
      
      const Array4<const Real>& B_fx_array = B_fx.const_array(pti);
      const Array4<const Real>& B_fy_array = B_fy.const_array(pti);
      const Array4<const Real>& B_fz_array = B_fz.const_array(pti);
      const Array4<const Real>& E_n_array = E_n.const_array(pti);
      
      for (myPType& p : particles) {
         std::array<Real,3> B_p = face2r(B_fx_array, B_fy_array, B_fz_array, p.pos(0), p.pos(1), p.pos(2), dx, dom_min);
         for (Real& val : B_p) {
            // Rescale magnetic field by beta = dt*theta*q_p/m_p
            val *= beta;
         }
         const std::array<Real,3> E_p = node2r<std::array<Real,3>>(E_n_array, p.pos(0), p.pos(1), p.pos(2), dx, dom_min);
         
         // const std::array<Real,9> alpha = compute_alpha_arr(B_p);
         const matrix<Real,3,3> alpha = compute_alpha(B_p);
         
         const std::array<Real,3> v_acc = {
            p.rdata(pExtra_real_ind::vx_i) + beta*E_p[0],
            p.rdata(pExtra_real_ind::vy_i) + beta*E_p[1],
            p.rdata(pExtra_real_ind::vz_i) + beta*E_p[2]
         };
         
         // const std::array<Real,3> alpha_v = {
         //    alpha[0]*vx_acc + alpha[1]*vy_acc + alpha[2]*vz_acc,
         //    alpha[3]*vx_acc + alpha[4]*vy_acc + alpha[5]*vz_acc,
         //    alpha[6]*vx_acc + alpha[7]*vy_acc + alpha[8]*vz_acc
         // };
         const std::array<Real,3> alpha_v = alpha.mmult(v_acc);
         
         p.rdata(pExtra_real_ind::vx_i) *= 1 - 1/theta;
         p.rdata(pExtra_real_ind::vy_i) *= 1 - 1/theta;
         p.rdata(pExtra_real_ind::vz_i) *= 1 - 1/theta;
         
         p.rdata(pExtra_real_ind::vx_i) += alpha_v[0]/theta;
         p.rdata(pExtra_real_ind::vy_i) += alpha_v[1]/theta;
         p.rdata(pExtra_real_ind::vz_i) += alpha_v[2]/theta;
      }
   }
}

// Computes the rotated current jHat (current due to initial velocity, rotated by magnetic field) for a single population
void compute_jHat(MultiFab& jHat, const Real beta, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const myPContainer& pContainer, const Real factor) {
   constexpr int lev = 0;

   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const myAoS& particles = pti.GetArrayOfStructs();
      
      const Array4<Real>& jHat_array = jHat.array(pti);
      const Array4<const Real>& B_fx_array = B_fx.const_array(pti);
      const Array4<const Real>& B_fy_array = B_fy.const_array(pti);
      const Array4<const Real>& B_fz_array = B_fz.const_array(pti);
      
      for (const myPType& p : particles) {
         const IntVect cell_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 1),
            get_pos_index(p.pos(1), dx[1], dom_min[1], 1),
            get_pos_index(p.pos(2), dx[2], dom_min[2], 1)
         };
         
         const std::array<Real,2>
            x_weight = CIC_weights_1D(p.pos(0), dx[0], cell_indices[0], dom_min[0], 1),
            y_weight = CIC_weights_1D(p.pos(1), dx[1], cell_indices[1], dom_min[1], 1),
            z_weight = CIC_weights_1D(p.pos(2), dx[2], cell_indices[2], dom_min[2], 1);
         
         std::array<Real,3> B_p = face2r(B_fx_array, B_fy_array, B_fz_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);
         for (Real& val : B_p) {
            // Rescale magnetic field by beta = dt*theta*q_p/m_p
            val *= beta;
         }

         const std::array<Real,3> v_i = {
            p.rdata(pExtra_real_ind::vx_i),
            p.rdata(pExtra_real_ind::vy_i),
            p.rdata(pExtra_real_ind::vz_i)
         };
         
         // const std::array<Real,9> alpha = compute_alpha(B_p);
         const matrix<Real,3,3> alpha = compute_alpha(B_p);

         // const std::array<Real,3> alpha_n = {
         //    alpha[0]*p.rdata(pExtra_real_ind::vx_i) + alpha[1]*p.rdata(pExtra_real_ind::vy_i) + alpha[2]*p.rdata(pExtra_real_ind::vz_i),
         //    alpha[3]*p.rdata(pExtra_real_ind::vx_i) + alpha[4]*p.rdata(pExtra_real_ind::vy_i) + alpha[5]*p.rdata(pExtra_real_ind::vz_i),
         //    alpha[6]*p.rdata(pExtra_real_ind::vx_i) + alpha[7]*p.rdata(pExtra_real_ind::vy_i) + alpha[8]*p.rdata(pExtra_real_ind::vz_i)
         // };

         const std::array<Real,3> alpha_n = alpha.mmult(v_i);

         for (int ii=0; ii<2; ++ii) {
            for (int jj=0; jj<2; ++jj) {
               for (int kk=0; kk<2; ++kk) {
                  for (int nn=0; nn<3; ++nn) {
                     jHat_array(cell_indices[0]+ii, cell_indices[1]+jj, cell_indices[2]+kk, nn) += factor*p.rdata(pExtra_real_ind::weight_i)*x_weight[ii]*y_weight[jj]*z_weight[kk]*alpha_n[nn];
                  }
               }
            }
         }
      }
   }
}

// Computes the rotated current jHat_star (current due to both initial velocity (jHat) and initial electric field (jDag), rotated by magnetic field) for a single population
void compute_jHat_star(MultiFab& jHat_star, const Real beta, const Real theta, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const MultiFab& E_n, const myPContainer& pContainer, const Real factor) {
   constexpr int lev = 0;

   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const myAoS& particles = pti.GetArrayOfStructs();
      
      const Array4<Real>& jHat_star_array = jHat_star.array(pti);
      const Array4<const Real>& B_fx_array = B_fx.const_array(pti);
      const Array4<const Real>& B_fy_array = B_fy.const_array(pti);
      const Array4<const Real>& B_fz_array = B_fz.const_array(pti);
      const Array4<const Real>& E_n_array = E_n.const_array(pti);
      
      for (const myPType& p : particles) {
         const IntVect cell_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 1),
            get_pos_index(p.pos(1), dx[1], dom_min[1], 1),
            get_pos_index(p.pos(2), dx[2], dom_min[2], 1)
         };

         const std::array<std::array<Real,2>,3> weights = {
            CIC_weights_1D(p.pos(0), dx[0], cell_indices[0], dom_min[0], 1),
            CIC_weights_1D(p.pos(1), dx[1], cell_indices[1], dom_min[1], 1),
            CIC_weights_1D(p.pos(2), dx[2], cell_indices[2], dom_min[2], 1)
         };
         
         std::array<Real,3> B_p = face2r(B_fx_array, B_fy_array, B_fz_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);
         for (Real& val : B_p) {
            // Rescale magnetic field by beta = dt*theta*q_p/m_p
            val *= beta;
         }
         const std::array<Real,3> E_p = node2r<std::array<Real,3>>(E_n_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);
         
         // const std::array<Real,9> alpha = compute_alpha(B_p);
         const matrix<Real,3,3> alpha = compute_alpha(B_p);
         
         const std::array<Real,3> v_star = {
            p.rdata(pExtra_real_ind::vx_i) + (1-theta)*beta*E_p[0],
            p.rdata(pExtra_real_ind::vy_i) + (1-theta)*beta*E_p[1],
            p.rdata(pExtra_real_ind::vz_i) + (1-theta)*beta*E_p[2]
         };
         
         // const std::array<Real,3> alpha_v = {
         //    alpha[0]*vx_star + alpha[1]*vy_star + alpha[2]*vz_star,
         //    alpha[3]*vx_star + alpha[4]*vy_star + alpha[5]*vz_star,
         //    alpha[6]*vx_star + alpha[7]*vy_star + alpha[8]*vz_star
         // };
         
         const std::array<Real,3> alpha_v = alpha.mmult(v_star);

         for (int ii=0; ii<2; ++ii) {
            for (int jj=0; jj<2; ++jj) {
               for (int kk=0; kk<2; ++kk) {
                  for (int nn=0; nn<3; ++nn) {
                     jHat_star_array(cell_indices[0]+ii, cell_indices[1]+jj, cell_indices[2]+kk, nn) += factor*p.rdata(pExtra_real_ind::weight_i)*weights[0][ii]*weights[1][jj]*weights[2][kk]*alpha_v[nn];
                  }
               }
            }
         }
      }
   }
}

// Computes the rotated current j_E (current due to resultant electric field, rotated by magnetic field) for a single population
// Theta parameter reduces acceleration time by given factor
void compute_j_E(MultiFab& j_E, const Real beta, const Real theta, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const MultiFab& E_n, const myPContainer& pContainer, const Real factor) {
   constexpr int lev = 0;

   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();

   // Print() << "E_n: ";
   // view_MultiFab(E_n);
   
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const myAoS& particles = pti.GetArrayOfStructs();
      
      const Array4<Real>& j_E_array = j_E.array(pti);
      const Array4<const Real>& B_fx_array = B_fx.const_array(pti);
      const Array4<const Real>& B_fy_array = B_fy.const_array(pti);
      const Array4<const Real>& B_fz_array = B_fz.const_array(pti);
      const Array4<const Real>& E_n_array = E_n.const_array(pti);
      
      for (const myPType& p : particles) {
         const IntVect cell_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 1),
            get_pos_index(p.pos(1), dx[1], dom_min[1], 1),
            get_pos_index(p.pos(2), dx[2], dom_min[2], 1)
         };
         
         const std::array<Real,2>
            x_weight = CIC_weights_1D(p.pos(0), dx[0], cell_indices[0], dom_min[0], 1),
            y_weight = CIC_weights_1D(p.pos(1), dx[1], cell_indices[1], dom_min[1], 1),
            z_weight = CIC_weights_1D(p.pos(2), dx[2], cell_indices[2], dom_min[2], 1);
         
         std::array<Real,3> B_p = face2r(B_fx_array, B_fy_array, B_fz_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);
         for (Real& val : B_p) {
            // Rescale magnetic field by beta = dt*theta*q_p/m_p
            val *= beta;
         }
         const std::array<Real,3> E_p = node2r<std::array<Real,3>>(E_n_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);

         // const std::array<Real,9> alpha = compute_alpha(B_p);
         const matrix<Real,3,3> alpha = compute_alpha(B_p);

         const std::array<Real,3> v_E = {
            beta*E_p[0],
            beta*E_p[1],
            beta*E_p[2]
         };
         
         // const std::array<Real,3> alpha_v = {
         //    alpha[0]*vx_E + alpha[1]*vy_E + alpha[2]*vz_E,
         //    alpha[3]*vx_E + alpha[4]*vy_E + alpha[5]*vz_E,
         //    alpha[6]*vx_E + alpha[7]*vy_E + alpha[8]*vz_E
         // };
         const std::array<Real,3> alpha_v = alpha.mmult(v_E);
         
         for (int ii=0; ii<2; ++ii) {
            for (int jj=0; jj<2; ++jj) {
               for (int kk=0; kk<2; ++kk) {
                  for (int nn=0; nn<3; ++nn) {
                     j_E_array(cell_indices[0]+ii, cell_indices[1]+jj, cell_indices[2]+kk, nn) += factor*p.rdata(pExtra_real_ind::weight_i)*x_weight[ii]*y_weight[jj]*z_weight[kk]*alpha_v[nn];
                  }
               }
            }
         }
      }
   }
}

// Accumulates number density, current and kinetic energy simultaneously
// Note: Density and current should be set to zero beforehand
// Also Note: This and accumulateTemperature assume that particle and multifab boxes line up
void accumulateDensityCurrentKE(MultiFab& density, MultiFab& current, MultiFab& KE_Energy, const myPContainer& pContainer, const Population& pop) {
   // AMR level, currently no amr so = 0
   constexpr int lev = 0;

   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const Array4<Real>& density_array = density.array(pti);
      const Array4<Real>& current_array = current.array(pti);
      const Array4<Real>& KE_Energy_array = KE_Energy.array(pti);
      
      const myAoS& particles = pti.GetArrayOfStructs();
      
      for (const myPType& p : particles) {
         const IntVect p_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 0),
            get_pos_index(p.pos(1), dx[1], dom_min[1], 0),
            get_pos_index(p.pos(2), dx[2], dom_min[2], 0)
         };
         
         density_array(p_indices[0],p_indices[1],p_indices[2]) += p.rdata(pExtra_real_ind::weight_i);
         
         current_array(p_indices[0],p_indices[1],p_indices[2],0) += p.rdata(pExtra_real_ind::vx_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;
         current_array(p_indices[0],p_indices[1],p_indices[2],1) += p.rdata(pExtra_real_ind::vy_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;
         current_array(p_indices[0],p_indices[1],p_indices[2],2) += p.rdata(pExtra_real_ind::vz_i)*p.rdata(pExtra_real_ind::weight_i)*pop.charge;

         KE_Energy_array(p_indices[0],p_indices[1],p_indices[2]) += pop.mass*p.rdata(pExtra_real_ind::weight_i)*(math::square(p.rdata(pExtra_real_ind::vx_i)) + math::square(p.rdata(pExtra_real_ind::vy_i)) + math::square(p.rdata(pExtra_real_ind::vz_i)))/2;
      }
   }
}

// Accumulates temperature. Note: should be set to zero beforehand
void accumulateTemperature(MultiFab& temperature, const MultiFab& velocity, const MultiFab& density, const myPContainer& pContainer, const Population& pop) {
   constexpr int lev = 0;

   const GpuArray<Real,3>
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();
   
   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const Array4<Real>& temperature_array = temperature.array(pti);
      const Array4<const Real>& velocity_array = velocity.const_array(pti);
      
      const myAoS& particles = pti.GetArrayOfStructs();
      
      for (const myPType& p : particles) {
         const IntVect p_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 0),
            get_pos_index(p.pos(1), dx[1], dom_min[1], 0),
            get_pos_index(p.pos(2), dx[2], dom_min[2], 0)
         };
          
         Real v2 = math::square(p.rdata(pExtra_real_ind::vx_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],0));
         v2 += math::square(p.rdata(pExtra_real_ind::vy_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],1));
         v2 += math::square(p.rdata(pExtra_real_ind::vz_i) - velocity_array(p_indices[0],p_indices[1],p_indices[2],2));
         
         temperature_array(p_indices[0],p_indices[1],p_indices[2]) += v2*p.rdata(pExtra_real_ind::weight_i);
      }

      const Array4<const Real>& density_array = density.const_array(pti);
      
      const Box& bx_c = pti.tilebox(AMReXConst::btype_c);
      
      // Temperature_array now holds total square velocity,
      // so divide by particle count = number density*cell volume to get mean square
      // Could do this within particle loop above, this way cuts down on * and / operations
      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         temperature_array(ii,jj,kk) *= pop.mass/(3*PhysConst::k*density_array(ii,jj,kk)*math::product(dx));
      });
   }
}

void fillMassMatrices(LayoutData<matrix<Real>>& mat_mass, const int nghost, const Real beta, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const myPContainer& pContainer, const Population& pop) {
   constexpr int lev = 0;
   
   const GpuArray<Real,3>&
      dom_min = pContainer.Geom(lev).ProbLoArray(),
      dx = pContainer.Geom(lev).CellSizeArray();

   for (myPIterConst pti(pContainer,lev); pti.isValid(); ++pti) {
      const Box&
         bx_n = pti.tilebox(AMReXConst::btype_n),
         bx_n_ghost = grow(bx_n, nghost);

      const IntVect
         len_n = bx_n.length(),
         len_n_ghost = bx_n_ghost.length();

      const myAoS& particles = pti.GetArrayOfStructs();
      
      matrix<Real>& matA = mat_mass[pti];
      const Array4<const Real>& B_fx_array = B_fx.const_array(pti);
      const Array4<const Real>& B_fy_array = B_fy.const_array(pti);
      const Array4<const Real>& B_fz_array = B_fz.const_array(pti);

      const IntVect base_index { bx_n.smallEnd() };
      
      for (const myPType& p : particles) {
         // particle cell index
         const IntVect cell_indices = {
            get_pos_index(p.pos(0), dx[0], dom_min[0], 1) - base_index[0],
            get_pos_index(p.pos(1), dx[1], dom_min[1], 1) - base_index[1],
            get_pos_index(p.pos(2), dx[2], dom_min[2], 1) - base_index[2]
         };
         // surrounding node indices
         const std::array<IntVect,8> nodes = { 
            cell_indices,
            cell_indices + IntVect(1,0,0),
            cell_indices + IntVect(0,1,0),
            cell_indices + IntVect(1,1,0),
            cell_indices + IntVect(0,0,1),
            cell_indices + IntVect(1,0,1),
            cell_indices + IntVect(0,1,1),
            cell_indices + IntVect(1,1,1)
         };

         // node IDs
         const std::array<int,8>
            nIDs_row = get_cellID(nodes, len_n),
            nIDs_col = get_cellID(nodes, len_n_ghost);

         const std::array<Real,2>
            x_weight = CIC_weights_1D(p.pos(0), dx[0], cell_indices[0], dom_min[0], 1),
            y_weight = CIC_weights_1D(p.pos(1), dx[1], cell_indices[1], dom_min[1], 1),
            z_weight = CIC_weights_1D(p.pos(2), dx[2], cell_indices[2], dom_min[2], 1);

         std::array<Real,3> B_p = face2r(B_fx_array, B_fy_array, B_fz_array, p.pos(0), p.pos(1), p.pos(2), cell_indices, dx, dom_min);
         for (Real& val : B_p) {
            // Rescale magnetic field by beta = dt*theta*q_p/m_p
            val *= beta;
         }

         // std::array<Real,9> alpha = compute_alpha(B_p);
         matrix<Real,3,3> alpha = compute_alpha(B_p);

         // We need to multiply by particle charge and weight at some point
         Real scale_factor = pop.charge*p.rdata(pExtra_real_ind::weight_i);
         // for (int ii=0; ii<9; ++ii) {
         //    alpha[ii] *= scale_factor;
         // }
         alpha.scale(scale_factor);

         // Final weight should be x_weight[xi]*x_weight[xj]*y_weight[xi]*y_weight[xj]*z_weight[xi]*z_weight[xj]
         // Instead of repeated products, weight is accumulated progressively up through the loop layers
         for (int xi=0; xi<2; ++xi) {
            const Real x_wi = x_weight[xi];
            for (int xj=0; xj<2; ++xj) {
               const Real x_wj = x_wi*x_weight[xj];
               for (int yi=0; yi<2; ++yi) {
                  const Real y_wi = x_wj*y_weight[yi];
                  for (int yj=0; yj<2; ++yj) {
                     const Real y_wj = y_wi*y_weight[yj];
                     for (int zi=0; zi<2; ++zi) {
                        const Real z_wi = y_wj*z_weight[zi];
                        const int nID_row = 3*nIDs_row[xi+2*(yi+2*zi)];
                        for (int zj=0; zj<2; ++zj) {
                           const Real z_wj = z_wi*z_weight[zj];
                           const int nID_col = 3*nIDs_col[xj+2*(yj+2*zj)];
                           for (int ii=0; ii<2; ++ii) {
                              for (int jj=0; jj<2; ++jj) {
                                 matA(nID_row+ii,nID_col+jj) +=
                                    z_wj*alpha(ii,jj);
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

void fillMassMatrices(LayoutData<sp_matrix<Real>>& mat_mass, const Real beta, const MultiFab& B_fx, const MultiFab& B_fy, const MultiFab& B_fz, const myPContainer& pContainer, const Population& pop) {
   
}

void constructEmptyMassMatrix(sp_matrix<Real>& mat_mass, const Box& bx, const int nghost) {
   const Box&
      bx_n = convert(bx,AMReXConst::btype_n),
      bx_n_ghost = grow(bx_n, nghost);

   const IntVect
      len_n = bx_n.length(),
      len_n_ghost = bx_n_ghost.length();

   const int
      total_n = math::product(len_n),
      total_n_ghost = math::product(len_n_ghost);

   // 81 Entries per row - i.e. for each dimension i,
   // all 3 components of E on all 27 surrounding 3x3 nodes affect the local J_i
   constexpr int cols_per_row = 27;

   std::vector<int>
      row_indices, col_indices;

   row_indices.reserve(cols_per_row*total_n);
   
   col_indices.reserve(cols_per_row*total_n_ghost);

   // Since indices are global we must make them local by subtracting the smallest index
   // Could also use manual for loops instead to avoid this
   const IntVect base_index { bx_n.smallEnd() };

   ParallelFor(bx_n, [&](int ii, int jj, int kk) {
      const std::array<IntVect,27> nodes = {
         IntVect(ii-1, jj-1, kk-1) - base_index,
         nodes[0] + IntVect(1,0,0),
         nodes[0] + IntVect(2,0,0),
         nodes[0] + IntVect(0,1,0),
         nodes[0] + IntVect(1,1,0),
         nodes[0] + IntVect(2,1,0),
         nodes[0] + IntVect(0,2,0),
         nodes[0] + IntVect(1,2,0),
         nodes[0] + IntVect(2,2,0),
         nodes[0] + IntVect(0,0,1),
         nodes[0] + IntVect(1,0,1),
         nodes[0] + IntVect(2,0,1),
         nodes[0] + IntVect(0,1,1),
         nodes[0] + IntVect(1,1,1),
         nodes[0] + IntVect(2,1,1),
         nodes[0] + IntVect(0,2,1),
         nodes[0] + IntVect(1,2,1),
         nodes[0] + IntVect(2,2,1),
         nodes[0] + IntVect(0,0,2),
         nodes[0] + IntVect(1,0,2),
         nodes[0] + IntVect(2,0,2),
         nodes[0] + IntVect(0,1,2),
         nodes[0] + IntVect(1,1,2),
         nodes[0] + IntVect(2,1,2),
         nodes[0] + IntVect(0,2,2),
         nodes[0] + IntVect(1,2,2),
         nodes[0] + IntVect(2,2,2)
      };

      // row node ID
      const int nID_000_row = get_cellID(nodes[13], len_n);
      // col node IDs
      const std::array<int,27> nIDs = get_cellID(nodes, len_n_ghost, nghost);
      
      for (int mm=0; mm<27; ++mm) {
         for (int nn=0; nn<3; ++nn) {
            row_indices.push_back(3*nID_000_row + nn);
            col_indices.push_back(3*nIDs[mm] + nn);
         }
      }
   });

   mat_mass.add_chunk(0.0, row_indices, col_indices);
}

void constructEmptyMassMatrices_ba(LayoutData<matrix<Real>>& mat_mass, const int nghost, const BoxArray& ba, const DistributionMapping& dm) {
   for (MFIter mfi(ba,dm); mfi.isValid(); ++mfi) {
      const Box&
         bx_n = mfi.tilebox(AMReXConst::btype_n),
         bx_n_ghost = grow(bx_n, nghost);

      const IntVect
         len_n = bx_n.length(),
         len_n_ghost = bx_n_ghost.length();

      const int
         total_n = math::product(len_n),
         total_n_ghost = math::product(len_n_ghost);
      
      mat_mass[mfi] = matrix<Real>(3*total_n, 3*total_n_ghost, 0.0);
   }
}

void constructEmptyMassMatrices_ba(LayoutData<sp_matrix<Real>>& mat_mass, const int nghost, const BoxArray& ba, const DistributionMapping& dm) {
   for (MFIter mfi(ba,dm); mfi.isValid(); ++mfi) {
      const Box&
         bx_n = mfi.tilebox(AMReXConst::btype_n),
         bx_n_ghost = grow(bx_n, nghost);

      const IntVect
         len_n = bx_n.length(),
         len_n_ghost = bx_n_ghost.length();

      const int
         total_n = math::product(len_n),
         total_n_ghost = math::product(len_n_ghost);

      constexpr int
         cols_per_row = 27;
      
      mat_mass[mfi] = sp_matrix<Real>(3*cols_per_row*total_n, 3*total_n, 3*total_n_ghost);

      constructEmptyMassMatrix(mat_mass[mfi], bx_n, nghost);
   }
}
