/*
Main source code for aPIC.

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

#include <output.h>
#include <constants.h>
#include <operators.h>
#include <particles.h>
#include <gmres.h>

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <AMReX_Print.H>

#include <vector>

using namespace amrex;

int main(int argc, char* argv[]) {
   Initialize(argc,argv);
   BL_PROFILE_VAR("main()",pmain);
   
   {
      
      constexpr int nghost = 1;
      constexpr IntVect vectghost(nghost,nghost,nghost);
      
      const int nprocs = amrex::ParallelDescriptor::NProcs();
      const int myRank = amrex::ParallelDescriptor::MyProc();
 
      int x_size, y_size, z_size;
      Real
         x_min = 0.0, x_max = 1.0,
         y_min = 0.0, y_max = 1.0,
         z_min = 0.0, z_max = 1.0;

      Array<int,3> periodicity = {false,false,false};
      
      int max_grid_size = 10;

      int seed = 0, dimensions = 3;
      int steps, save_steps = 0;
      Real dt, theta = 0.5, rtol = 1e-15, atol = -1, Vdt_dx_cap = 0.75;
      Real inp_dx = -1, inp_dy = -1, inp_dz = -1;
      std::vector<std::string> pop_name_list, Btype, Etype;
      Real Bx = 0, By = 0, Bz = 0, Ex = 0, Ey = 0, Ez = 0,
         rand_Bx_min = 0, rand_Bx_max = 0,
         rand_By_min = 0, rand_By_max = 0,
         rand_Bz_min = 0, rand_Bz_max = 0,
         rand_Ex_min = 0, rand_Ex_max = 0,
         rand_Ey_min = 0, rand_Ey_max = 0,
         rand_Ez_min = 0, rand_Ez_max = 0;
      Real mass_ratio = PhysConst::m_p/PhysConst::m_e;
      int verbosity = 0;
      int max_gmres = 2000;
      
      std::vector<Population> pop_list;
      
      {
         // Get inputs with ParmParse
         ParmParse inp_m("main");

         inp_m.query("seed", seed);
         inp_m.query("dimensions", dimensions);
         
         ParmParse inp_s("simulation");
         
         inp_s.get("steps", steps);
         inp_s.get("dt", dt);
         inp_s.getarr("pop_list", pop_name_list);
         inp_s.query("theta", theta);
         inp_s.query("rtol", rtol);
         inp_s.query("atol", atol);
         inp_s.query("Vdt_dx_cap", Vdt_dx_cap);
         inp_s.query("save_steps", save_steps);
         inp_s.query("mass_ratio", mass_ratio);
         inp_s.query("verbosity", verbosity);
         inp_s.query("max_gmres", max_gmres);
         
         ParmParse inp_d("domain");
            
         inp_d.get("x_size", x_size);
         inp_d.get("y_size", y_size);
         inp_d.get("z_size", z_size);
         
         inp_d.query("dx", inp_dx);
         inp_d.query("dy", inp_dy);
         inp_d.query("dz", inp_dz);

         // If dx, dy, dz provided then calculated min and max based on it
         if (inp_dx > 0) {
            x_min = -inp_dx*x_size/2;
            x_max = inp_dx*x_size/2;
         } else {
            inp_d.query("x_min", x_min);
            inp_d.query("x_max", x_max);
         }
         if (inp_dy > 0) {
            y_min = -inp_dy*y_size/2;
            y_max = inp_dy*y_size/2;
         } else {
            inp_d.query("y_min", y_min);
            inp_d.query("y_max", y_max);
         }
         if (inp_dz > 0) {
            z_min = -inp_dz*z_size/2;
            z_max = inp_dz*z_size/2;
         } else {
            inp_d.query("z_min", z_min);
            inp_d.query("z_max", z_max);
         }
         
         inp_d.query("period", periodicity);

         inp_d.query("max_grid_size", max_grid_size);

         ParmParse inp_B("magnetic_field");

         inp_B.getarr("type", Btype);
         for (std::string tmp : Btype) {
            if (tmp == "uniform") {
               inp_B.query("Bx", Bx);
               inp_B.query("By", By);
               inp_B.query("Bz", Bz);
            } else if (tmp == "rand") {
               inp_B.query("rand_Bx_min", rand_Bx_min);
               inp_B.query("rand_Bx_max", rand_Bx_max);
               inp_B.query("rand_By_min", rand_By_min);
               inp_B.query("rand_By_max", rand_By_max);
               inp_B.query("rand_Bz_min", rand_Bz_min);
               inp_B.query("rand_Bz_max", rand_Bz_max);
            }
         }

         ParmParse inp_E("electric_field");

         inp_E.getarr("type", Etype);
         for (std::string tmp : Etype) {
            if (tmp == "uniform") {
               inp_E.query("Ex", Ex);
               inp_E.query("Ey", Ey);
               inp_E.query("Ez", Ez);
            } else if (tmp == "rand") {
               inp_E.query("rand_Ex_min", rand_Ex_min);
               inp_E.query("rand_Ex_max", rand_Ex_max);
               inp_E.query("rand_Ey_min", rand_Ey_min);
               inp_E.query("rand_Ey_max", rand_Ey_max);
               inp_E.query("rand_Ez_min", rand_Ez_min);
               inp_E.query("rand_Ez_max", rand_Ez_max);
            }
         }

         for (std::string pop : pop_name_list) {
            Population tmp;
            
            ParmParse inp_pop(pop);

            tmp.name = pop;
            
            // Particle default type is false
            bool electron = false;
            
            inp_pop.query("electron", electron);
            
            inp_pop.query("mass", tmp.mass);
            inp_pop.query("charge", tmp.charge);
            inp_pop.query("temperature", tmp.temperature);
            inp_pop.queryarr("velocity", tmp.velocity);
            inp_pop.query("density", tmp.density);
            inp_pop.query("macro", tmp.macro);
            
            tmp.mass *= PhysConst::m_p;
            tmp.charge *= PhysConst::q_e;
            
            if (electron) {
               tmp.mass /= mass_ratio;
            }
            
            tmp.vth = std::sqrt(PhysConst::k*tmp.temperature/tmp.mass);
            
            pop_list.push_back(tmp);
         }
      }

      const size_t pop_count = pop_list.size();

      // Add process rank to seed to ensure each process has a different seed
      InitRandom(seed + myRank, nprocs, 0);

      std::ofstream datalog(Log::fieldlog_filename);
      initialise_datalog(datalog);
      
      Box
         box_c(IntVect{0,0,0}, IntVect{x_size-1, y_size-1, z_size-1}); // cell-centred

      RealBox real_box ({x_min,y_min,z_min}, {x_max,y_max,z_max});
      
      Geometry geom;
      geom.define(box_c, real_box, CoordSys::cartesian, periodicity);
      
      GpuArray<Real,3> dx = geom.CellSizeArray();

      Real
         dV = math::product(dx),
         dV_inv = 1/dV;
      
      BoxArray
         ba_c(box_c); // cell-centred, others generated via conversion as needed

      // Divide domain into boxes
      ba_c.maxSize(max_grid_size);
      
      BoxArray
         ba_n = convert(ba_c,AMReXConst::btype_n),
         ba_fx = convert(ba_c,AMReXConst::btype_fx),
         ba_fy = convert(ba_c,AMReXConst::btype_fy),
         ba_fz = convert(ba_c,AMReXConst::btype_fz),
         ba_ex = convert(ba_c,AMReXConst::btype_ex),
         ba_ey = convert(ba_c,AMReXConst::btype_ey),
         ba_ez = convert(ba_c,AMReXConst::btype_ez);
      
      DistributionMapping dm(ba_c);

      // Vector containing B_f and E_n states
      BE
         EM_state(ba_n, dm, nghost, geom.periodicity()),
         // Mid state; used for particle acceleration, no need for ghost cells
         // (Unless particle cloud is larger than one cell length)
         EM_state_mid(ba_n, dm, 0, geom.periodicity());
      
      MultiFab* E_n = &EM_state.getE_n();
      MultiFab* E_n_mid = &EM_state_mid.getE_n();
      
      std::array<MultiFab*,3> B_f = {
         &EM_state.getB_fx(),
         &EM_state.getB_fy(),
         &EM_state.getB_fz()
      };
      
      std::array<MultiFab*,3> B_f_mid = {
         &EM_state_mid.getB_fx(),
         &EM_state_mid.getB_fy(),
         &EM_state_mid.getB_fz()
      };
      
      // Vector of particle containers
      std::vector<myPContainer> pContainer_list;
      pContainer_list.reserve(pop_count);
      
      // std::array<MultiFab,3> B_f = {
      //    MultiFab(ba_fx, dm, 1, nghost),
      //    MultiFab(ba_fy, dm, 1, nghost),
      //    MultiFab(ba_fz, dm, 1, nghost)
      // };
      
      MultiFab
         curl_B_n(ba_n, dm, 3, 0),
         curl_E_fx(ba_fx, dm, 3, 0),
         curl_E_fy(ba_fy, dm, 3, 0),
         curl_E_fz(ba_fz, dm, 3, 0),
         jHat(ba_n, dm, 3, 0);
      
      curl_B_n.setVal(0.0);
      curl_E_fx.setVal(0.0);
      curl_E_fy.setVal(0.0);
      curl_E_fz.setVal(0.0);
      jHat.setVal(0.0);
      
      std::array<MultiFab,3> Jp_f = {
         MultiFab(ba_fx, dm, 1, nghost),
         MultiFab(ba_fy, dm, 1, nghost),
         MultiFab(ba_fz, dm, 1, nghost)
      };

      // Distributed matrices for curl operators
#if USE_CURLB_MATRIX
      std::array<LayoutData<sp_matrix<Real>>,3> matA_B2E = {
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm)
      };
#endif
#if USE_CURLE_MATRIX
      std::array<LayoutData<sp_matrix<Real>>,3> matA_E2B = {
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm)
      };
#endif
#if USE_CURRENT_MATRIX
      LayoutData<matrix<Real>> matA_E2E = LayoutData<sp_matrix<Real>>(ba_c,dm);
#endif
      
      E_n->setVal(0.0);
      for (int nn=0; nn<3; ++nn) {
         Jp_f[nn].setVal(0.0);
         B_f[nn]->setVal(0.0);
      }
      
      /*
      if (myRank == dm[0]) {
         const Array4<Real>&
            En_array = (*E_n)[0].array(),
            Bf_array_x = (*B_f[0])[0].array(),
            Bf_array_y = (*B_f[1])[0].array(),
            Bf_array_z = (*B_f[2])[0].array();

         ParallelFor(ba_n[0], [&](int ii, int jj, int kk) {
            En_array(ii,jj,kk,0) = 0;
            En_array(ii,jj,kk,1) = 0;
            En_array(ii,jj,kk,2) = 0;
         });

         ParallelFor(ba_fx[0], [&](int ii, int jj, int kk) {
            Bf_array_x(ii,jj,kk) = 0;
         });

         ParallelFor(ba_fy[0], [&](int ii, int jj, int kk) {
            Bf_array_y(ii,jj,kk) = 0;
         });

         ParallelFor(ba_fz[0], [&](int ii, int jj, int kk) {
            Bf_array_z(ii,jj,kk) = 1;
         });
      }
      */

      // B-E fields initial condition
      for (MFIter mfi(*E_n); mfi.isValid(); ++mfi) {
         const Box&
            bx_n = mfi.tilebox(AMReXConst::btype_n),
            bx_fx = mfi.tilebox(AMReXConst::btype_fx),
            bx_fy = mfi.tilebox(AMReXConst::btype_fy),
            bx_fz = mfi.tilebox(AMReXConst::btype_fz);
         const Array4<Real>&
            En_array = E_n->array(mfi),
            Bf_array_x = B_f[0]->array(mfi),
            Bf_array_y = B_f[1]->array(mfi),
            Bf_array_z = B_f[2]->array(mfi);
         
         ParallelFor(bx_n, [&](int ii, int jj, int kk) {
            // Real
            //    x = ii*dx[0] + x_min,
            //    y = jj*dx[1] + y_min,
            //    z = kk*dx[2] + z_min;
            // En_array(ii,jj,kk,0) = 0.0;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            // En_array(ii,jj,kk,0) = jj;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            // En_array(ii,jj,kk,1) = std::cos(2*M_PI*x/(x_max - x_min))*std::sin(2*M_PI*y/(y_max - y_min))*std::sin(2*M_PI*z/(z_max - z_min));
            // En_array(ii,jj,kk,2) = std::sin(2*M_PI*x/(x_max - x_min))*std::cos(2*M_PI*y/(y_max - y_min))*std::cos(2*M_PI*z/(z_max - z_min));
            
            for (std::string tmp : Etype) {
               if (tmp == "uniform") {
                  En_array(ii,jj,kk,0) += Ex;
                  En_array(ii,jj,kk,1) += Ey;
                  En_array(ii,jj,kk,2) += Ez;
               } else if (tmp == "rand") {
                  En_array(ii,jj,kk,0) += Random()*(rand_Ex_max - rand_Ex_min) + rand_Ex_min;
                  En_array(ii,jj,kk,1) += Random()*(rand_Ey_max - rand_Ey_min) + rand_Ey_min;
                  En_array(ii,jj,kk,2) += Random()*(rand_Ez_max - rand_Ez_min) + rand_Ez_min;
               }
            }  
         });
         
         ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
            // Real
            //    x = ii*dx[0] + x_min,
            //    y = (jj+0.5)*dx[1] + y_min,
            //    z = (kk+0.5)*dx[2] + z_min;
            // Bf_array_x(ii,jj,kk) = 0.0;
            
            for (std::string tmp : Btype) {
               if (tmp == "uniform") {
                  Bf_array_x(ii,jj,kk) += Bx;
               } else if (tmp == "rand") {
                  Bf_array_x(ii,jj,kk) += Random()*(rand_Bx_max - rand_Bx_min) + rand_Bx_min;
               }
            }  
         });

         ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
            // Real
            //    x = (ii+0.5)*dx[0] + x_min,
            //    y = jj*dx[1] + y_min,
            //    z = (kk+0.5)*dx[2] + z_min;
            // Bf_array_y(ii,jj,kk) = ii;
            
            for (std::string tmp : Btype) {
               if (tmp == "uniform") {
                  Bf_array_y(ii,jj,kk) += By;
               } else if (tmp == "rand") {
                  Bf_array_y(ii,jj,kk) += Random()*(rand_By_max - rand_By_min) + rand_By_min;
               }
            }  
         });

         ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
            // Real
            //    x = (ii+0.5)*dx[0] + x_min,
            //    y = (jj+0.5)*dx[1] + y_min,
            //    z = kk*dx[2] + z_min;
            // Bf_array_z(ii,jj,kk) = ii*ii;
            
            for (std::string tmp : Btype) {
               if (tmp == "uniform") {
                  Bf_array_z(ii,jj,kk) += Bz;
               } else if (tmp == "rand") {
                  Bf_array_z(ii,jj,kk) += Random()*(rand_Bz_max - rand_Bz_min) + rand_Bz_min;
               }
            }  
         });
      }

      std::vector<Real> max_spds;
      
      // Particles initial condition; fill with uniform distribution
      for (size_t ii=0; ii<pop_count; ++ii) {
         pContainer_list.emplace_back(geom, dm, ba_c);
         uniform_injector(pContainer_list[ii], pop_list[ii]);
         max_spds.push_back(max_spd(pContainer_list[ii]));
         if (max_spds[ii] * dt > Vdt_dx_cap * std::min({dx[0],dx[1],dx[2]})) {
            Print() << "WARNING: Maximum initial particle speed of population "
                    << pop_list[ii].name << "exceeds minimum dx * Vdt_cap";
         }
      }
      
      // Boundary conditions
      E_n->FillBoundary(geom.periodicity());
      for (int nn=0; nn<3; ++nn) {
         B_f[nn]->FillBoundary(geom.periodicity());
      }

      // Make sure nodes in periodic data are also synced (fillboundary does not update "valid" nodes on edge, which will match those on other side of domain with periodic BCs)
      // There (should) not be any reason why this will be violated at future time
      // Data is identical as it is shared across boundaries
      // Only(?) question is if calculations are done in same order
      E_n->EnforcePeriodicity(geom.periodicity());
      for (int nn=0; nn<3; ++nn) {
         B_f[nn]->EnforcePeriodicity(geom.periodicity());
      }

      // std::unique_ptr<iMultiFab>
      //    omask_fx_ghost(*B_f[0], geom.periodicity(), vectghost),
      //    omask_fy_ghost(*B_f[1], geom.periodicity(), vectghost),
      //    omask_fz_ghost(*B_f[2], geom.periodicity(), vectghost),
      //    omask_n_ghost(*E_n, geom.periodicity(), vectghost);
      
      // Ensure boundary conditions were not violated in last step
      E_n->FillBoundary(geom.periodicity());
      for (int nn=0; nn<3; ++nn) {
         B_f[nn]->FillBoundary(geom.periodicity());
      }

      // Forces valid data points shared between boxes to be equal
      // May be unnecessary - EnforcePeriodicity() probably already does this?
      E_n->OverrideSync(geom.periodicity());
      for (int nn=0; nn<3; ++nn) {
         B_f[nn]->OverrideSync(geom.periodicity());
      }
      
      // Generate curl operators
#if USE_CURLB_MATRIX
      get_curl_f2n_operator_ba(matA_B2E, nghost, dx, E_n->boxArray(), dm);
#endif
#if USE_CURLE_MATRIX
      get_curl_n2f_operator_ba(matA_E2B, nghost, dx, E_n->boxArray(), dm);
#endif

#if USE_CURRENT_MATRIX
      // Generate empty mass matrices
      constructEmptyMassMatrices_ba(matA_E2E, nghost, E_n->boxArray(), dm);
#endif
      
      // const IntVect& sym_dir = AMReXConst::btype_ex;
      
      // Print() << "E_n: " << sym_test(*E_n,sym_dir) << std::endl
      //         << "B_fx: " << sym_test(*B_f[0],sym_dir) << std::endl
      //         << "B_fy: " << sym_test(*B_f[1],sym_dir) << std::endl
      //         << "B_fz: " << sym_test(*B_f[2],sym_dir) << std::endl;
      
      Real time = 0.0;

      // Decenter particles (probably not strictly necessary)
      particlePusher_all(pContainer_list, -dt/2);
      
      for (int step=0; step<steps; ++step) {
         Print()  << std::endl << "Step: " << step << std::endl;

         // Copy previous state into mid-state
         BE::Copy(EM_state_mid, EM_state, 0);
         
         if (step % save_steps == 0) {
            saveState(step, time, EM_state, pContainer_list, pop_list, geom, datalog);
         }
         
         // Advance particle positions
         particlePusher_all(pContainer_list, dt);
         
         // Calculate rotated current jHat
         jHat.setVal(0.0);
         // compute_jHat_all(jHat, dt, theta, *B_f[0], *B_f[1], *B_f[2], pContainer_list, pop_list, dV_inv);
         compute_jHat_pr_all(jHat, dt, theta, *B_f[0], *B_f[1], *B_f[2], *E_n, pContainer_list, pop_list, dV_inv);

#if USE_CURRENT_MATRIX
         // Compute mass matrices
         for (MFIter mfi(matA_E2E); mfi.isValid(); ++mfi) {
            matA_E2E[mfi].setVal(0.0);
         }
         fillMassMatrices_all(matA_E2E, nghost, dt, *B_f[0], *B_f[1], *B_f[2], pContainer_list, pop_list, dV_inv);
#endif
         
         gmres_step(EM_state, jHat, pop_list, pContainer_list,
#if USE_CURLB_MATRIX
                    matA_B2E,
#endif
#if USE_CURLE_MATRIX
                    matA_E2B,
#endif
#if USE_CURRENT_MATRIX
                    matA_E2E,
#endif
                    dx, dt, theta, rtol, atol, verbosity, max_gmres);
         
         // Print() << "E_n: " << sym_test(*E_n,sym_dir) << std::endl
         //         << "B_fx: " << sym_test(*B_f[0],sym_dir) << std::endl
         //         << "B_fy: " << sym_test(*B_f[1],sym_dir) << std::endl
         //         << "B_fz: " << sym_test(*B_f[2],sym_dir) << std::endl;

         // Complete boundary conditions
         E_n->FillBoundary(geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            B_f[nn]->FillBoundary(geom.periodicity());
         }

         E_n->OverrideSync(geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            B_f[nn]->OverrideSync(geom.periodicity());
         }

         // Combine previous E with new E to create mid-step E for particle accelerator
         EM_state_mid.mult_En(1 - theta);
         BE::Saxpy_En(EM_state_mid, EM_state.getE_n_const(), 0, theta);
         
         particleAccelerator_all(pContainer_list, pop_list, *B_f_mid[0], *B_f_mid[1], *B_f_mid[2], *E_n_mid, dt, theta);
         
         time += dt;

      }
      saveState(steps, time, EM_state, pContainer_list, pop_list, geom, datalog);
   }
      
   BL_PROFILE_VAR_STOP(pmain);
   Finalize();
}
