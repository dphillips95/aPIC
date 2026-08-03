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

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <constants.h>
#include <operators.h>
#include <populations.h>
#include <gmres.h>

using namespace amrex;

int main(int argc, char* argv[]) {
   Initialize(argc,argv);
   BL_PROFILE_VAR("main()",pmain);
   
   {
      
      constexpr int nghost = 1;
      constexpr IntVect vectghost(nghost,nghost,nghost);
      
      int x_size, y_size, z_size;
      Real
         x_min = 0.0, x_max = 1.0,
         y_min = 0.0, y_max = 1.0,
         z_min = 0.0, z_max = 1.0;

      Array<int,3> periodicity = {false,false,false};

      const int nprocs = ParallelDescriptor::NProcs();
      
      int max_grid_size = 10;

      int seed = 0, dimensions = 3;
      bool v_1D = false;
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
      
      std::vector<Population> pop_list;
      
      {
         // Get inputs with ParmParse
         ParmParse inp_m("main");

         inp_m.query("seed", seed);
         inp_m.query("dimensions", dimensions);
         inp_m.query("1v", v_1D);

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
            
            bool electron;
            
            inp_pop.query("electron", electron);
            
            inp_pop.query("mass", tmp.mass);
            inp_pop.query("charge", tmp.charge);
            inp_pop.query("temperature", tmp.temperature);
            inp_pop.queryarr("velocity", tmp.velocity);
            inp_pop.query("density", tmp.density);
            inp_pop.query("macro", tmp.macro);
            
            if (electron) {
               tmp.mass /= mass_ratio;
            }

            pop_list.push_back(tmp);
         }
      }

      InitRandom(seed, nprocs, 0);
      
      std::ofstream datalog(Log::fieldlog_filename);
      
      datalog << std::setw(Log::stepWidth) << "Step"
              << std::setw(Log::datWidth) << "B_energy"
              << std::setw(Log::datWidth) << "E_energy"
              << std::setw(Log::datWidth) << "Total_energy" << std::endl;
      
      Box
         box_c(IntVect{0,0,0}, IntVect{x_size-1, y_size-1, z_size-1}); // cell-centred
      
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
      
      MultiFab
         Jp_c(ba_c, dm, 3, nghost),
         B_n(ba_n, dm, 3, nghost),
         E_n(ba_n, dm, 3, nghost),
         B_c(ba_c, dm, 3, nghost),
         E_c(ba_c, dm, 3, nghost),
         Energy_c(ba_c, dm, 3, nghost);
      
      MultiFab
         curl_B_n(ba_n, dm, 3, 0),
         curl_E_fx(ba_fx, dm, 3, 0),
         curl_E_fy(ba_fy, dm, 3, 0),
         curl_E_fz(ba_fz, dm, 3, 0);
         // curl_B_n_sp(ba_n, dm, 3, 0),
         // curl_E_fx_sp(ba_fx, dm, 3, 0),
         // curl_E_fy_sp(ba_fy, dm, 3, 0),
         // curl_E_fz_sp(ba_fz, dm, 3, 0);

      curl_B_n.setVal(0.0);
      curl_E_fx.setVal(0.0);
      curl_E_fy.setVal(0.0);
      curl_E_fz.setVal(0.0);
      // curl_B_n_sp.setVal(0.0);
      // curl_E_fx_sp.setVal(0.0);
      // curl_E_fy_sp.setVal(0.0);
      // curl_E_fz_sp.setVal(0.0);
      
      std::array<MultiFab,3> Jp_f = {
         MultiFab(ba_fx, dm, 1, nghost),
         MultiFab(ba_fy, dm, 1, nghost),
         MultiFab(ba_fz, dm, 1, nghost)
      };

      std::array<MultiFab,3> B_f = {
         MultiFab(ba_fx, dm, 1, nghost),
         MultiFab(ba_fy, dm, 1, nghost),
         MultiFab(ba_fz, dm, 1, nghost)
      };

      /*
      // Distributed matrices for curl operators
      std::array<LayoutData<matrix<Real>>,3> matA_B2E = {
         LayoutData<matrix<Real>>(ba_c,dm),
         LayoutData<matrix<Real>>(ba_c,dm),
         LayoutData<matrix<Real>>(ba_c,dm)
      };
      std::array<LayoutData<matrix<Real>>,3> matA_E2B = {
         LayoutData<matrix<Real>>(ba_c,dm),
         LayoutData<matrix<Real>>(ba_c,dm),
         LayoutData<matrix<Real>>(ba_c,dm)
      };
      LayoutData<matrix<Real>> matA_E2E(ba_c,dm);
      */
      
      // Distributed sparse matrices for curl operators
      std::array<LayoutData<sp_matrix<Real>>,3> matA_B2E = {
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm)
      };
      std::array<LayoutData<sp_matrix<Real>>,3> matA_E2B = {
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm),
         LayoutData<sp_matrix<Real>>(ba_c,dm)
      };
      LayoutData<sp_matrix<Real>> matA_E2E(ba_c,dm);
      
      Jp_c.setVal(0.0);
      B_n.setVal(0.0);
      E_n.setVal(0.0);
      for (int nn=0; nn<3; ++nn) {
         Jp_f[nn].setVal(0.0);
         B_f[nn].setVal(0.0);
      }
      Energy_c.setVal(0.0);
      
      RealBox real_box ({x_min,y_min,z_min}, {x_max,y_max,z_max});
      
      Geometry geom;
      geom.define(box_c, real_box, CoordSys::cartesian, periodicity);
      
      GpuArray<Real,3> dx = geom.CellSizeArray();
      
      for (MFIter mfi(E_n); mfi.isValid(); ++mfi) {
         const Box&
            bx_n = mfi.tilebox(AMReXConst::btype_n),
            bx_fx = mfi.tilebox(AMReXConst::btype_fx),
            bx_fy = mfi.tilebox(AMReXConst::btype_fy),
            bx_fz = mfi.tilebox(AMReXConst::btype_fz);
         const Array4<Real>&
            En_array = E_n.array(mfi),
            Bf_array_x = B_f[0].array(mfi),
            Bf_array_y = B_f[1].array(mfi),
            Bf_array_z = B_f[2].array(mfi);

         // En_array(0,0,0,0) = 0;
         // En_array(0,0,0,1) = 1;
         // En_array(0,0,0,2) = 0;
         // Bf_array_x(0,0,0) = 0;
         // Bf_array_y(0,0,0) = 1;
         // Bf_array_z(0,0,0) = 0;
         
         ParallelFor(bx_n, [&](int ii, int jj, int kk) {
            // Real
            //    x = ii*dx[0] + x_min;
            //    y = jj*dx[1] + y_min,
            //    z = kk*dx[2] + z_min;
            // En_array(ii,jj,kk,0) = 0.0;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            // En_array(ii,jj,kk,0) = jj;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            // En_array(ii,jj,kk,1) = std::sin(2*M_PI*x/(x_max - x_min));//*std::sin(y)*std::sin(z);
            // En_array(ii,jj,kk,2) = std::cos(2*M_PI*x/(x_max - x_min));//*std::cos(y)*std::cos(z);

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
      
      // Fix non cell-centred data periodicity so that last valid point
      // is equal to first along each dimension that is both periodic and not cell-centred
      // i.e. face data is modified only along the face dimension
      //         (x-face data only modifies in x-dimension),
      //      edge data is modified only along non-edge dimensions
      //         (x-directed edge data modifies in y- and z-dimensions),
      //      node data is modified along all dimensions,
      //      cell-centred data is not modified at all.
      // Provided that the respective dimension is also periodic
      //
      // (FillBoundary does not do this, as only invalid (ghost) data is modified
      // and the far ends are not identified as invalid)
      //
      // N.B. Theoretically this should only be necessary to run at initialisation of each field variable
      // At future times this should hold during evolution
      // If this rule is violated at any future time then this indicates a bug in the code
      //
      // !!! WARNING: Current implementation assumes only one box for entire domain!!!
      if (geom.periodicity().isAnyPeriodic()) {
         node_period(E_n, geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            node_period(B_f[nn], geom.periodicity());
         }
      }
      
      // Complete boundary conditions
      E_n.FillBoundary(geom.periodicity());
      for (int nn=0; nn<3; ++nn) {
         B_f[nn].FillBoundary(geom.periodicity());
      }

      // Generate curl operators
      for (MFIter mfi(E_n); mfi.isValid(); ++mfi) {
         const Box&
            bx_n = mfi.tilebox(AMReXConst::btype_n),
            bx_n_ghost = grow(bx_n, nghost);

         const IntVect
            len_n = bx_n.length(),
            len_n_ghost = bx_n_ghost.length();
         
         const int
            total_n = len_n[0]*len_n[1]*len_n[2],
            total_n_ghost = len_n_ghost[0]*len_n_ghost[1]*len_n_ghost[2];
         
         const std::array<sp_matrix<Real>,3> operator_curl_B2E = get_curl_f2n_operator_sparse(bx_n, nghost, dx);
         const std::array<sp_matrix<Real>,3> operator_curl_E2B = get_curl_n2f_operator_sparse(bx_n, nghost, dx);
         
         for (int ii=0; ii<3; ++ii) {
            matA_B2E[ii][mfi] = operator_curl_B2E[ii];
            matA_E2B[ii][mfi] = operator_curl_E2B[ii];
         }

         // matA_E2E[mfi] = matrix<Real>(total_n_ghost, total_n, 0.0);
         matA_E2E[mfi] = sp_matrix<Real>(27*total_n_ghost, total_n_ghost, total_n);
         
         /*
         auto matA_Bx2E_mf = matA_B2E[0][mfi];
         auto matA_By2E_mf = matA_B2E[1][mfi];
         auto matA_Bz2E_mf = matA_B2E[2][mfi];
         auto matA_E2Bx_mf = matA_E2B[0][mfi];
         auto matA_E2By_mf = matA_E2B[1][mfi];
         auto matA_E2Bz_mf = matA_E2B[2][mfi];
         auto matA_Bx2E_sp_mf = matA_B2E_sp[0][mfi];
         auto matA_By2E_sp_mf = matA_B2E_sp[1][mfi];
         auto matA_Bz2E_sp_mf = matA_B2E_sp[2][mfi];
         auto matA_E2Bx_sp_mf = matA_E2B_sp[0][mfi];
         auto matA_E2By_sp_mf = matA_E2B_sp[1][mfi];
         auto matA_E2Bz_sp_mf = matA_E2B_sp[2][mfi];
         
         const FArrayBox& En_data { E_n[mfi] };
         const FArrayBox& Bfx_data { B_f[0][mfi] };
         const FArrayBox& Bfy_data { B_f[1][mfi] };
         const FArrayBox& Bfz_data { B_f[2][mfi] };

         amrex::FArrayBox& curl_Bn_data { curl_B_n[mfi] };
         amrex::FArrayBox& curl_Efx_data { curl_E_fx[mfi] };
         amrex::FArrayBox& curl_Efy_data { curl_E_fy[mfi] };
         amrex::FArrayBox& curl_Efz_data { curl_E_fz[mfi] };
         amrex::FArrayBox& curl_Bn_sp_data { curl_B_n_sp[mfi] };
         amrex::FArrayBox& curl_Efx_sp_data { curl_E_fx_sp[mfi] };
         amrex::FArrayBox& curl_Efy_sp_data { curl_E_fy_sp[mfi] };
         amrex::FArrayBox& curl_Efz_sp_data { curl_E_fz_sp[mfi] };

         const amrex::IntVect
            len_En = En_data.length(),
            len_Bfx = Bfx_data.length(),
            len_Bfy = Bfy_data.length(),
            len_Bfz = Bfz_data.length(),
            len_cBn = curl_Bn_data.length(),
            len_cEfx = curl_Efx_data.length(),
            len_cEfy = curl_Efy_data.length(),
            len_cEfz = curl_Efz_data.length(),
            len_cBn_sp = curl_Bn_data.length(),
            len_cEfx_sp = curl_Efx_sp_data.length(),
            len_cEfy_sp = curl_Efy_sp_data.length(),
            len_cEfz_sp = curl_Efz_sp_data.length();
            
         const size_t
            total_En = math::product(len_En),
            total_Bfx = math::product(len_Bfx),
            total_Bfy = math::product(len_Bfy),
            total_Bfz = math::product(len_Bfz),
            total_cBn = math::product(len_cBn),
            total_cEfx = math::product(len_cEfx),
            total_cEfy = math::product(len_cEfy),
            total_cEfz = math::product(len_cEfz),
            total_cBn_sp = math::product(len_cBn_sp),
            total_cEfx_sp = math::product(len_cEfx_sp),
            total_cEfy_sp = math::product(len_cEfy_sp),
            total_cEfz_sp = math::product(len_cEfz_sp);

         const std::span<const Real>
            En_span(En_data.dataPtr(0), 3*total_En),
            Bfx_span(Bfx_data.dataPtr(0), total_Bfx),
            Bfy_span(Bfy_data.dataPtr(0), total_Bfy),
            Bfz_span(Bfz_data.dataPtr(0), total_Bfz),
            Enx_span(&(En_span[0]), total_En),
            Eny_span(&(En_span[total_En]), total_En),
            Enz_span(&(En_span[2*total_En]), total_En);
         
         std::span<Real>
            cBn_span(curl_Bn_data.dataPtr(0), 3*total_cBn),
            cEfx_span(curl_Efx_data.dataPtr(0), total_cEfx),
            cEfy_span(curl_Efy_data.dataPtr(0), total_cEfy),
            cEfz_span(curl_Efz_data.dataPtr(0), total_cEfz),
            cBn_sp_span(curl_Bn_sp_data.dataPtr(0), 3*total_cBn_sp),
            cEfx_sp_span(curl_Efx_sp_data.dataPtr(0), total_cEfx_sp),
            cEfy_sp_span(curl_Efy_sp_data.dataPtr(0), total_cEfy_sp),
            cEfz_sp_span(curl_Efz_sp_data.dataPtr(0), total_cEfz_sp),
            cBnx_span(&(cBn_span[0]), total_cBn),
            cBny_span(&(cBn_span[total_cBn]), total_cBn),
            cBnz_span(&(cBn_span[2*total_cBn]), total_cBn),
            cBnx_sp_span(&(cBn_sp_span[0]), total_cBn),
            cBny_sp_span(&(cBn_sp_span[total_cBn]), total_cBn),
            cBnz_sp_span(&(cBn_sp_span[2*total_cBn]), total_cBn);
         
         matA_Bx2E_mf.mmult_add(cBn_span, Bfx_span, 1);
         matA_By2E_mf.mmult_add(cBn_span, Bfy_span, 1);
         matA_Bz2E_mf.mmult_add(cBn_span, Bfz_span, 1);
         matA_E2Bx_mf.mmult_add(cEfx_span, En_span, 1);
         matA_E2By_mf.mmult_add(cEfy_span, En_span, 1);
         matA_E2Bz_mf.mmult_add(cEfz_span, En_span, 1);
         
         matA_Bx2E_sp_mf.mmult_add(cBn_sp_span, Bfx_span, 1);
         matA_By2E_sp_mf.mmult_add(cBn_sp_span, Bfy_span, 1);
         matA_Bz2E_sp_mf.mmult_add(cBn_sp_span, Bfz_span, 1);
         matA_E2Bx_sp_mf.mmult_add(cEfx_sp_span, En_span, 1);
         matA_E2By_sp_mf.mmult_add(cEfy_sp_span, En_span, 1);
         matA_E2Bz_sp_mf.mmult_add(cEfz_sp_span, En_span, 1);
         
         Print() << "Done!" << std::endl;
         */
      }
      
      // const IntVect& sym_dir = AMReXConst::btype_ex;
      
      // Print() << "E_n: " << sym_test(E_n,sym_dir) << std::endl
      //         << "B_fx: " << sym_test(B_f[0],sym_dir) << std::endl
      //         << "B_fy: " << sym_test(B_f[1],sym_dir) << std::endl
      //         << "B_fz: " << sym_test(B_f[2],sym_dir) << std::endl;
      
      Real time = 0.0;
      
      for (int step=0; step<steps; ++step) {
         Print()  << std::endl << "Step: " << step << std::endl;
         
         if (step % save_steps == 0) {
            const std::string& pltfile = amrex::Concatenate("plt", step, 5);
            
            MultiFab plt_Fab(ba_c, dm, 9, nghost);
            
            B_c = face2cell(B_f);
            E_c = node2cell(E_n);
            
            B_c.FillBoundary(geom.periodicity());
            E_c.FillBoundary(geom.periodicity());
            
            Energy_c = compute_EM_energy(B_c,E_c);

            Real
               total_B_energy = 0.0,
               total_E_energy = 0.0,
               total_EM_energy = 0.0;

            for (MFIter mfi(Energy_c); mfi.isValid(); ++mfi) {
               const Box& bx_c = mfi.tilebox(AMReXConst::btype_c);
               const Array4<Real>& Energy_c_array = Energy_c.array(mfi);

               ParallelFor(bx_c, [&](int ii, int jj, int kk) {
                  total_B_energy += Energy_c_array(ii,jj,kk,0);
                  total_E_energy += Energy_c_array(ii,jj,kk,1);
                  total_EM_energy += Energy_c_array(ii,jj,kk,2);
               });
            }

            datalog << std::setw(Log::stepWidth) << step
                    << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_B_energy
                    << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_E_energy
                    << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_EM_energy << std::endl;
            
            MultiFab::Copy(plt_Fab, B_c, 0, 0, 3, nghost);
            MultiFab::Copy(plt_Fab, E_c, 0, 3, 3, nghost);
            MultiFab::Copy(plt_Fab, Energy_c, 0, 6, 3, nghost);
            
            WriteSingleLevelPlotfileHDF5(pltfile, plt_Fab, {"Bx","By","Bz","Ex","Ey","Ez","B_Energy","E_Energy","EM_Energy"}, geom, time, step);
         }

         // gmres_step(B_f, E_n, dx, dt, theta, geom.periodicity(), rtol, atol, verbosity);
         gmres_step_matrix(B_f, E_n, matA_B2E, matA_E2B, matA_E2E, dx, dt, theta, geom.periodicity(), rtol, atol, verbosity);
         
         // Print() << "E_n: " << sym_test(E_n,sym_dir) << std::endl
         //         << "B_fx: " << sym_test(B_f[0],sym_dir) << std::endl
         //         << "B_fy: " << sym_test(B_f[1],sym_dir) << std::endl
         //         << "B_fz: " << sym_test(B_f[2],sym_dir) << std::endl;

         // Complete boundary conditions
         E_n.FillBoundary(geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            B_f[nn].FillBoundary(geom.periodicity());
         }
         
         time += dt;
      }
      const std::string& pltfile = amrex::Concatenate("plt", steps, 5);
      
      MultiFab plt_Fab(ba_c, dm, 9, nghost);
      
      B_c = face2cell(B_f);
      E_c = node2cell(E_n);

      B_c.FillBoundary(geom.periodicity());
      E_c.FillBoundary(geom.periodicity());

      Energy_c = compute_EM_energy(B_c,E_c);

      Real
         total_B_energy = 0.0,
         total_E_energy = 0.0,
         total_EM_energy = 0.0;
      
      for (MFIter mfi(Energy_c); mfi.isValid(); ++mfi) {
         const Box& bx_c = mfi.tilebox(AMReXConst::btype_c);
         const Array4<Real>& Energy_c_array = Energy_c.array(mfi);
         
         ParallelFor(bx_c, [&](int ii, int jj, int kk) {
            total_B_energy += Energy_c_array(ii,jj,kk,0);
            total_E_energy += Energy_c_array(ii,jj,kk,1);
            total_EM_energy += Energy_c_array(ii,jj,kk,2);
         });
      }
      
      datalog << std::setw(Log::stepWidth) << steps
              << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_B_energy
              << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_E_energy
              << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_EM_energy << std::endl;
      
      MultiFab::Copy(plt_Fab, B_c, 0, 0, 3, nghost);
      MultiFab::Copy(plt_Fab, E_c, 0, 3, 3, nghost);
      MultiFab::Copy(plt_Fab, Energy_c, 0, 6, 3, nghost);
      
      WriteSingleLevelPlotfileHDF5(pltfile, plt_Fab, {"Bx","By","Bz","Ex","Ey","Ez","B_Energy","E_Energy","EM_Energy"}, geom, time, steps);
   }

   BL_PROFILE_VAR_STOP(pmain);
   Finalize();
}
