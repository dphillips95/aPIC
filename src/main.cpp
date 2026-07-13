#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <constants.h>
#include <operators.h>
#include <populations.h>

using namespace amrex;

int main(int argc, char* argv[]) {
   Initialize(argc,argv);
   
   {

      constexpr int nghost = 1;
      constexpr IntVect vectghost(nghost,nghost,nghost);
      
      int x_size, y_size, z_size;
      Real
         x_min = 0.0, x_max = 1.0,
         y_min = 0.0, y_max = 1.0,
         z_min = 0.0, z_max = 1.0;

      Array<int,3> periodicity = {false,false,false};

      int max_grid_size = 10;

      int seed = 0, dimensions = 3;
      bool v_1D = false;
      int steps, save_steps = 0;
      Real dt, theta = 0.5, rtol = 1e-15, atol = 0, Vdt_dx_cap = 0.75;
      std::vector<std::string> pop_name_list, Btype, Etype;
      Real Bx = 0, By = 0, Bz = 0, Ex = 0, Ey = 0, Ez = 0,
         rand_Bx_min = 0, rand_Bx_max = 0,
         rand_By_min = 0, rand_By_max = 0,
         rand_Bz_min = 0, rand_Bz_max = 0;
      Real mass_ratio = PhysConst::m_p/PhysConst::m_e;

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
         
         ParmParse inp_d("domain");
            
         inp_d.get("x_size", x_size);
         inp_d.get("y_size", y_size);
         inp_d.get("z_size", z_size);
         
         inp_d.query("x_min", x_min);
         inp_d.query("x_max", x_max);
         inp_d.query("y_min", y_min);
         inp_d.query("y_max", y_max);
         inp_d.query("z_min", z_min);
         inp_d.query("z_max", z_max);

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
      
      Box box_c(IntVect{0,0,0}, IntVect{x_size-1, y_size-1, z_size-1}); // cell-centred
      
      BoxArray
         ba_c(box_c), // cell-centred, others generated via conversion as needed
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
         E_c(ba_c, dm, 3, nghost);

      std::array<MultiFab,3> Jp_f = {
         MultiFab(ba_fx, dm, 1, nghost),
         MultiFab(ba_fy, dm, 1, nghost),
         MultiFab(ba_fz, dm, 1, nghost)};

      std::array<MultiFab,3> B_f = {
         MultiFab(ba_fx, dm, 1, nghost),
         MultiFab(ba_fy, dm, 1, nghost),
         MultiFab(ba_fz, dm, 1, nghost)};
      
      Jp_c.setVal(0.0);
      B_n.setVal(0.0);
      E_n.setVal(0.0);
      for (int nn=0; nn<3; ++nn) {
         Jp_f[nn].setVal(0.0);
         B_f[nn].setVal(0.0);
      }
      
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
         
         ParallelFor(bx_n, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
            Real
               x = ii*dx[0] + x_min,
               y = jj*dx[1] + y_min,
               z = kk*dx[2] + z_min;
            // En_array(ii,jj,kk,0) = 0.0;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            // En_array(ii,jj,kk,0) = jj;
            // En_array(ii,jj,kk,1) = 0.0;
            // En_array(ii,jj,kk,2) = 0.0;
            En_array(ii,jj,kk,1) = std::sin(x);//*std::sin(y)*std::sin(z);
            En_array(ii,jj,kk,2) = std::cos(x);//*std::cos(y)*std::cos(z);
         });
         
         ParallelFor(bx_fx, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
            Real
               x = ii*dx[0] + x_min,
               y = (jj+0.5)*dx[1] + y_min,
               z = (kk+0.5)*dx[2] + z_min;
            // Bf_array_x(ii,jj,kk) = 0.0;
         });

         ParallelFor(bx_fy, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
            Real
               x = (ii+0.5)*dx[0] + x_min,
               y = jj*dx[1] + y_min,
               z = (kk+0.5)*dx[2] + z_min;
            // Bf_array_y(ii,jj,kk) = 0.0;
         });

         ParallelFor(bx_fz, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
            Real
               x = (ii+0.5)*dx[0] + x_min,
               y = (jj+0.5)*dx[1] + y_min,
               z = kk*dx[2] + z_min;
            // Bf_array_z(ii,jj,kk) = ii;
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
            
            MultiFab plt_Fab(ba_c, dm, 6, nghost);

            B_c = face2cell(B_f);
            E_c = node2cell(E_n);

            B_c.FillBoundary(geom.periodicity());
            E_c.FillBoundary(geom.periodicity());
            
            MultiFab::Copy(plt_Fab, B_c, 0, 0, 3, nghost);
            MultiFab::Copy(plt_Fab, E_c, 0, 3, 3, nghost);               
            
            WriteSingleLevelPlotfile(pltfile, plt_Fab, {"Bx","By","Bz","Ex","Ey","Ez"}, geom, time, step);
         }
         
         MultiFab curlB_n = curl_f2n(B_f, dx);
         std::array<MultiFab,3> E_e = node2edge(E_n);
         for (int nn=0; nn<3; ++nn) {
            E_e[nn].FillBoundary(geom.periodicity());
         }
         std::array<MultiFab,3> curlE_f = curl_e2f(E_e, dx);
         
         curlB_n.FillBoundary(geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            curlE_f[nn].FillBoundary(geom.periodicity());
         }

         // Print() << "curlB_n: " << sym_test(curlB_n,sym_dir) << std::endl
         //         << "E_ex: " << sym_test(E_e[0],sym_dir) << std::endl
         //         << "E_ey: " << sym_test(E_e[1],sym_dir) << std::endl
         //         << "E_ez: " << sym_test(E_e[2],sym_dir) << std::endl
         //         << "curlE_fx: " << sym_test(curlE_f[0],sym_dir) << std::endl
         //         << "curlE_fy: " << sym_test(curlE_f[1],sym_dir) << std::endl
         //         << "curlE_fz: " << sym_test(curlE_f[2],sym_dir) << std::endl;
         
         // Print() << "Step: " << step << std::endl
         //         << "B_f:" << std::endl
         //         << B_f[0].sum() + B_f[1].sum() + B_f[2].sum() << std::endl
         //         << "E_n:" << std::endl
         //         << E_n.sum() << std::endl
         //         << "E_e" << std::endl
         //         << E_e[0].sum() + E_e[1].sum() + E_e[2].sum() << std::endl
         //         << "curl B_n:" << std::endl
         //         << curlB_n.sum() << std::endl
         //         << "curl E_f:" << std::endl
         //         << curlE_f[0].sum() + curlE_f[1].sum() + curlE_f[2].sum() << std::endl << std::endl;
         
         for (MFIter mfi(E_n); mfi.isValid(); ++mfi) {
            const Box&
               bx_n = mfi.tilebox(),
               bx_fx = mfi.tilebox(AMReXConst::btype_fx),
               bx_fy = mfi.tilebox(AMReXConst::btype_fy),
               bx_fz = mfi.tilebox(AMReXConst::btype_fz);
            const Array4<Real>&
               En_array = E_n.array(mfi),
               Bnc_array = curlB_n.array(mfi),
               Bf_array_x = B_f[0].array(mfi),
               Bf_array_y = B_f[1].array(mfi),
               Bf_array_z = B_f[2].array(mfi),
               Efc_array_x = curlE_f[0].array(mfi),
               Efc_array_y = curlE_f[1].array(mfi),
               Efc_array_z = curlE_f[2].array(mfi);
            
            // Print() << "curl B_n: " << std::endl;
            ParallelFor(bx_n, 3, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk, int nn) {
               // Print() << ii << "," << jj << "," << kk << "," << nn << ": " << Bnc_array(ii,jj,kk,nn) << std::endl;
               En_array(ii,jj,kk,nn) += dt*Bnc_array(ii,jj,kk,nn)*(PhysConst::c*PhysConst::c);
            });
               
            ParallelFor(bx_fx, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
               Bf_array_x(ii,jj,kk) -= dt*Efc_array_x(ii,jj,kk);
            });
            
            ParallelFor(bx_fy, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
               Bf_array_y(ii,jj,kk) -= dt*Efc_array_y(ii,jj,kk);
            });
            
            ParallelFor(bx_fz, [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) {
               Bf_array_z(ii,jj,kk) -= dt*Efc_array_z(ii,jj,kk);
            });
         }
         
         E_n.FillBoundary(geom.periodicity());
         for (int nn=0; nn<3; ++nn) {
            B_f[nn].FillBoundary(geom.periodicity());
         }

      // Print() << "E_n: " << sym_test(E_n,sym_dir) << std::endl
      //         << "B_fx: " << sym_test(B_f[0],sym_dir) << std::endl
      //         << "B_fy: " << sym_test(B_f[1],sym_dir) << std::endl
      //         << "B_fz: " << sym_test(B_f[2],sym_dir) << std::endl;
         
         time += dt;
      }
      const std::string& pltfile = amrex::Concatenate("plt", steps, 5);
      
      MultiFab plt_Fab(ba_c, dm, 6, nghost);
      
      B_c = face2cell(B_f);
      E_c = node2cell(E_n);

      B_c.FillBoundary(geom.periodicity());
      E_c.FillBoundary(geom.periodicity());
      
      MultiFab::Copy(plt_Fab, B_c, 0, 0, 3, nghost);
      MultiFab::Copy(plt_Fab, E_c, 0, 3, 3, nghost);
      
      WriteSingleLevelPlotfile(pltfile, plt_Fab, {"Bx","By","Bz","Ex","Ey","Ez"}, geom, time, steps);
   }
   Finalize();
}
