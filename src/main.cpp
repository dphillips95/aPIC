#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

using namespace amrex;

int main(int argc, char* argv[]) {
   Initialize(argc,argv);
   
   {

      constexpr int nghost = 0;
      
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
      Vector<std::string> pop_list;
      
      {
         // Get inputs with ParmParse
         ParmParse inp_m("main");

         inp_m.query("seed", seed);
         inp_m.query("dimensions", dimensions);
         inp_m.query("1v", v_1D);

         ParmParse inp_s("simulation");
         inp_s.get("steps", steps);
         inp_s.get("dt", dt);
         inp_s.get("pop_list", pop_list);
         inp_s.query("theta", theta);
         inp_s.query("rtol", rtol);
         inp_s.query("atol", atol);
         inp_s.query("Vdt_dx_cap", Vdt_dx_cap);
         inp_s.query("save_steps", save_steps);
         Print() << pop_list << std::endl;

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
      }
         
      IntVect
         dom_lo(0,0,0),
         dom_hi(x_size-1, y_size-1, z_size-1);
      
      Box domain(dom_lo, dom_hi);
      
      BoxArray ba(domain);
      
      DistributionMapping dm(ba);

      MultiFab mf(ba, dm, 1, nghost);
      
      RealBox real_box ({x_min,y_min,z_min}, {x_max,y_max,z_max});
      
      Geometry geom;
      geom.define(domain, real_box, amrex::CoordSys::cartesian, periodicity);
      
      GpuArray<Real,3> dx = geom.CellSizeArray();
      
      for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
         const Box& bx = mfi.validbox();
         const Array4<Real>& mf_array = mf.array(mfi);
         
         ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Real x = (i+0.5) * dx[0] + x_min;
            Real y = (j+0.5) * dx[1] + y_min;
            Real z = (k+0.5) * dx[2] + z_min;
            Real rsquared = x*x + y*y + z*z;
            mf_array(i,j,k) = std::exp(-rsquared);
            
         });
      }
      
      WriteSingleLevelPlotfile("plt001", mf, {"comp0"}, geom, 0., 0);
   }
   Finalize();
}
