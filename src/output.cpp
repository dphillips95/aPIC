/*
Output methods for aPIC.

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

#include <constants.h>
#include <gmres.h>
#include <particles.h>
#include <output.h>

#include <AMReX_REAL.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_Print.H>

#include <vector>

using namespace amrex;

void initialise_datalog(std::ofstream& datalog) {
   datalog << std::setw(Log::stepWidth) << "Step"
           << std::setw(Log::datWidth) << "B_energy"
           << std::setw(Log::datWidth) << "E_energy"
           << std::setw(Log::datWidth) << "EM_energy"
           << std::setw(Log::datWidth) << "KE_energy"
           << std::setw(Log::datWidth) << "Total_energy" << std::endl;
}
   
void saveState(int step, Real time, const BE& EM_state, std::vector<myPContainer>& pContainer_list, const std::vector<Population>& pop_list, const Geometry& geom, std::ofstream& datalog) {
   static int save_count = 0;

   const MultiFab* E_n = &EM_state.getE_n_const();
   std::array<const MultiFab*,3> B_f = {
      &EM_state.getB_fx_const(),
      &EM_state.getB_fy_const(),
      &EM_state.getB_fz_const()
   };

   const int pop_count = pop_list.size();
   
   // Number of individual components in output MultiFab plt_fab
   // B + E + Energy_EM + Energy_KE + Energy_total + current_total
   // + pop_count*(density + current + KE + velocity + temperature)
   // = 3 + 3 + 3 + 1 + 1 + 3 + pop_count*(1 + 3 + 1 + 3 + 1)
   const int output_components = 14 + 9*pop_count;

   const int nghost = EM_state.nghost();
   const BoxArray& ba_c = EM_state.getBoxArray();
   const DistributionMapping& dm = EM_state.getDistributionMap();
   
   static MultiFab
      B_c(ba_c, dm, 3, nghost),
      E_c(ba_c, dm, 3, nghost),
      J_c(ba_c, dm, 3, nghost),
      EM_Energy_c(ba_c, dm, 3, nghost),
      KE_Energy_c(ba_c, dm, 1, nghost),
      total_Energy_c(ba_c, dm, 1, nghost);
   
   B_c.setVal(0.0);
   E_c.setVal(0.0);
   J_c.setVal(0.0);
   KE_Energy_c.setVal(0.0);
   EM_Energy_c.setVal(0.0);
   total_Energy_c.setVal(0.0);
   
   static std::vector<MultiFab> rho_c, Jp_c, vp_c, temp_c, KEp_Energy_c;
   
   for (int pp=0; pp<pop_count; ++pp) {
      if (save_count == 0) {
         rho_c.emplace_back(MultiFab(ba_c, dm, 1, nghost));
         Jp_c.emplace_back(MultiFab(ba_c, dm, 3, nghost));
         vp_c.emplace_back(MultiFab(ba_c, dm, 3, nghost));
         KEp_Energy_c.emplace_back(MultiFab(ba_c, dm, 3, nghost));
         temp_c.emplace_back(MultiFab(ba_c, dm, 1, nghost));
      }
      
      rho_c[pp].setVal(0.0);
      Jp_c[pp].setVal(0.0);
      vp_c[pp].setVal(0.0);
      KEp_Energy_c[pp].setVal(0.0);
      temp_c[pp].setVal(0.0);
   }
   
   const std::string& pltfile = amrex::Concatenate("plt", step, 5);
   
   MultiFab plt_Fab(ba_c, dm, output_components, nghost);
            
   B_c = face2cell(*B_f[0], *B_f[1], *B_f[2]);
   E_c = node2cell(*E_n);

   B_c.FillBoundary(geom.periodicity());
   E_c.FillBoundary(geom.periodicity());
            
   for (int pp=0; pp<pop_count; ++pp) {
      rho_c[pp].setVal(0.0);
      Jp_c[pp].setVal(0.0);
      vp_c[pp].setVal(0.0);
      temp_c[pp].setVal(0.0);
      accumulateDensityCurrentKE(rho_c[pp], Jp_c[pp], KEp_Energy_c[pp], pContainer_list[pp], pop_list[pp]);
      MultiFab::Copy(vp_c[pp], Jp_c[pp], 0, 0, 3, nghost);
      vp_c[pp].mult(1/pop_list[pp].charge, nghost);
      for (int nn=0; nn<3; ++nn) {
         MultiFab::Divide(vp_c[pp], rho_c[pp], 0, nn, 1, nghost);
      }
      accumulateTemperature(temp_c[pp], vp_c[pp], rho_c[pp], pContainer_list[pp], pop_list[pp]);

      rho_c[pp].FillBoundary(geom.periodicity());
      Jp_c[pp].FillBoundary(geom.periodicity());
      KEp_Energy_c[pp].FillBoundary(geom.periodicity());
      vp_c[pp].FillBoundary(geom.periodicity());
      temp_c[pp].FillBoundary(geom.periodicity());
      
      MultiFab::Add(J_c, Jp_c[pp], 0, 0, 3, nghost);
      MultiFab::Add(KE_Energy_c, KEp_Energy_c[pp], 0, 0, 1, nghost);
   }
            
   J_c.FillBoundary(geom.periodicity());
   KE_Energy_c.FillBoundary(geom.periodicity());
            
   EM_Energy_c = compute_EM_energy(B_c,E_c);

   MultiFab::LinComb(total_Energy_c, 1, KE_Energy_c, 0, 1, EM_Energy_c, 0, 0, 1, nghost);
            
   Real
      total_B_energy = compute_B_energy_total(*B_f[0], *B_f[1], *B_f[2], geom.periodicity()),
      total_E_energy = compute_E_energy_total(*E_n, geom.periodicity()),
      total_EM_energy = total_B_energy + total_E_energy,
      total_KE_energy = KE_Energy_c.sum(0),
      total_energy = total_EM_energy + total_KE_energy;
   
   datalog << std::setw(Log::stepWidth) << step
           << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_B_energy
           << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_E_energy
           << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_EM_energy
           << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_KE_energy
           << std::setw(Log::datWidth) << std::setprecision(Log::datPrecision) << total_energy << std::endl;

   for (size_t nn=0; nn<pContainer_list.size(); ++nn) {
      Print() << pop_list[nn].name << " count: " << pContainer_list[nn].TotalNumberOfParticles() << std::endl;
      Print() << pop_list[nn].name << " max speed: " << max_spd(pContainer_list[nn]) << std::endl;
   }
   
   Print() << "B_x range: " << B_f[0]->min(0) << " - " << B_f[0]->max(0) << std::endl;
   Print() << "B_y range: " << B_f[1]->min(0) << " - " << B_f[1]->max(0) << std::endl;
   Print() << "B_z range: " << B_f[2]->min(0) << " - " << B_f[2]->max(0) << std::endl;
   Print() << "E_x range: " << E_n->min(0) << " - " << E_n->max(0) << std::endl;
   Print() << "E_y range: " << E_n->min(1) << " - " << E_n->max(1) << std::endl;
   Print() << "E_z range: " << E_n->min(2) << " - " << E_n->max(2) << std::endl;
   Print() << "j_x range: " << J_c.min(0) << " - " << J_c.max(0) << std::endl;
   Print() << "j_y range: " << J_c.min(1) << " - " << J_c.max(1) << std::endl;
   Print() << "j_z range: " << J_c.min(2) << " - " << J_c.max(2) << std::endl;

   plt_Fab.setVal(0.0);
   
   int comps = 0;
   Vector<std::string> names;
   
   MultiFab::Copy(plt_Fab, B_c, 0, comps, 3, nghost);
   comps += 3; names.push_back("Bx"); names.push_back("By"); names.push_back("Bz");
   MultiFab::Copy(plt_Fab, E_c, 0, comps, 3, nghost);
   comps += 3; names.push_back("Ex"); names.push_back("Ey"); names.push_back("Ez");
   MultiFab::Copy(plt_Fab, J_c, 0, comps, 3, nghost);
   comps += 3; names.push_back("Jx"); names.push_back("Jy"); names.push_back("Jz");
   for (int pp=0; pp<pop_count; ++pp) {
      MultiFab::Copy(plt_Fab, rho_c[pp], 0, comps, 1, nghost);
      comps += 1; names.push_back("rho_" + pop_list[pp].name);
      MultiFab::Copy(plt_Fab, vp_c[pp], 0, comps, 3, nghost);
      comps += 3; names.push_back("vx_" + pop_list[pp].name); names.push_back("vy_" + pop_list[pp].name); names.push_back("vz_" + pop_list[pp].name);
      MultiFab::Copy(plt_Fab, Jp_c[pp], 0, comps, 3, nghost);
      comps += 3; names.push_back("Jx_" + pop_list[pp].name); names.push_back("Jy_" + pop_list[pp].name); names.push_back("Jz_" + pop_list[pp].name);
      MultiFab::Copy(plt_Fab, KEp_Energy_c[pp], 0, comps, 1, nghost);
      comps += 1; names.push_back("KE_" + pop_list[pp].name);
      MultiFab::Copy(plt_Fab, temp_c[pp], 0, comps, 1, nghost);
      comps += 1; names.push_back("temp_" + pop_list[pp].name);
   }
   MultiFab::Copy(plt_Fab, EM_Energy_c, 0, comps, 3, nghost);
   comps += 3; names.push_back("B_Energy"); names.push_back("E_Energy"); names.push_back("EM_Energy");
   MultiFab::Copy(plt_Fab, KE_Energy_c, 0, comps, 1, nghost);
   comps += 1; names.push_back("KE_Energy");
   MultiFab::Copy(plt_Fab, total_Energy_c, 0, comps, 1, nghost);
   comps += 1; names.push_back("total_Energy");
   
   WriteSingleLevelPlotfileHDF5(pltfile, plt_Fab, names, geom, time, step);
   
   ++save_count;
}
