#ifndef POPULATIONS_H_
#define POPULATIONS_H_

#include <AMReX_REAL.H>

struct Population {
   amrex::Real mass = 0.0;
   amrex::Real charge = 0.0;
   amrex::Real temperature = 0.0;
   std::vector<amrex::Real> velocity = {0.0,0.0,0.0};
   amrex::Real density = 0.0;
   amrex::Real macro = 0.0;
};

#endif
