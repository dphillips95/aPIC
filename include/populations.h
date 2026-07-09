#ifndef POPULATIONS_H_
#define POPULATIONS_H_

#include <AMReX_REAL.H>

using namespace amrex;

struct Population {
   Real mass = 0.0;
   Real charge = 0.0;
   Real temperature = 0.0;
   std::vector<Real> velocity = {0.0,0.0,0.0};
   Real density = 0.0;
   Real macro = 0.0;
};

#endif
