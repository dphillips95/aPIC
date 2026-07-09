#ifndef OPERATORS_H_
#define OPERATORS_H_

#include <AMReX_REAL.H>

using namespace amrex;

// Interpolators
MultiFab node2cell(const MultiFab& node_data);
MultiFab face2cell(const std::array<MultiFab,3>& face_data);
std::array<MultiFab,3> node2edge(const MultiFab& node_data);

// Curl operators
std::array<MultiFab,3> curl_e2f(const std::array<MultiFab,3>& edge_data, GpuArray<Real,3>& dx);
MultiFab curl_f2n(const std::array<MultiFab,3>& face_data, GpuArray<Real,3>& dx);

// Tests
bool sym_test(const MultiFab& mf, const IntVect& dir);

#endif
