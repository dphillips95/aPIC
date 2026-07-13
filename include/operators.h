#ifndef OPERATORS_H_
#define OPERATORS_H_

#include <AMReX_REAL.H>

// Interpolators
amrex::MultiFab node2cell(const amrex::MultiFab& node_data);
amrex::MultiFab face2cell(const std::array<amrex::MultiFab,3>& face_data);
std::array<amrex::MultiFab,3> node2edge(const amrex::MultiFab& node_data);

// Boundary Conditions
void node_period(amrex::MultiFab& mf, const amrex::Periodicity& period);

// Curl operators
std::array<amrex::MultiFab,3> curl_e2f(const std::array<amrex::MultiFab,3>& edge_data, amrex::GpuArray<amrex::Real,3>& dx);
amrex::MultiFab curl_f2n(const std::array<amrex::MultiFab,3>& face_data, amrex::GpuArray<amrex::Real,3>& dx);

// Tests
bool sym_test(const amrex::MultiFab& mf, const amrex::IntVect& dir);

#endif
