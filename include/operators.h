#ifndef OPERATORS_H_
#define OPERATORS_H_

#include <AMReX_REAL.H>

// Interpolators
amrex::MultiFab node2cell(const amrex::MultiFab& node_data, int nghost = -1);

amrex::MultiFab face2cell(const std::array<amrex::MultiFab,3>& face_data, int nghost = -1);

std::array<amrex::MultiFab,3> node2edge(const amrex::MultiFab& node_data,int nghost = -1);

// Diagnostics
amrex::MultiFab compute_energy(const amrex::MultiFab& B_c, const amrex::MultiFab& E_c, int nghost = -1);

// Boundary Conditions
void node_period(amrex::MultiFab& mf, const amrex::Periodicity& period);

// Derivative operators
std::array<amrex::MultiFab,3> curl_e2f(const std::array<amrex::MultiFab,3>& edge_data, const amrex::GpuArray<amrex::Real,3>& dx, int nghost = -1);

std::array<amrex::MultiFab,3> curl_n2f(const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, int nghost = -1);

amrex::MultiFab curl_f2n(const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx, int nghost = -1);

amrex::MultiFab div_f2c(const amrex::MultiFab& xface_data, const amrex::MultiFab& yface_data, const amrex::MultiFab& zface_data, const amrex::GpuArray<amrex::Real,3>& dx, int nghost = -1);

amrex::MultiFab div_n2c(const amrex::MultiFab& node_data, const amrex::GpuArray<amrex::Real,3>& dx, int nghost = -1);

// Tests
bool sym_test(const amrex::MultiFab& mf, const amrex::IntVect& dir);

#endif
