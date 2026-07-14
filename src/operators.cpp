#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <AMReX_Print.H>

#include <operators.h>
#include <constants.h>
#include <math.h>

using namespace amrex;

// Interpolators: Node to Cell - Value in cell centre is calculated as average of value at all 8 nodes
MultiFab node2cell(const MultiFab& node_data) {

   int nvar = node_data.nComp();
   
   MultiFab cell_data(convert(node_data.boxArray(),AMReXConst::btype_c),node_data.distributionMap,nvar,node_data.n_grow);

   cell_data.setVal(0.0);
   
   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.validbox();
      const Array4<const Real>& n_array = node_data.const_array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx_c, nvar, [&](int ii, int jj, int kk, int nn) {
         c_array(ii,jj,kk,nn) += n_array(ii,jj,kk,nn);
         c_array(ii,jj,kk,nn) += n_array(ii,jj,kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii,jj+1,kk,nn);
         c_array(ii,jj,kk,nn) += n_array(ii,jj+1,kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj,kk,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj,kk+1,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj+1,kk,nn);
         c_array(ii,jj,kk,nn) += n_array(ii+1,jj+1,kk+1,nn);
         c_array(ii,jj,kk,nn) /= 8;
      });
   }

   return cell_data;
}

// Interpolators: Face to Cell - Each face stores single component of vector perpendicular to face; cell centre data is calculated as average in each component of the two faces of the cell containing the given component
MultiFab face2cell(const std::array<MultiFab,3>& face_data) {

   MultiFab cell_data(convert(face_data[0].boxArray(),AMReXConst::btype_c),face_data[0].distributionMap,3,face_data[0].n_grow);

   cell_data.setVal(0.0);

   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.validbox();
      const Array4<const Real>&
         f_array_x = face_data[0].const_array(mfi),
         f_array_y = face_data[1].const_array(mfi),
         f_array_z = face_data[2].const_array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         c_array(ii,jj,kk,0) += (f_array_x(ii,jj,kk) + f_array_x(ii+1,jj,kk))/2;
         c_array(ii,jj,kk,1) += (f_array_y(ii,jj,kk) + f_array_y(ii,jj+1,kk))/2;
         c_array(ii,jj,kk,2) += (f_array_z(ii,jj,kk) + f_array_z(ii,jj,kk+1))/2;
      });
   }

   return cell_data;
}

// Interpolators: Node to Edge - Edge data stores single component of vector data parallel to edge; edge data is calculated as average of given component from nodes at either end of edge
std::array<MultiFab,3> node2edge(const MultiFab& node_data) {

   std::array<MultiFab,3> edge_data = {
      MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ex),node_data.distributionMap, 1, node_data.n_grow),
      MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ey),node_data.distributionMap, 1, node_data.n_grow),
      MultiFab(convert(node_data.boxArray(),AMReXConst::btype_ez),node_data.distributionMap, 1, node_data.n_grow)
   };

   for (int nn=0; nn<3; ++nn) {
      edge_data[nn].setVal(0.0);
   }
   
   for (MFIter mfi(edge_data[0]); mfi.isValid(); ++mfi) {
      const Box&
         bx_ex = mfi.tilebox(AMReXConst::btype_ex),
         bx_ey = mfi.tilebox(AMReXConst::btype_ey),
         bx_ez = mfi.tilebox(AMReXConst::btype_ez);
      const Array4<const Real>& n_array = node_data.const_array(mfi);
      const Array4<Real>&
         e_array_x = edge_data[0].array(mfi),
         e_array_y = edge_data[1].array(mfi),
         e_array_z = edge_data[2].array(mfi);
      
      ParallelFor(bx_ex, [&](int ii, int jj, int kk) {
         e_array_x(ii,jj,kk) += (n_array(ii,jj,kk,0) + n_array(ii+1,jj,kk,0))/2;
      });
      
      ParallelFor(bx_ey, [&](int ii, int jj, int kk) {
         e_array_y(ii,jj,kk) += (n_array(ii,jj,kk,1) + n_array(ii,jj+1,kk,1))/2;
      });
      
      ParallelFor(bx_ez, [&](int ii, int jj, int kk) {
         e_array_z(ii,jj,kk) += (n_array(ii,jj,kk,2) + n_array(ii,jj,kk+1,2))/2;
      });
   }

   return edge_data;
}

// Diagnostics: Compute cell energy
MultiFab compute_energy(const MultiFab& B_c, const MultiFab& E_c) {

   MultiFab Energy_c = MultiFab(B_c.boxArray(),B_c.distributionMap, 1, B_c.n_grow);
   Energy_c.setVal(0.0);
   
   for(MFIter mfi(B_c); mfi.isValid(); ++mfi) {
      const Box& bx_c = mfi.fabbox();
      const Array4<const Real>&
         Bc_array = B_c.const_array(mfi),
         Ec_array = E_c.const_array(mfi);
      const Array4<Real>& Energy_c_array = Energy_c.array(mfi);
      
      ParallelFor(bx_c, [&](int ii, int jj, int kk) {
         Real
            B_mag = square(Bc_array(ii,jj,kk,0)) + square(Bc_array(ii,jj,kk,1)) + square(Bc_array(ii,jj,kk,2)),
            E_mag = square(Ec_array(ii,jj,kk,0)) + square(Ec_array(ii,jj,kk,1)) + square(Ec_array(ii,jj,kk,2));
         
         Energy_c_array(ii,jj,kk) += (B_mag/PhysConst::mu0 + E_mag*PhysConst::eps0)/2;
      });
   }

   return Energy_c;
}

// Boundaries: Fix Nodal Periodicity - Fixes nodal data (i.e. face, edge or node data) for periodic BCs so that final 'valid' node is equal to first 'valid' node
void node_period(MultiFab& mf, const Periodicity& period) {

   int nvar = mf.nComp();
   
   for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const Box&
         bx = mfi.validbox(),
         bx_full = mfi.fabbox();
      const Array4<Real>& mf_array = mf.array(mfi);
      const Dim3&
         int_valid_min = lbound(bx),
         int_valid_max = ubound(bx),
         int_min = lbound(bx_full),
         int_max = ubound(bx_full);
      
      const IntVect& btype = bx.type();
      
      if (period.isPeriodic(0) && btype[0] == 1) {
         for (int jj=int_min.y; jj<int_max.y; ++jj) {
            for (int kk=int_min.z; kk<int_max.z; ++kk) {
               for (int nn=0; nn<nvar; ++nn) {
                  mf_array(int_valid_max.x,jj,kk,nn) = mf_array(int_valid_min.x,jj,kk,nn);
               }
            }
         }
      }
      
      if (period.isPeriodic(1) && btype[1] == 1) {
         for (int ii=int_min.x; ii<int_max.x; ++ii) {
            for (int kk=int_min.z; kk<int_max.z; ++kk) {
               for (int nn=0; nn<nvar; ++nn) {
                  mf_array(ii,int_valid_max.y,kk,nn) = mf_array(ii,int_valid_min.y,kk,nn);
               }
            }
         }
      }
      
      if (period.isPeriodic(2) && btype[2] == 1) {
         for (int ii=int_min.x; ii<int_max.x; ++ii) {
            for (int jj=int_min.y; jj<int_max.y; ++jj) {
               for (int nn=0; nn<nvar; ++nn) {
                  mf_array(ii,jj,int_valid_max.z,nn) = mf_array(ii,jj,int_valid_min.z,nn);
               }
            }
         }
      }
   }
}

// Curl operators: Edge to Face - Curl is calculated at face centres using edges surrounding face
std::array<MultiFab,3> curl_e2f(const std::array<MultiFab,3>& edge_data, const GpuArray<Real,3>& dx) {

   std::array<MultiFab,3> face_curl = {
      MultiFab(convert(edge_data[0].boxArray(),AMReXConst::btype_fx),edge_data[0].distributionMap, 1, edge_data[0].n_grow),
      MultiFab(convert(edge_data[0].boxArray(),AMReXConst::btype_fy),edge_data[0].distributionMap, 1, edge_data[0].n_grow),
      MultiFab(convert(edge_data[0].boxArray(),AMReXConst::btype_fz),edge_data[0].distributionMap, 1, edge_data[0].n_grow)
   };

   for (int nn=0; nn<3;++nn) {
      face_curl[nn].setVal(0.0);
   }
   
   for (MFIter mfi(face_curl[0]); mfi.isValid(); ++mfi) {
      const Box&
         bx_fx = mfi.tilebox(AMReXConst::btype_fx),
         bx_fy = mfi.tilebox(AMReXConst::btype_fy),
         bx_fz = mfi.tilebox(AMReXConst::btype_fz);
      const Array4<const Real>&
         e_array_x = edge_data[0].const_array(mfi),
         e_array_y = edge_data[1].const_array(mfi),
         e_array_z = edge_data[2].const_array(mfi);
      const Array4<Real>&
         fc_array_x = face_curl[0].array(mfi),
         fc_array_y = face_curl[1].array(mfi),
         fc_array_z = face_curl[2].array(mfi);
      
      ParallelFor(bx_fx, [&](int ii, int jj, int kk) {
         fc_array_x(ii,jj,kk) += (e_array_z(ii,jj+1,kk) - e_array_z(ii,jj,kk))/dx[1];
         fc_array_x(ii,jj,kk) -= (e_array_y(ii,jj,kk+1) - e_array_y(ii,jj,kk))/dx[2];
      });
      
      ParallelFor(bx_fy, [&](int ii, int jj, int kk) {
         fc_array_y(ii,jj,kk) += (e_array_x(ii,jj,kk+1) - e_array_x(ii,jj,kk))/dx[2];
         fc_array_y(ii,jj,kk) -= (e_array_z(ii+1,jj,kk) - e_array_z(ii,jj,kk))/dx[0];
      });
      
      ParallelFor(bx_fz, [&](int ii, int jj, int kk) {
         fc_array_z(ii,jj,kk) += (e_array_y(ii+1,jj,kk) - e_array_y(ii,jj,kk))/dx[0];
         fc_array_z(ii,jj,kk) -= (e_array_x(ii,jj+1,kk) - e_array_x(ii,jj,kk))/dx[1];
      });
   }
   
   return face_curl;
}

// Curl operators: Face to Node - Curl is calculated at nodes component-wise using faces adjacent each edge connecting to node
MultiFab curl_f2n(const std::array<MultiFab,3>& face_data, const GpuArray<Real,3>& dx) {

   MultiFab node_curl(convert(face_data[0].boxArray(),AMReXConst::btype_n),face_data[0].distributionMap, 3, face_data[0].n_grow);
   
   node_curl.setVal(0.0);
   
   for (MFIter mfi(node_curl); mfi.isValid(); ++mfi) {
      const Box& bx_n = mfi.validbox();
      const Array4<const Real>&
         f_array_x = face_data[0].const_array(mfi),
         f_array_y = face_data[1].const_array(mfi),
         f_array_z = face_data[2].const_array(mfi);
      const Array4<Real>& nc_array = node_curl.array(mfi);
      
      ParallelFor(bx_n, [&](int ii, int jj, int kk) {
         nc_array(ii,jj,kk,0) += (f_array_z(ii-1,jj,kk) - f_array_z(ii-1,jj-1,kk))/(2*dx[1]);
         nc_array(ii,jj,kk,0) += (f_array_z(ii,jj,kk) - f_array_z(ii,jj-1,kk))/(2*dx[1]);
         nc_array(ii,jj,kk,0) -= (f_array_y(ii-1,jj,kk) - f_array_y(ii-1,jj,kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,0) -= (f_array_y(ii,jj,kk) - f_array_y(ii,jj,kk-1))/(2*dx[2]);
         
         nc_array(ii,jj,kk,1) += (f_array_x(ii,jj-1,kk) - f_array_x(ii,jj-1,kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,1) += (f_array_x(ii,jj,kk) - f_array_x(ii,jj,kk-1))/(2*dx[2]);
         nc_array(ii,jj,kk,1) -= (f_array_z(ii,jj-1,kk) - f_array_z(ii-1,jj-1,kk))/(2*dx[0]);
         nc_array(ii,jj,kk,1) -= (f_array_z(ii,jj,kk) - f_array_z(ii-1,jj,kk))/(2*dx[0]);
         
         nc_array(ii,jj,kk,2) += (f_array_y(ii,jj,kk-1) - f_array_y(ii-1,jj,kk-1))/(2*dx[0]);
         nc_array(ii,jj,kk,2) += (f_array_y(ii,jj,kk) - f_array_y(ii-1,jj,kk))/(2*dx[0]);
         nc_array(ii,jj,kk,2) -= (f_array_x(ii,jj,kk-1) - f_array_x(ii,jj-1,kk-1))/(2*dx[1]);
         nc_array(ii,jj,kk,2) -= (f_array_x(ii,jj,kk) - f_array_x(ii,jj-1,kk))/(2*dx[1]);
      });
   }

   return node_curl;
}

// Test: Symmetry - Checks multifab data is symmetric in given direction(s); symmetry direction(s) are given by '1' in 3D 'dir' vector.
// If all values of dir are 1 then tests if all values in multifab are constant
// If two values of dir are 1 then tests if data is 1D data, non-constant in only the remaining dimension
// If one value of dir is 1 then tests if data is 2D data, constant only in the given symmetry direction
// If no values of dir are 1 then test is skipped as no symmetry is being tested
bool sym_test(const MultiFab& mf, const IntVect& dir) {

   bool test = true;

   int nvar = mf.nComp();
   
   for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& dat = mf.const_array(mfi);
      const Dim3&
         int_min = lbound(bx),
         int_max = ubound(bx);
      
      for (int nn=0; nn<nvar; ++nn) {
         if (dir[0] == 1) {
            if (dir[1] == 1) {
               if (dir[2] == 1) {
                  Real init = dat(0,0,0,nn);
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                     Real init = dat(0,0,kk,nn);
                     for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                        for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            } else {
               if (dir[2] == 1) {
                  for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                     Real init = dat(0,jj,0,nn);
                     for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                     for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                        Real init = dat(0,jj,kk,nn);
                        for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            }
         } else {
            if (dir[1] == 1) {
               if (dir[2] == 1) {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     Real init = dat(ii,0,0,nn);
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                        Real init = dat(ii,0,kk,nn);
                        for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               }
            } else {
               if (dir[2] == 1) {
                  for (int ii=int_min.x; ii<int_max.x+1; ++ii) {
                     for (int jj=int_min.y; jj<int_max.y+1; ++jj) {
                        Real init = dat(ii,jj,0,nn);
                        for (int kk=int_min.z; kk<int_max.z+1; ++kk) {
                           if (dat(ii,jj,kk,nn) != init) {
                              test = false;
                              break;
                           }
                        }
                        if (test == false) {
                           break;
                        }
                     }
                     if (test == false) {
                        break;
                     }
                  }
               } else {
                  Print() << "Symmetry test has no symmetry directions!" << std::endl;
               }
            }
         }
         if (test == false) {
            break;
         }
      }
      if (test == false) {
         break;
      }
   }

   return test;
}
