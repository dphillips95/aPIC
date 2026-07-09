#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include <AMReX_Print.H>

#include <operators.h>

using namespace amrex;

// Interpolators: Node to Cell
MultiFab node2cell(const MultiFab& node_data) {

   int nvar = node_data.nComp();
   
   MultiFab cell_data(convert(node_data.boxArray(),IntVect{0,0,0}),node_data.distributionMap,nvar,node_data.n_grow);

   cell_data.setVal(0.0);
   
   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& n_array = node_data.array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx, nvar, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk, int nn) {
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

MultiFab face2cell(const std::array<MultiFab,3>& face_data) {

   MultiFab cell_data(convert(face_data[0].boxArray(),IntVect{0,0,0}),face_data[0].distributionMap,3,face_data[0].n_grow);

   cell_data.setVal(0.0);

   for (MFIter mfi(cell_data); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>&
         f_array_x = face_data[0].array(mfi),
         f_array_y = face_data[1].array(mfi),
         f_array_z = face_data[2].array(mfi);
      const Array4<Real>& c_array = cell_data.array(mfi);

      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         c_array(ii,jj,kk,0) += (f_array_x(ii,jj,kk) + f_array_x(ii+1,jj,kk))/2;
         c_array(ii,jj,kk,1) += (f_array_y(ii,jj,kk) + f_array_y(ii,jj+1,kk))/2;
         c_array(ii,jj,kk,2) += (f_array_z(ii,jj,kk) + f_array_z(ii,jj,kk+1))/2;
      });
   }

   return cell_data;
}

// Interpolators: Node to Edge
std::array<MultiFab,3> node2edge(const MultiFab& node_data) {

   std::array<MultiFab,3> edge_data = {
      MultiFab(convert(node_data.boxArray(),IntVect{0,1,1}),node_data.distributionMap, 1, node_data.n_grow),
      MultiFab(convert(node_data.boxArray(),IntVect{1,0,1}),node_data.distributionMap, 1, node_data.n_grow),
      MultiFab(convert(node_data.boxArray(),IntVect{1,1,0}),node_data.distributionMap, 1, node_data.n_grow)
   };

   for (int nn=0; nn<3; ++nn) {
      edge_data[nn].setVal(0.0);
   }
   
   for (MFIter mfi(edge_data[0]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& n_array = node_data.array(mfi);
      const Array4<Real>& e_array_x = edge_data[0].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         e_array_x(ii,jj,kk) += (n_array(ii,jj,kk,0) + n_array(ii+1,jj,kk,0))/2;
      });
   }

   for (MFIter mfi(edge_data[1]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& n_array = node_data.array(mfi);
      const Array4<Real>& e_array_y = edge_data[1].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         e_array_y(ii,jj,kk) += (n_array(ii,jj,kk,1) + n_array(ii,jj+1,kk,1))/2;
      });
   }

   for (MFIter mfi(edge_data[2]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& n_array = node_data.array(mfi);
      const Array4<Real>& e_array_z = edge_data[2].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         e_array_z(ii,jj,kk) += (n_array(ii,jj,kk,2) + n_array(ii,jj,kk+1,2))/2;
      });
   }

   return edge_data;
}

// Curl operators: Edge to Face
std::array<MultiFab,3> curl_e2f(const std::array<MultiFab,3>& edge_data, GpuArray<Real,3>& dx) {

   std::array<MultiFab,3> face_curl = {
      MultiFab(convert(edge_data[0].boxArray(),IntVect{1,0,0}),edge_data[0].distributionMap, 1, edge_data[0].n_grow),
      MultiFab(convert(edge_data[0].boxArray(),IntVect{0,1,0}),edge_data[0].distributionMap, 1, edge_data[0].n_grow),
      MultiFab(convert(edge_data[0].boxArray(),IntVect{0,0,1}),edge_data[0].distributionMap, 1, edge_data[0].n_grow)
   };

   for (int nn=0; nn<3;++nn) {
      face_curl[0].setVal(0.0);
   }
   
   for (MFIter mfi(face_curl[0]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>&
         e_array_y = edge_data[1].array(mfi),
         e_array_z = edge_data[2].array(mfi);
      const Array4<Real>& fc_array_x = face_curl[0].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         fc_array_x(ii,jj,kk) += (e_array_z(ii,jj,kk) - e_array_z(ii,jj-1,kk))/dx[1];
         fc_array_x(ii,jj,kk) -= (e_array_y(ii,jj,kk) - e_array_y(ii,jj,kk-1))/dx[2];
      });
   }

   for (MFIter mfi(face_curl[1]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>&
         e_array_x = edge_data[0].array(mfi),
         e_array_z = edge_data[2].array(mfi);
      const Array4<Real>& fc_array_y = face_curl[1].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         fc_array_y(ii,jj,kk) += (e_array_x(ii,jj,kk) - e_array_x(ii,jj,kk-1))/dx[2];
         fc_array_y(ii,jj,kk) -= (e_array_z(ii,jj,kk) - e_array_z(ii-1,jj,kk))/dx[0];
      });
   }

   for (MFIter mfi(face_curl[2]); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>&
         e_array_x = edge_data[0].array(mfi),
         e_array_y = edge_data[1].array(mfi);
      const Array4<Real>& fc_array_z = face_curl[2].array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         fc_array_z(ii,jj,kk) += (e_array_y(ii,jj,kk) - e_array_y(ii-1,jj,kk))/dx[0];
         fc_array_z(ii,jj,kk) -= (e_array_x(ii,jj,kk) - e_array_x(ii,jj-1,kk))/dx[1];
      });
   }
   
   return face_curl;
}

// Curl operators: Face to Node
MultiFab curl_f2n(const std::array<MultiFab,3>& face_data, GpuArray<Real,3>& dx) {

   MultiFab node_curl(convert(face_data[0].boxArray(),IntVect{1,1,1}),face_data[0].distributionMap, 3, face_data[0].n_grow);
   
   node_curl.setVal(0.0);
   
   for (MFIter mfi(node_curl); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>&
         f_array_x = face_data[0].array(mfi),
         f_array_y = face_data[1].array(mfi),
         f_array_z = face_data[2].array(mfi);
      const Array4<Real>& nc_array = node_curl.array(mfi);
      
      ParallelFor(bx, [=] AMREX_GPU_DEVICE (int ii, int jj, int kk) {
         // Print() << ii << "," << jj << "," << kk << std::endl
         //         << "f_array_x: " << std::endl
         //         << "(" << ii << "," << jj << "," << kk << "): " << f_array_x(ii,jj,kk) << std::endl
         //         << "(" << ii << "," << jj << "," << kk-1 << "): " << f_array_x(ii,jj,kk-1) << std::endl
         //         << "(" << ii << "," << jj-1 << "," << kk << "): " << f_array_x(ii,jj-1,kk) << std::endl
         //         << "(" << ii << "," << jj-1 << "," << kk-1 << "): " << f_array_x(ii,jj-1,kk-1) << std::endl << std::endl
         //         << "f_array_y: " << std::endl
         //         << "(" << ii << "," << jj << "," << kk << "): " << f_array_y(ii,jj,kk) << std::endl
         //         << "(" << ii << "," << jj << "," << kk-1 << "): " << f_array_y(ii,jj,kk-1) << std::endl
         //         << "(" << ii-1 << "," << jj << "," << kk << "): " << f_array_y(ii-1,jj,kk) << std::endl
         //         << "(" << ii-1 << "," << jj << "," << kk-1 << "): " << f_array_y(ii-1,jj,kk-1) << std::endl << std::endl
         //         << "f_array_z: " << std::endl
         //         << "(" << ii << "," << jj << "," << kk << "): " << f_array_z(ii,jj,kk) << std::endl
         //         << "(" << ii << "," << jj-1 << "," << kk << "): " << f_array_z(ii,jj-1,kk) << std::endl
         //         << "(" << ii-1 << "," << jj << "," << kk << "): " << f_array_z(ii-1,jj,kk) << std::endl
         //         << "(" << ii-1 << "," << jj-1 << "," << kk << "): " << f_array_z(ii-1,jj-1,kk) << std::endl << std::endl;
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

// Test: Symmetry
bool sym_test(const MultiFab& mf, const IntVect& dir) {

   bool test = true;

   int nvar = mf.nComp();
   
   for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.validbox();
      const Array4<const Real>& dat = mf.array(mfi);
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
