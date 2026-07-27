/*
-- INSERT FILE DESCRIPTION --

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

#ifndef MATRIX_H_
#define MATRIX_H_

#include <numeric>
#include <span>

template <typename T>
class matrix {
private:
   int nrows,ncols;
   std::vector<T> dat;
public:
   matrix(int n, int m) : nrows(n), ncols(m), dat(n*m) {}
   
   matrix(int n, int m, T val) : nrows(n), ncols(m), dat(n*m, val) {}
   
   T& operator()(int ii, int jj) { return dat[ii*ncols + jj]; }

   // Return span as "view" of matrix entries, i.e. no copying
   std::span<T> slice_row(int ii) {
      return std::span<T>(dat.begin() + ii*ncols, ncols);
   }
   
   // Non-contiguous slice requires copying
   std::vector<T> slice_col(int jj) {
      std::vector<T> ret(nrows, 0.0);
      for (int ii=0; ii<nrows; ++ii) {
         ret[ii] = dat(ii,jj);
      }
      
      return ret;
   }

   std::array<int,2> size() { return {nrows,ncols}; }
   
   std::vector<T> mmult(const matrix<T>& A, const std::vector<T>& x) {
      assert(x.size() == ncols);
      std::vector<T> ret(nrows, 0.0);
      for (int ii=0; ii<nrows; ++ii) {
         std::span<T> row = A.slice_row(ii);
         ret[ii] = std::inner_product(row.begin(), row.end(), x.begin(), 0);
      }
      
      return ret;
   }
   
   // Alternate matrix multiplication given dataPtr for first entry
   // WARNING: Data size is assumed to be matching dimensions of A
   // so there is no bounds checking
   std::vector<T> mmult(const matrix<T>& A, const std::span<const T> data) {
      std::vector<T> ret(nrows, 0.0);
      for (int ii=0; ii<nrows; ++ii) {
         std::span<T> row = A.slice_row(ii);
         ret[ii] = std::inner_product(row.begin(), row.end(), data.begin(), 0);
      }

      return ret;
   }

   std::vector<T> mmult(const std::vector<T>& x) { return mmult(this, x); }

   std::vector<T> mmult(const std::span<const T> data) { return mmult(this, data); }

   static matrix<T> identity_matrix(int n) {
      T tmp = 0;
      matrix<T> ret(n, n, tmp);
      for (int ii=0; ii<n; ++ii) {
         ret(ii,ii) = 1;
      }
      return ret;
   }
};

#endif
