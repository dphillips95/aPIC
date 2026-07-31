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

#include <AMReX_Print.H>

// Non-sparse Matrix class; dimensions are defined at run time but are fixed
template <typename T>
class matrix {
private:
   size_t m_nrows,m_ncols;
   std::vector<T> m_dat;
public:
   matrix() : m_nrows(0), m_ncols(0), m_dat() {}
   
   matrix(size_t n, size_t m) : m_nrows(n), m_ncols(m), m_dat(n*m) {}
   
   matrix(size_t n, size_t m, T val) : m_nrows(n), m_ncols(m), m_dat(n*m, val) {}
   
   T& operator()(size_t ii, size_t jj) { return m_dat[ii*m_ncols + jj]; }
   
   const T& operator()(const size_t ii, const size_t jj) const { return m_dat[ii*m_ncols + jj]; }

   // Returns span as "view" of matrix entries, i.e. no copying
   std::span<const T> slice_row_const(size_t ii) const {
      return std::span<const T>(m_dat.data() + ii*m_ncols, m_ncols);
   }

   // Returns span as "view" of matrix entries, i.e. no copying
   std::span<T> slice_row(size_t ii) {
      return std::span<T>(m_dat.data() + ii*m_ncols, m_ncols);
   }
   
   // Non-contiguous slice requires copying, so returns vector
   std::vector<T> slice_col(size_t jj) {
      std::vector<T> ret(m_nrows, 0.0);
      for (size_t ii=0; ii<m_nrows; ++ii) {
         ret[ii] = m_dat(ii,jj);
      }
      
      return ret;
   }

   std::array<size_t,2> size() const { return {m_nrows,m_ncols}; }

   static void mmult(std::span<T> ret, const matrix<T>& A, const std::span<const T> x) {
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         ret[ii] = 0;
         for (size_t jj=0; jj<A.m_ncols; ++jj) {
            ret[ii] += A.m_dat[ii*A.m_ncols + jj]*x[jj];
         }
      }
   }
   
   static std::vector<T> mmult(const matrix<T>& A, const std::span<const T> x) {
      std::vector<T> ret(A.m_nrows, 0.0);
      matrix<T>::mmult(ret, A, x);
      return ret;
   }

   static std::vector<T> mmult(const matrix<T>& A, const std::vector<T>& x) {
      std::vector<T> ret(A.m_nrows, 0.0);
      matrix<T>::mmult(ret, A, x);
      return ret;
   }

   static void mmult(std::span<T> ret, const matrix<T>& A, const std::vector<T>& x) {
      matrix<T>::mmult(ret, A, std::span<T>(x.begin(), x.end()));
   }
   
   std::vector<T> mmult(const std::vector<T>& x) const { return mmult(*this, x); }

   std::vector<T> mmult(const std::span<const T> x) const { return mmult(*this, x); }

   void mmult(std::span<T> ret, const std::span<const T> x) const { mmult(ret, *this, x); }

   static void mmult_add(std::span<T> ret, const matrix<T>& A, const std::span<const T> x, amrex::Real a) {
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         for (size_t jj=0; jj<A.m_ncols; ++jj) {
            ret[ii] += a*A.m_dat[ii*A.m_ncols + jj]*x[jj];
         }
      }
   }
   
   void mmult_add(std::span<T> ret, const std::span<const T> x, amrex::Real a) const { mmult_add(ret, *this, x, a); }
   
   static matrix<T> identity(size_t n) {
      T tmp = 0;
      matrix<T> ret(n, n, tmp);
      for (size_t ii=0; ii<n; ++ii) {
         ret(ii,ii) = 1;
      }
      return ret;
   }
   
   void print_rowsums(auto name) const {
      for (size_t ii=0; ii<m_nrows; ++ii) {
         const std::span<const amrex::Real> row = this->slice_row_const(ii);
         amrex::Print() << name << "[" << ii << "] sum: " << std::reduce(row.begin(), row.end()) << std::endl;
      }
   }
};

// Sparse Matrix class; dimensions are defined at run time but are fixed
// Sparse format is CSR (compressed sparse row) for efficient matrix-vector multiplication
template <typename T>
class sp_matrix {
private:
   size_t m_nvals, m_nrows, m_ncols;
   std::vector<T> m_dat;
   std::vector<size_t> m_col_indices, m_row_indices;
public:
   sp_matrix() : m_nvals(0), m_nrows(0), m_ncols(0), m_dat(), m_col_indices(), m_row_indices() {}
   
   sp_matrix(size_t n, size_t m) : m_nvals(0), m_nrows(n), m_ncols(m), m_dat(), m_col_indices(), m_row_indices(n+1) {
      m_row_indices[0] = 0;
   }
   
   sp_matrix(size_t size_est, size_t n, size_t m) : m_nvals(0), m_nrows(n), m_ncols(m), m_dat(), m_col_indices(), m_row_indices(n+1) {
      m_row_indices[0] = 0;
      m_dat.reserve(size_est);
      m_col_indices.reserve(size_est);
   }
   
   // Add single entry to matrix; entry must have same row as previous and larger column index (unless it is the first entry of the row)
   void add_entry(T dat, size_t col_index) {
      ++m_nvals;
      m_dat.push_back(dat);
      m_col_indices.push_back(col_index);
   }
   
   // End row by adding current last entry data index to m_row_indices
   void end_row() { m_row_indices.push_back(m_nvals); }

   void add_chunk(const std::vector<T>& dat, const std::vector<int>& row_indices, const std::vector<int>& col_indices) {
      for (size_t ii=0; ii<dat.size(); ++ii) {
         this->add_empty_rows(row_indices[ii] - row_indices[ii-1]);
         this->add_entry(dat[ii], col_indices[ii]);
      }
      this->end_row();
   }

   void add_empty_rows(int empty_rows) {
      for (size_t ii=0; ii<empty_rows; ++ii) {
         this->end_row();
      }
   }
   
   std::array<size_t,2> size() const { return {m_nrows,m_ncols}; }

   static void mmult(std::span<T> ret, const sp_matrix<T>& A, const std::span<const T> x) {
      size_t ind = 0;
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         ret[ii] = 0;
         for (size_t jj=0; jj < A.m_row_indices[ii+1] - A.m_row_indices[ii]; ++jj) {
            ret[ii] += A.m_dat[ind]*x[A.m_col_indices[ind]];
            ++ind;
         }
      }
   }
   
   static std::vector<T> mmult(const sp_matrix<T>& A, const std::vector<T>& x) {
      std::vector<T> ret(A.m_nrows, 0.0);
      sp_matrix<T>::mmult(ret, A, x);
      return ret;
   }
   
   static std::vector<T> mmult(const sp_matrix<T>& A, const std::span<const T> x) {
      std::vector<T> ret(A.m_nrows, 0.0);
      sp_matrix<T>::mmult(ret, A, x);
      return ret;
   }
   
   static void mmult(std::span<T> ret, const sp_matrix<T>& A, const std::vector<T>& x) {
      sp_matrix<T>::mmutl(ret, A, std::span<const T>(x.begin(), x.end()));
   }
   
   std::vector<T> mmult(const std::vector<T>& x) const { return mmult(*this, x); }

   std::vector<T> mmult(const std::span<const T> x) const { return mmult(*this, x); }

   void mmult(std::span<T> ret, const std::span<const T> x) const { mmult(ret, *this, x); }

   static void mmult_add(std::span<T> ret, const sp_matrix<T>& A, const std::span<const T> x, amrex::Real a) {
      size_t ind = 0;
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         for (size_t jj=0; jj < A.m_row_indices[ii+1] - A.m_row_indices[ii]; ++jj) {
            ret[ii] += a*A.m_dat[ind]*x[A.m_col_indices[ind]];
            ++ind;
         }
      }
   }
   
   void mmult_add(std::span<T> ret, const std::span<const T> x, amrex::Real a) const { mmult_add(ret, *this, x, a); }
   
   void print_rowsums(auto name) const {
      size_t ind = 0;
      for (size_t ii=0; ii<m_nrows; ++ii) {
         T rowsum = 0;
         for (size_t jj=0; jj < m_row_indices[ii+1] - m_row_indices[ii]; ++jj) {
            rowsum += m_dat[ind];
            ++ind;
         }
         amrex::Print() << name << "[" << ii << "] sum: " << rowsum << std::endl;
      }
   }
};

#endif
