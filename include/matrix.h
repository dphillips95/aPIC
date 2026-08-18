/*
Matrix and sparse matrix classes for aPIC.

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

#include <vector>
#include <numeric>
#include <span>
#include <assert.h>

// Check if row and col indices are in the correct ordering for insertion
// i.e. first by row, then by col
// This method does not sort, and is intended only for debugging
bool test_order(const std::vector<int>& row_indices, const std::vector<int>& col_indices);

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

   // matrix(const matrix<T>&) = delete;
   // matrix<T>& operator=(const matrix<T>&) = delete;

   // matrix(matrix<T>&&) = default;
   // matrix<T>& operator=(matrix<T>&&) = default;

   // Set all data entries to val but keep structure
   void setVal(const T val) {
      std::fill(m_dat.begin(), m_dat.end(), val);
   }
   
   T& operator()(const size_t ii, const size_t jj) {
#ifdef AMREX_DEBUG
      assert(ii < m_nrows);
      assert(jj < m_ncols);
#endif
      return m_dat[ii*m_ncols + jj];
   }
   
   const T& operator()(const size_t ii, const size_t jj) const {
#ifdef AMREX_DEBUG
      assert(ii < m_nrows);
      assert(jj < m_ncols);
#endif
      return m_dat[ii*m_ncols + jj];
   }

   // Returns span as "view" of matrix entries, i.e. no copying
   std::span<const T> slice_row_const(size_t ii) const {
#ifdef AMREX_DEBUG
      assert(ii < m_nrows);
#endif
      return std::span<const T>(m_dat.data() + ii*m_ncols, m_ncols);
   }

   // Returns span as "view" of matrix entries, i.e. no copying
   std::span<T> slice_row(size_t ii) {
#ifdef AMREX_DEBUG
      assert(ii < m_nrows);
#endif
      return std::span<T>(m_dat.data() + ii*m_ncols, m_ncols);
   }
   
   // Non-contiguous slice requires copying, so returns vector
   std::vector<T> slice_col(size_t jj) {
#ifdef AMREX_DEBUG
      assert(jj < m_ncols);
#endif
      std::vector<T> ret(m_nrows, 0.0);
      for (size_t ii=0; ii<m_nrows; ++ii) {
         ret[ii] = m_dat(ii,jj);
      }
      
      return ret;
   }

   std::array<size_t,2> size() const { return {m_nrows,m_ncols}; }

   static void mmult(std::span<T> ret, const matrix<T>& A, const std::span<const T> x) {
#ifdef AMREX_DEBUG
      assert(A.m_ncols == x.size());
      assert(A.m_nrows == ret.size());
#endif
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

   static void mmult_add(std::span<T> ret, const matrix<T>& A, const std::span<const T> x, T a) {
#ifdef AMREX_DEBUG
      assert(A.m_ncols == x.size());
      assert(A.m_nrows == ret.size());
#endif
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         for (size_t jj=0; jj<A.m_ncols; ++jj) {
            ret[ii] += a*A.m_dat[ii*A.m_ncols + jj]*x[jj];
         }
      }
   }
   
   void mmult_add(std::span<T> ret, const std::span<const T> x, T a) const { mmult_add(ret, *this, x, a); }
   
   static matrix<T> identity(size_t n) {
      T tmp = 0;
      matrix<T> ret(n, n, tmp);
      for (size_t ii=0; ii<n; ++ii) {
         ret(ii,ii) = 1;
      }
      return ret;
   }

   // Scale entire matrix by given factor
   void scale(const T fact) {
      for (size_t ii=0; ii<m_nrows*m_ncols; ++ii) {
         m_dat[ii] *= fact;
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
   
   sp_matrix(size_t n, size_t m) : m_nvals(0), m_nrows(n), m_ncols(m), m_dat(), m_col_indices(), m_row_indices() {
      m_row_indices.reserve(n+1);
      m_row_indices.push_back(0);
   }
   
   sp_matrix(size_t size_est, size_t n, size_t m) : m_nvals(0), m_nrows(n), m_ncols(m), m_dat(), m_col_indices(), m_row_indices() {
      m_dat.reserve(size_est);
      m_row_indices.reserve(n+1);
      m_row_indices.push_back(0);
      m_col_indices.reserve(size_est);
   }

   // Set all data entries to val but keep structure
   void setVal(const T val) {
      std::fill(m_dat.begin(), m_dat.end(), val);
   }
   
   // Add single entry to matrix; entry must have same row as previous and larger column index (unless it is the first entry of the row)
   void add_entry(T dat, size_t col_index) {
#ifdef AMREX_DEBUG
      assert(col_index < m_ncols);
#endif
      ++m_nvals;
      m_dat.push_back(dat);
      m_col_indices.push_back(col_index);
   }
   
   // End row by adding current last entry data index to m_row_indices
   void end_row() { m_row_indices.push_back(m_nvals); }

   // Main construction method; provide data in COO format - list of data, list of row indices, list of col indices
   // Input is assumed to be in correct order - first by row, then by col
   // If order is wrong this WILL fail
   void add_chunk(const std::vector<T>& dat, const std::vector<int>& row_indices, const std::vector<int>& col_indices) {
#ifdef AMREX_DEBUG
      assert(dat.size() == row_indices.size());
      assert(dat.size() == col_indices.size());
      assert(test_order(row_indices, col_indices));
#endif
      this->add_rows(row_indices[0] - m_row_indices.size() + 1);
      this->add_entry(dat[0], col_indices[0]);
      for (size_t ii=1; ii<dat.size(); ++ii) {
         this->add_rows(row_indices[ii] - row_indices[ii-1]);
         this->add_entry(dat[ii], col_indices[ii]);
      }
      this->end_row();
   }

   // Add chunk but set all data to given value
   void add_chunk(const T& val, const std::vector<int>& row_indices, const std::vector<int>& col_indices) {
#ifdef AMREX_DEBUG
      assert(row_indices.size() == col_indices.size());
      assert(test_order(row_indices, col_indices));
#endif
      this->add_rows(row_indices[0] - m_row_indices.size() + 1);
      this->add_entry(val, col_indices[0]);
      for (size_t ii=1; ii<row_indices.size(); ++ii) {
         this->add_rows(row_indices[ii] - row_indices[ii-1]);
         this->add_entry(val, col_indices[ii]);
      }
      this->end_row();
   }
   
   void add_rows(size_t num_rows) {
#ifdef AMREX_DEBUG
      assert(m_row_indices.size() + num_rows <= m_nrows + 1);
#endif
      for (size_t ii=0; ii<num_rows; ++ii) {
         this->end_row();
      }
   }
   
   void add_rows(size_t start, size_t end) {
#ifdef AMREX_DEBUG
      assert(m_row_indices.size() + end <= m_nrows + 1 + start);
#endif
      for (size_t ii=start; ii<end; ++ii) {
         this->end_row();
      }
   }

   // Fill in extra rows of m_row_indices at end if blank
   void finalise() {
#ifdef AMREX_DEBUG
      assert(m_row_indices.size() <= m_nrows + 1);
#endif
      this->add_rows(m_row_indices.size(), m_nrows + 1);
   }
   
   std::array<size_t,2> size() const { return {m_nrows,m_ncols}; }

   static void mmult(std::span<T> ret, const sp_matrix<T>& A, const std::span<const T> x) {
#ifdef AMREX_DEBUG
      assert(A.m_ncols == x.size());
      assert(A.m_nrows == ret.size());
#endif
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

   static void mmult_add(std::span<T> ret, const sp_matrix<T>& A, const std::span<const T> x, T a) {
#ifdef AMREX_DEBUG
      assert(A.m_ncols == x.size());
      assert(A.m_nrows == ret.size());
#endif
      size_t ind = 0;
      for (size_t ii=0; ii<A.m_nrows; ++ii) {
         for (size_t jj=0; jj < A.m_row_indices[ii+1] - A.m_row_indices[ii]; ++jj) {
            ret[ii] += a*A.m_dat[ind]*x[A.m_col_indices[ind]];
            ++ind;
         }
      }
   }
   
   void mmult_add(std::span<T> ret, const std::span<const T> x, T a) const { mmult_add(ret, *this, x, a); }

   // Scale entire matrix by given factor
   void scale(const T fact) {
      for (size_t ii=0; ii<m_dat.size(); ++ii) {
         m_dat[ii] *= fact;
      }
   }
};

#endif
