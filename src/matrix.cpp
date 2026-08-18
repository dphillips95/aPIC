/*
Matrix and sparse matrix methods for aPIC.

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

#include <matrix.h>

#include <vector>

// Check if row and col indices are in the correct ordering for insertion
// i.e. first by row, then by col
// This method does no sorting, and is intended only for debugging
bool test_order(const std::vector<int>& row_indices, const std::vector<int>& col_indices) {
   for (size_t ii=1; ii<row_indices.size(); ++ii) {
      if (row_indices[ii] < row_indices[ii-1]) { return false; }
      else if ((row_indices[ii] == row_indices[ii-1]) &&
               (col_indices[ii] < col_indices[ii-1])) { return false; }
   }
   return true;
}
