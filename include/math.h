/*
Mathematical functions for aPIC.

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

#ifndef MATH_H_
#define MATH_H_

#include <cmath>
#include <numeric>


auto square(const auto& x) {
   if constexpr(requires{ x[0]; }) {
      std::remove_cvref_t<decltype(x[0])> ret{0};
      for (const auto& y: x) ret += square(y);
      return ret;
   } else {
      return x*x;
   }
}

auto square(const auto& x, const auto&... rest) {
   return square(x) + square(rest...);
}
template <class T> T square(std::initializer_list<T> x) {
	T ret{0};
	for (const auto& y: x) ret += y;
	return ret;
}

auto norm(const auto&... x) {
	return std::sqrt(square(x...));
}
template <class T> T norm(std::initializer_list<T> x) {
	return std::sqrt(square(x));
}

#endif
