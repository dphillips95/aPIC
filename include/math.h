#include <AMReX.H>

auto square(){ return 0; }

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
