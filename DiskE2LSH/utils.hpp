#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>

template<typename Dtype>
void L2Normalize(vector<Dtype>& vec) {
  Dtype norm = 0;
  for (auto el = vec.begin(); el != vec.end(); el++) {
    norm += (*el) * (*el);
  }
  norm = sqrt(norm);
  for (auto el = vec.begin(); el != vec.end(); el++) {
    *el = (*el) / norm;
  }
}

#endif

