#ifndef DISKE2LSH_UTILS_HPP
#define DISKE2LSH_UTILS_HPP

#include <cmath>
#include <boost/filesystem.hpp>
#include <fstream>
#include "config.hpp"

namespace fs = boost::filesystem;

template<typename Dtype>
void L2Normalize(vector<Dtype>& vec) {
  Dtype norm = 0;
  for (auto el = vec.begin(); el != vec.end(); el++) {
    norm += (*el) * (*el);
  }
  norm = sqrt(norm);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = vec[i] / norm;
  }
}

template<typename Dtype>
void readList(const fs::path& fpath, vector<Dtype>& output) {
  output.clear();
  Dtype el;
  ifstream ifs(fpath.string());
  while (ifs >> el) {
    output.push_back(el);
  } 
  ifs.close();
}

void getAllSearchspace(const vector<int>& featcounts,
    unordered_set<long long int>& searchspace) {
  searchspace.clear();
  for (long long int i = 0; i < featcounts.size(); i++) {
    for (long long int j = 0; j < featcounts[i]; j++) {
      searchspace.insert(i * MAXFEATPERIMG + j);
    }
  }
}

#endif

