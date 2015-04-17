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
  if (! ifs.is_open()) {
    cerr << "Unable to open " << fpath << endl;
    return;
  }
  while (ifs >> el) {
    output.push_back(el);
  } 
  ifs.close();
}

void getAllSearchspace(const vector<int>& featcounts,
    unordered_set<long long int>& searchspace) {
  searchspace.clear();
  for (long long int i = 1; i <= featcounts.size(); i++) {
    for (long long int j = 1; j <= featcounts[i]; j++) {
      searchspace.insert(i * MAXFEATPERIMG + j);
    }
  }
}

std::vector<std::string> &split(const std::string &s, 
    char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, elems);
  return elems;
}

void readResults(const fs::path& fpath,
    vector<vector<pair<float, long long>>>& allres) {
  ifstream fin(fpath.string());
  if (!fin.is_open()) {
    cerr << "Unable to read file " << fpath << endl;
    return;
  }
  string line;
  int lno = -1;
  while (getline(fin, line)) {
    lno++;
    if (line.length() <= 0) continue;
    vector<string> elems = split(line, ' ');
    for (int i = 0; i < elems.size(); i++) {
      if (elems[i].length() <= 0) continue;
      vector<string> p = split(elems[i], ':');
      allres[lno].push_back(make_pair(stof(p[0]), stoll(p[1])));
    }
  }
  fin.close();
}

/**
 * Assumes both input featid and imgid are 1 indexed
 */
int computeFeatId(int imgid, int featid) {
  return imgid * MAXFEATPERIMG + featid;
}

#endif

