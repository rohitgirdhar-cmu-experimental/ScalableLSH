#ifndef LSH_HPP
#define LSH_HPP

#include "Table.hpp"

class LSH {
  vector<Table> tables;
public:
  LSH(int k, int L, int dim) {
    for (int i = 0; i < L; i++) {
      tables.push_back(Table(k, dim));
    }
  }
  void insert(const vector<float>& feat, int label) {
    #pragma omp parallel for
    for (int i = 0; i < tables.size(); i++) {
      tables[i].insert(feat, label);
    }
  }
  void search(const vector<float>& feat, unordered_set<int>& output) {
    output.clear();
    for (auto it = tables.begin(); it != tables.end(); it++) {
      unordered_set<int> part;
      it->search(feat, part);
      output.insert(part.begin(), part.end());
    }
  }
};

#endif
