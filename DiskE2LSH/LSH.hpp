#ifndef LSH_HPP
#define LSH_HPP

#include "Table.hpp"
#include <boost/serialization/serialization.hpp>

class LSH {
  friend class boost::serialization::access;
  vector<Table> tables;
public:
  LSH(int k, int L, int dim) {
    for (int i = 0; i < L; i++) {
      tables.push_back(Table(k, dim));
    }
  }
  void insert(const vector<float>& feat, long long int label) {
    #pragma omp parallel for
    for (int i = 0; i < tables.size(); i++) {
      tables[i].insert(feat, label);
    }
  }
  void search(const vector<float>& feat, unordered_set<long long int>& output) const {
    output.clear();
    #pragma omp parallel for
    for (int i = 0; i < tables.size(); i++) {
      unordered_set<long long int> part;
      tables[i].search(feat, part);
      #pragma omp critical
      output.insert(part.begin(), part.end());
    }
  }
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & tables; 
  }
};

#endif
