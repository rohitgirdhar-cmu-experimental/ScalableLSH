#ifndef TABLE_HPP
#define TABLE_HPP

#include "LSHFunc.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp> 
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/unordered_set.hpp>
#include <boost/functional/hash.hpp>

struct vectorint_hash {
  std::size_t operator()(vector<int> const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
};

class Table {
  friend class boost::serialization::access;
  LSHFunc lshFunc;
  unordered_map<vector<int>, unordered_set<long long int>, vectorint_hash> index;
public:
  Table(int k, int dim) : lshFunc(k, dim) {}
  Table() {} // used for serializing
  void insert(const vector<float>& feat, long long int label) {
    vector<int> hash;
    lshFunc.computeHash(feat, hash);

    auto pos = index.find(hash);
    if (pos == index.end()) {
      unordered_set<long long int> lst; 
      lst.insert(label);
      index[hash] = lst;
    } else {
      pos->second.insert(label);
    }
  }
  bool search(const vector<float>& feat, unordered_set<long long int>& output) const {
    vector<int> hash;
    lshFunc.computeHash(feat, hash);
    auto pos = index.find(hash);
    if (pos != index.end()) {
      output = pos->second;
      return true;
    }
    return false;
  }
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & lshFunc;
    ar & index;
  }
};

#endif

