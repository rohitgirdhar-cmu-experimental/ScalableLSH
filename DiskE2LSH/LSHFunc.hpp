#ifndef LSHFUNC_HPP
#define LSHFUNC_HPP

#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include <functional>

using namespace std;
namespace fs = boost::filesystem;

class LSHFunc {
  float w;
  int k; // number of bits in a function (length of key)
  int dim; // dimension of features
  Eigen::MatrixXf A;
  Eigen::MatrixXf b;
  
public:
  LSHFunc(int _k, int _dim) {
    k = _k;
    dim = _dim;
    genLSHfunc();
  }

  void genLSHfunc() {
    w = 24; // default value
    A = Eigen::MatrixXf::Random(dim, k); // TODO: Use normal distribution to sample (as in GS code)
    typedef boost::mt19937 RNGType;
    RNGType rng;
    boost::uniform_real<> generator(0, w);
    boost::variate_generator<RNGType, boost::uniform_real<>> dice(rng, generator);

    b = Eigen::MatrixXf::Random(1, k);
    for (int i = 0; i < k; i++) {
      b(0, i) = dice();
    }
  }

  void computeHash(const vector<float>& _feat, vector<int>& hash) {
    hash.clear();
    Eigen::MatrixXf feat = Eigen::VectorXf::Map(&_feat[0], _feat.size());
    Eigen::MatrixXf res = (feat.transpose() * A - b.replicate(feat.cols(), 1)) / w;
    for (int i = 0; i < res.size(); i++) {
      hash.push_back((int) floor(res(i)));
    }
  }

};

#endif

