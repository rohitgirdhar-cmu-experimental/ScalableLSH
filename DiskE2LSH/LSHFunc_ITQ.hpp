#ifndef LSHFUNC_ITQ_HPP
#define LSHFUNC_ITQ_HPP

#ifndef EIGEN_CONFIG_H_
#define EIGEN_CONFIG_H_

#include <boost/serialization/array.hpp>
// w.r.t Eigen_3.2.4/Eigen/Core
#define EIGEN_DENSEBASE_PLUGIN "../../../../EigenDenseBaseAddons.hpp"
#include <Eigen/Core>
#endif // EIGEN_CONFIG_H_


#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include <functional>
#include <chrono>
#include "config.hpp"

using namespace std;
using namespace std::chrono;
namespace fs = boost::filesystem;

class LSHFunc_ITQ {
  int k; // number of bits in a function (length of key)
  int dim; // dimension of features
  Eigen::VectorXf mean; // for centering the data
  Eigen::MatrixXf R; // rotation matrix (ITQ)
  Eigen::MatrixXf pc; // PCA embedding (the top-<dim> eigen vectors
  
public:
  LSHFunc_ITQ(int _k): k(_k) {}
  LSHFunc_ITQ() {} // used while serializing

  void train(const vector<vector<float>>& sampleDataAsVec, int nTrainIter = 50) {
    assert(sampleDataAsVec.size() > 0);
    dim = sampleDataAsVec[0].size();
    genLSHfunc(sampleDataAsVec, nTrainIter);
  }

  void computeAndSetCenter(const Eigen::MatrixXf& sampleData) {
    mean = sampleData.colwise().mean();
  }

  void centerData(Eigen::MatrixXf& data) {
    data = data.rowwise() - mean.adjoint();
  }

  void learnPCAEmbedding(const Eigen::MatrixXf& data) {
    high_resolution_clock::time_point start = high_resolution_clock::now();
    cout << "Learning PCA Embedding ... ";
    cout.flush();
    Eigen::MatrixXf cov = data.adjoint() * data;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
    pc = eig.eigenvectors().rightCols(dim);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    cout << "Done in " << duration_cast<minutes>(end - start).count() << "min" << endl;
  }

  void pcaEmbed(Eigen::MatrixXf& data) {
    data *= pc;
  }

  void genLSHfunc(const vector<vector<float>>& sampleDataAsVec, int nIter) {
    // Map to Eigen::Matrix
    Eigen::MatrixXf sampleData(sampleDataAsVec.size(), sampleDataAsVec[0].size());
    for (int i = 0; i < sampleDataAsVec.size(); i++) {
      sampleData.row(i) = Eigen::VectorXf::Map(&sampleDataAsVec[i][0], sampleDataAsVec[i].size());
    }
    // directly translated Guo's code
    learnPCAEmbedding(sampleData);
    pcaEmbed(sampleData);
    computeAndSetCenter(sampleData);
    centerData(sampleData);
    R = Eigen::MatrixXf::Random(dim, dim);
    cout << "Running ITQ Training..." << endl;
    cout.flush();
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(R);
    R = svd.matrixU().leftCols(dim);
    for (int iter = 0; iter < nIter; iter++) {
      cout << "Running iteration " << iter << " ...";
      cout.flush();
      high_resolution_clock::time_point start = high_resolution_clock::now();
      Eigen::MatrixXf Z = sampleData * R;
      Eigen::MatrixXf UX = Eigen::MatrixXf::Zero(Z.rows(), Z.cols()) * (-1);
      for (int i = 0; i < UX.rows(); i++) {
        for (int j = 0; j < UX.cols(); j++) {
          UX(i, j) = Z(i, j) > 0 ? 1 : 0;
        }
      }
      Eigen::MatrixXf C = UX.adjoint() * sampleData;
      Eigen::JacobiSVD<Eigen::MatrixXf> svd2(C);
      R = svd2.matrixU() * svd2.matrixV().adjoint();
      high_resolution_clock::time_point end = high_resolution_clock::now();
      cout << "Done in " << duration_cast<minutes>(end - start).count()
           << "min" << endl;
    }
  }

  void computeHash(const vector<float>& _feat, vector<int>& hash) const {
    return;
    /*
    if (_feat.size() == 0) {
      return;
    }
    hash.clear();
    Eigen::MatrixXf feat = Eigen::VectorXf::Map(&_feat[0], _feat.size());
    #if NORMALIZE_FEATS == 1
      feat = feat / feat.norm(); // normalize the feature
    #endif
    Eigen::MatrixXf res = (feat.transpose() * A - b.replicate(feat.cols(), 1)) / w;
    for (int i = 0; i < res.size(); i++) {
      hash.push_back((int) floor(res(i)));
    }
    */
  }

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & k;
    ar & dim;
    ar & mean;
    ar & pc;
    ar & R;
  }
};

#endif

