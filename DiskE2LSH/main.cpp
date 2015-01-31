#include <iostream>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "storage/DiskVector.hpp"
#include "LSH.hpp"
#include "Resorter.hpp"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

#define RESDIR "tempdata/matches/"

int main(int argc, char* argv[]) {
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Show this help")
    ("datapath,d", po::value<string>(),
     "Path to leveldb where the normalized data is stored")
    ("sized,n", po::value<int>(),
     "Number of features in datapath to hash")
    ("querypath,q", po::value<string>(),
     "Path to leveldb where the query data is stored")
    ("sizeq,m", po::value<int>(),
     "Number of features in querypath to search for")
    ("save,s", po::value<string>(),
     "Path to save the hash table")
    ("load,l", po::value<string>(),
     "Path to load the hash table from")
    ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("help")) {
    cerr << desc << endl;
    return -1;
  }
  try {
    po::notify(vm);
  } catch(po::error& e) {
    cerr << e.what() << endl;
    return -1;
  }

  LSH *l;
  DiskVector<vector<float>> tree(vm["datapath"].as<string>());
  if (vm.count("load")) {
    cout << "Loading model from " << vm["load"].as<string>() << "...";
    ifstream ifs(vm["load"].as<string>(), ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    l = new LSH(0,0,0); // need to create this dummy obj, don't know how else...
    ia >> *l;
    cout << "done." << endl;
  } else if (vm.count("datapath")) {
    l = new LSH(200, 15, 9216);
    vector<float> feat;
    for (int i = 0; i < vm["sized"].as<int>(); i++) {
      tree.Get(i, feat);
      l->insert(feat, i);
      if (i % 1000 == 0) {
        cout << "Done for " << i << endl;
      }
    }
  }

  if (vm.count("save")) {
    cout << "Saving model to " << vm["save"].as<string>() << "...";
    ofstream ofs(vm["save"].as<string>(), ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << *l;
    cout << "done." << endl;
  }

  if (vm.count("querypath")) {
    unordered_set<int> temp;
    vector<float> feat;
    DiskVector<vector<float>> q(vm["querypath"].as<string>());
    for (int i = 0; i < vm["sizeq"].as<int>(); i++) {
      std::chrono::high_resolution_clock::time_point t1 = 
        std::chrono::high_resolution_clock::now();
      q.Get(i, feat);
      l->search(feat, temp);
      vector<pair<float, int>> res = Resorter::resort(temp, tree, feat);
      ofstream fout(string(RESDIR) + "/" + to_string(i + 1) + ".txt");
      for (auto it = res.begin(); it != res.end(); it++) {
        fout << it->second + 1 << endl; 
      }
      fout.close();
      std::chrono::high_resolution_clock::time_point t2 = 
        std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (t2 - t1).count();
      cout << "Search done for " << i << " in " << duration << " ms" << endl;
    }
  }

  return 0;
}

