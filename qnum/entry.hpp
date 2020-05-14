#pragma once

#include "common.hpp"
#include "data.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include <tuple>

using namespace std;

int g_batch_size = 1;

template<typename T>
std::tuple<const dataset_t<T>*, const dataset_t<T>*> load_data(const string& dataset) {
  cout << "[DEBUG] loading data..." << endl;
  const dataset_t<T> *ptrain, *ptest;
  if (dataset == "cifar10") {
    ptrain = load_cifar10_train<T>();
    ptest = load_cifar10_test<T>();
  } else if (dataset == "mnist") {
    ptrain = load_mnist_train<T>();
    ptest = load_mnist_test<T>();
  } else {
    printf("error: unrecognized dataset %s\n", dataset.c_str());
    exit(-1);
  }
  return std::make_tuple(ptrain, ptest);
}

template<typename T> nn_t<T>* init_net(const string& arch, int nhidden, const dataset_t<T>* ptrain) {
  cout << "[DEBUG] initializing network..." << endl;
  if (arch == "mlp") {
    return new mlp_t<T>(ptrain->height * ptrain->width * ptrain->nchannel, nhidden, ptrain->nclass);
  } else if (arch == "cnn") {
    return new cnn_t<T>(ptrain->nchannel, ptrain->height, ptrain->width, ptrain->nclass);
  } else {
    printf("error: unrecognized network arch %s\n", arch.c_str());
    exit(-1);
  }
}

// forward declaration
template<typename T> void entry(int E, const string& arch, const string& dataset, double lr, int nhidden, const string& type, const char* checkpoint);

template<typename T, int D, typename ... Args> void entry_wrap_q(int E, Args... args)
{
  switch (E) {
#if !defined(PARTIAL_BUILD)
    case 1:
      entry<qspace_number_t<T, 1, D>>(E, args...);
      break;
    case 2:
      entry<qspace_number_t<T, 2, D>>(E, args...);
      break;
#endif
    case 3:
      entry<qspace_number_t<T, 3, D>>(E, args...);
      break;
#if !defined(PARTIAL_BUILD)
    case 4:
      entry<qspace_number_t<T, 4, D>>(E, args...);
      break;
    case 5:
      entry<qspace_number_t<T, 5, D>>(E, args...);
      break;
    case 6:
      entry<qspace_number_t<T, 6, D>>(E, args...);
      break;
    case 7:
      entry<qspace_number_t<T, 7, D>>(E, args...);
      break;
    case 8:
      entry<qspace_number_t<T, 8, D>>(E, args...);
      break;
#endif
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

template<int B, typename ... Args> void entry_wrap_flex16(int E, Args... args)
{
  switch (E) {
#if !defined(PARTIAL_BUILD)
    case 1:
      entry<flexfloat<1, B - 2>>(E, args...);
      break;
    case 2:
      entry<flexfloat<2, B - 3>>(E, args...);
      break;
    case 3:
      entry<flexfloat<3, B - 4>>(E, args...);
      break;
    case 4:
      entry<flexfloat<4, B - 5>>(E, args...);
      break;
    case 5:
      entry<flexfloat<5, B - 6>>(E, args...);
      break;
    case 6:
      entry<flexfloat<6, B - 7>>(E, args...);
      break;
    case 7:
      entry<flexfloat<7, B - 8>>(E, args...);
      break;
#endif
    case 8:
      entry<flexfloat<8, B - 9>>(E, args...);
      break;
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

int launch(int argc, char* argv[]) {

  if (argc < 8) {
    std::cout << "usage: " << argv[0] << " num_type arch[mlp|cnn] dataset[mnist|cifar10] batchsize ext_bits lr nhidden [checkpoint_file]" << std::endl;
    return -1;
  }

  std::string type = argv[1];
  std::string arch = argv[2];
  std::string dataset = argv[3];
  g_batch_size = atoi(argv[4]);
  int E = atoi(argv[5]);
  double lr = atof(argv[6]);
  int nhidden = atoi(argv[7]);
  char* chkpoint = nullptr;
  if (argc == 9) {
    chkpoint = argv[8];
  }

#if defined(PARTIAL_BUILD)
  if(type == "q16") entry_wrap_q<int16_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f32") entry<float>(0, arch, dataset, lr, nhidden, type, chkpoint);
#else

  if(type == "q8") entry_wrap_q<int8_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  //else if(type == "q11") entry_wrap_q<int16_t, 5>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if(type == "q12") entry_wrap_q<int16_t, 4>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if(type == "q13") entry_wrap_q<int16_t, 3>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if(type == "q14") entry_wrap_q<int16_t, 2>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if(type == "q15") entry_wrap_q<int16_t, 1>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q16") entry_wrap_q<int16_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  else if (type == "q32") entry_wrap_q<int32_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  else if (type == "f32") entry<float>(0, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f64") entry<double>(0, arch, dataset, lr, nhidden, type, chkpoint);

  //else if (type == "f11") entry_wrap_flex16<11>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f12") entry_wrap_flex16<12>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f13") entry_wrap_flex16<13>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f14") entry_wrap_flex16<14>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f15") entry_wrap_flex16<15>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f16") entry_wrap_flex16<16>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f17") entry_wrap_flex16<17>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f18") entry_wrap_flex16<18>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f19") entry_wrap_flex16<19>(E, arch, dataset, lr, nhidden, type, chkpoint);
  //else if (type == "f20") entry_wrap_flex16<20>(E, arch, dataset, lr, nhidden, type, chkpoint);
#endif

  else { cout << "unknown data type " << type << "." << endl; }

  return 0;
}
