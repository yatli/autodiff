#include "common.hpp"
#include "data.hpp"
#include "mlp.hpp"
#include <thread>
#include <tuple>

using namespace std;

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
    // TODO cnn
    return nullptr;
  } else {
    printf("error: unrecognized network arch %s\n", arch.c_str());
    exit(-1);
  }
}

template<typename T> void train(int E, const string& arch, const string& dataset, double lr, int nhidden, const string& type, const char* checkpoint) {
  auto dataset_tup = load_data<T>(dataset);
  const dataset_t<T>* ptrain = std::get<0>(dataset_tup);
  const dataset_t<T>* ptest = std::get<1>(dataset_tup);
  nn_t<T>* pnet = init_net<T>(arch, nhidden, ptrain);

  if(checkpoint != nullptr) {
    cout << "[DEBUG] Loading checkpoint from " << checkpoint << endl;
    pnet->load(checkpoint);
  }

  int nupdates = 0;

  for (int epoch = 0; epoch < 20; ++epoch) {
    auto samples = ptrain->shuffle();
#if NDEBUG
    auto batch_size = 8;
#else
    auto batch_size = 1;
#endif
    auto run = [&](const VectorXtvar<T> &img, 
                  const VectorXtvar<T> &label, 
                  double& loss_store,
                  int& correct_store,
                  bool backward) {
      auto label_predict = pnet->forward(img);
      auto loss = loss_crossent(label, label_predict);
      loss_store = static_cast<double>(loss.expr->val);
      correct_store = (argmax(label) == argmax(label_predict));
      if (backward) {
        pnet->backward(loss);
      }
    };

    std::vector<double> losses(batch_size);
    std::vector<int> corrects(batch_size);
    int total_correct = 0;
    double total_loss = 0.0;
    auto smpidx = ptrain->shuffle();

    for (auto i = 0; i < ptrain->size; i += batch_size) {

      if (i % 10000 == 0) {
        char buf[256];
        sprintf(buf, "%s-%s-%s-e%d-h%d-lr%f-epoch-%d-step-%d.dmp", type.data(), arch.data(), dataset.data(), E, nhidden, lr, epoch, i);
        pnet->save(buf);
      }

      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size; ++j) {
        threads.emplace_back([&](auto idx){
          run(ptrain->imgs[smpidx[idx]], ptrain->labels[smpidx[idx]], losses[idx - i], corrects[idx - i], true);
        }, i+j);
      }
      for(auto &t: threads) {
        t.join();
      }

      // update & print stats
      auto batch_loss = 0.0;
      for(auto l: losses) { batch_loss += l; }
      batch_loss /= batch_size;
      if (!std::isnormal(batch_loss) || batch_loss > 10.0) {
        cout << "[DEBUG] abnormal loss detected. dump and ignore now." << endl;
        cout << "[DEBUG] current batch is: ";
        for(auto j = 0; j < batch_size && i + j < ptrain->size; ++j) {
          cout << smpidx[i+j] << " ( " << losses[j] << " ) ";
        }
        cout << endl;
        char buf[256];
        sprintf(buf, "%s-%s-%s-e%d-h%d-lr%f-epoch-%d-step-%d.dmp", type.data(), arch.data(), dataset.data(), E, nhidden, lr, epoch, i);
        // pnet->save(buf);
        continue;
      }

      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }

      auto current_acc = total_correct / ((double)i + batch_size);
      auto current_loss = total_loss / (i+batch_size);

      nupdates += batch_size;

      cout 
        << "[TRAIN] epoch= "  << setw(3)  << epoch
        << " step= "          << setw(5)  << i
        << " batchloss= "     << setw(12) << batch_loss
        << " avgloss= "       << setw(12) << current_loss
        << " acc= "           << setw(12) << current_acc
        << " nupdates= "      << setw(10) << nupdates 
        << endl;

      pnet->learn(T(lr));

      if ((i/batch_size) % 10 == 0) {
        pnet->check_histogram();
        // TODO check saturation, but on all nodes, not just weights
      }

    }

    total_correct = 0;
    total_loss = 0.0;
    cout << "[TEST] epoch " << setw(4) << epoch;
    for (auto i = 0; i < ptest->size; i += batch_size) {
      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size; ++j) {
        threads.emplace_back([&](auto idx){
          run(ptest->imgs[idx], ptest->labels[idx], losses[idx - i], corrects[idx - i], false);
        }, i+j);
      }
      for(auto &t: threads) {
        t.join();
      }
      // update & print stats
      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }
    }
    cout 
      << " avgloss= " << setw(12) << total_loss / (double)ptest->size
      << " acc= "     << setw(12) << total_correct / (double)ptest->size
      << endl;
  }
}

template<typename T, int D, typename ... Args> void train_wrap_q(int E, Args... args)
{
  switch (E) {
#if NDEBUG
    case 1:
      train<qspace_number_t<T, 1, D>>(E, args...);
      break;
    case 2:
      train<qspace_number_t<T, 2, D>>(E, args...);
      break;
#endif
    case 3:
      train<qspace_number_t<T, 3, D>>(E, args...);
      break;
#if NDEBUG
    case 4:
      train<qspace_number_t<T, 4, D>>(E, args...);
      break;
    case 5:
      train<qspace_number_t<T, 5, D>>(E, args...);
      break;
    case 6:
      train<qspace_number_t<T, 6, D>>(E, args...);
      break;
    case 7:
      train<qspace_number_t<T, 7, D>>(E, args...);
      break;
    case 8:
      train<qspace_number_t<T, 8, D>>(E, args...);
      break;
#endif
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

template<int B, typename ... Args> void train_wrap_flex16(int E, Args... args)
{
  switch (E) {
#if NDEBUG
    case 1:
      train<flexfloat<1, B - 2>>(E, args...);
      break;
    case 2:
      train<flexfloat<2, B - 3>>(E, args...);
      break;
    case 3:
      train<flexfloat<3, B - 4>>(E, args...);
      break;
    case 4:
      train<flexfloat<4, B - 5>>(E, args...);
      break;
    case 5:
      train<flexfloat<5, B - 6>>(E, args...);
      break;
    case 6:
      train<flexfloat<6, B - 7>>(E, args...);
      break;
    case 7:
      train<flexfloat<7, B - 8>>(E, args...);
      break;
#endif
    case 8:
      train<flexfloat<8, B - 9>>(E, args...);
      break;
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cout << "usage: train num_type arch[mlp|cnn] dataset[mnist|cifar10] ext_bits lr nhidden [checkpoint_file]" << std::endl;
  }

  std::string type = argv[1];
  std::string arch = argv[2];
  std::string dataset = argv[3];
  int E = atoi(argv[4]);
  double lr = atof(argv[5]);
  int nhidden = atoi(argv[6]);
  char* chkpoint = nullptr;
  if (argc == 8) {
    chkpoint = argv[7];
  }

#if! NDEBUG
  if(type == "q16") train_wrap_q<int16_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f32") train<float>(0, arch, dataset, lr, nhidden, type, chkpoint);
#else

  if(type == "q8") train_wrap_q<int8_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  else if(type == "q11") train_wrap_q<int16_t, 5>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q12") train_wrap_q<int16_t, 4>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q13") train_wrap_q<int16_t, 3>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q14") train_wrap_q<int16_t, 2>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q15") train_wrap_q<int16_t, 1>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if(type == "q16") train_wrap_q<int16_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  else if (type == "q32") train_wrap_q<int32_t, 0>(E, arch, dataset, lr, nhidden, type, chkpoint);

  else if (type == "f32") train<float>(0, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f64") train<double>(0, arch, dataset, lr, nhidden, type, chkpoint);

  else if (type == "f11") train_wrap_flex16<11>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f12") train_wrap_flex16<12>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f13") train_wrap_flex16<13>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f14") train_wrap_flex16<14>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f15") train_wrap_flex16<15>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f16") train_wrap_flex16<16>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f17") train_wrap_flex16<17>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f18") train_wrap_flex16<18>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f19") train_wrap_flex16<19>(E, arch, dataset, lr, nhidden, type, chkpoint);
  else if (type == "f20") train_wrap_flex16<20>(E, arch, dataset, lr, nhidden, type, chkpoint);
#endif

  else { cout << "unknown data type " << type << "." << endl; }

  return 0;
}
