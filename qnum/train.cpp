#include "common.hpp"
#include "data.hpp"
#include "mlp.hpp"
#include <thread>

using namespace std;

template<typename T> void train(double lr, int nhidden, const string& type, const char* checkpoint) {
  cout << "loading data..." << endl;
  auto ptrain = load_train<T>();
  auto ptest = load_test<T>();
  cout << "initializing network..." << endl;
  mlp_t<T> net(28*28, nhidden, 10);

  if(checkpoint != nullptr) {
    cout << "[DEBUG] Loading checkpoint from " << checkpoint << endl;
    net.load(checkpoint);
  }

  for (int epoch = 0;; ++epoch) {
    cout << "[TRAIN] epoch " << epoch << endl;
    auto samples = ptrain->shuffle();
    auto batch_size = 8;
    auto run = [&](const VectorXtvar<T> &img, 
                  const VectorXtvar<T> &label, 
                  double& loss_store,
                  int& correct_store,
                  bool backward) {
      auto label_predict = net.forward(img);
      auto loss = loss_crossent(label, label_predict);
      loss_store = static_cast<double>(loss.expr->val);
      correct_store = (argmax(label) == argmax(label_predict));
      if (backward) {
        net.backward(loss);
      }
    };

    std::vector<double> losses(batch_size);
    std::vector<int> corrects(batch_size);
    int total_correct = 0;
    double total_loss = 0.0;
    auto smpidx = ptrain->shuffle();

    for (auto i = 0; i < ptrain->size(); i += batch_size) {

      if (i % 10000 == 0) {
        char buf[256];
        sprintf(buf, "%s-h%d-lr%f-epoch-%d-step-%d.dmp", type.data(), nhidden, lr, epoch, i);
        net.save(buf);
      }

      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
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
      if (!std::isnormal(batch_loss)) {
        cout << "[DEBUG] abnormal loss detected. dump and ignore now." << endl;
        cout << "[DEBUG] current batch is: ";
        for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
          cout << smpidx[i+j] << " ";
        }
        cout << endl;
        char buf[256];
        sprintf(buf, "%s-h%d-lr%f-epoch-%d-step-%d.dmp", type.data(), nhidden, lr, epoch, i);
        net.save(buf);
        continue;
      }

      for(auto c: corrects) { total_correct += c; }
      for(auto l: losses) { total_loss += l; }

      auto current_acc = total_correct / ((double)i + batch_size);
      auto current_loss = total_loss / (i+batch_size);

      cout 
        << "[TRAIN] batchloss = " << setw(12) << batch_loss
        << ", avgloss = "       << setw(12) << current_loss
        << ", acc = "           << setw(12) << current_acc
        << ", sample "          << setw(5)  << i << "/" << ptrain->size()
        << endl;

      net.learn(T(lr));

      if ((i/batch_size) % 10 == 0) {
        net.check_histogram();
        // TODO check saturation, but on all nodes, not just weights
      }

    }

    total_correct = 0;
    total_loss = 0.0;
    cout << "[TEST] epoch " << epoch << endl;
    for (auto i = 0; i < ptest->size(); i += batch_size) {
      std::vector<std::thread> threads;
      for(auto j = 0; j < batch_size && i + j < ptrain->size(); ++j) {
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
      << "[TEST] avgloss = " << setw(12) << total_loss / (double)ptest->size()
      << ", acc = "          << setw(12) << total_correct / (double)ptest->size()
      << endl;
  }
}

template<typename T, typename ... Args> void train_wrap(int E, Args... args)
{
  switch (E) {
    case 1:
      train<qspace_number_t<T, 1>>(args...);
      break;
    case 2:
      train<qspace_number_t<T, 2>>(args...);
      break;
    case 3:
      train<qspace_number_t<T, 3>>(args...);
      break;
    case 4:
      train<qspace_number_t<T, 4>>(args...);
      break;
    case 5:
      train<qspace_number_t<T, 5>>(args...);
      break;
    case 6:
      train<qspace_number_t<T, 6>>(args...);
      break;
    case 7:
      train<qspace_number_t<T, 7>>(args...);
      break;
    case 8:
      train<qspace_number_t<T, 8>>(args...);
      break;
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

int main(int argc, char* argv[]) {

  if (argc < 5) {
    std::cout << "usage: train num_type ext_bits lr nhidden [checkpoint_file]" << std::endl;
  }

  std::string type = argv[1];
  int E = atoi(argv[2]);
  double lr = atof(argv[3]);
  int nhidden = atoi(argv[4]);
  char* chkpoint = nullptr;
  if (argc == 6) {
    chkpoint = argv[5];
  }

  if(type == "q8") train_wrap<int8_t>(E, lr, nhidden, type, chkpoint);
  else if(type == "q16") train_wrap<int16_t>(E, lr, nhidden, type, chkpoint);
  else if (type == "q32") train_wrap<int32_t>(E, lr, nhidden, type, chkpoint);
  else if (type == "f32") train<float>(lr, nhidden, type, chkpoint);
  else if (type == "f64") train<double>(lr, nhidden, type, chkpoint);
  else if (type == "f16") train<float16_t>(lr, nhidden, type, chkpoint);
  else if (type == "bf16") train<bfloat16_t>(lr, nhidden, type, chkpoint);
  else { cout << "unknown data type " << type << "." << endl; }

  return 0;
}
