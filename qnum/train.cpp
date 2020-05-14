#include "entry.hpp"
#include <thread>

using namespace std;

// training entry
template<typename T> void entry(int E, const string& arch, const string& dataset, double lr, int nhidden, const string& type, const char* checkpoint) {
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
    auto batch_size = g_batch_size;
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

      pnet->seed();

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

int main(int argc, char* argv[]) {
  return launch(argc, argv);
}
