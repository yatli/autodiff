#include "entry.hpp"

using namespace std;

// inspect entry
template<typename T> void entry(int E, const string& arch, const string& dataset, double lr, int nhidden, const string& type, const char* checkpoint) {
  auto dataset_tup = load_data<T>(dataset);
  const dataset_t<T>* ptrain = std::get<0>(dataset_tup);
  const dataset_t<T>* ptest = std::get<1>(dataset_tup);
  nn_t<T>* pnet = init_net<T>(arch, nhidden, ptrain);

  if(checkpoint != nullptr) {
    cout << "[DEBUG] Loading checkpoint from " << checkpoint << endl;
    pnet->load(checkpoint);
  }

  //pnet->check_histogram();
  //pnet->check_saturation();
  //pnet->dump_weights();
  int smpidx;
  while(true) {
    scanf("%d", &smpidx);
    if(smpidx < 0 || smpidx >= ptrain->size) {
      break;
    }
    auto label = ptrain->labels[smpidx];
    auto img = ptrain->imgs[smpidx];
    auto label_predict = pnet->forward(img);
    auto loss = loss_crossent(label, label_predict);
    cout << "label: " << label << endl;
    cout << "prediction: " << label_predict << endl;
    cout << "loss: " << loss << endl;
  }
}

int main(int argc, char* argv[]) {
  return launch(argc, argv);
}
