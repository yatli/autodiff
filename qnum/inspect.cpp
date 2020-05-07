#include "common.hpp"
#include "data.hpp"
#include "mlp.hpp"
#include <thread>
using namespace std;

template<typename T> void train(double lr, int nhidden, const string& type, const char* checkpoint) {
  cout << "loading data..." << endl;
  auto ptrain = load_train<T>();
  cout << "initializing network..." << endl;
  mlp_t<T> net(28*28, nhidden, 10);

  if(checkpoint != nullptr) {
    cout << "[DEBUG] Loading checkpoint from " << checkpoint << endl;
    net.load(checkpoint);
  }

  //net.check_histogram();
  //net.check_saturation();
  //net.dump_weights();
  int smpidx;
  while(true) {
    scanf("%d", &smpidx);
    if(smpidx < 0 || smpidx >= ptrain->size()) {
      break;
    }
    auto label = ptrain->labels[smpidx];
    auto img = ptrain->imgs[smpidx];
    auto label_predict = net.forward_debug(img);
    auto loss = loss_crossent(label, label_predict);
    cout << "label: " << label << endl;
    cout << "prediction: " << label_predict << endl;
    cout << "loss: " << loss << endl;
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
    case 9:
      train<qspace_number_t<T, 9>>(args...);
      break;
    case 10:
      train<qspace_number_t<T, 10>>(args...);
      break;
    case 11:
      train<qspace_number_t<T, 11>>(args...);
      break;
    case 12:
      train<qspace_number_t<T, 12>>(args...);
      break;
    default:
      std::cout << "unsupported extension bit number" << std::endl;
  }
}

int main(int argc, char* argv[]) {
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
  else {
    cout << "unknown data type " << type << "." << endl;
  }

  return 0;
}
