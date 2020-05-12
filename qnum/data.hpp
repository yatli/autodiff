#pragma once

#include "common.hpp"
//#include <execution>
#include <algorithm>
#include <random>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr auto c_mnist_dir = "/media/data/mnist";
constexpr auto c_mnist_train_img_file = "train-images.idx3-ubyte";
constexpr auto c_mnist_train_label_file = "train-labels.idx1-ubyte";
constexpr auto c_mnist_train_size = 60000;
constexpr auto c_mnist_test_img_file = "t10k-images.idx3-ubyte";
constexpr auto c_mnist_test_label_file = "t10k-labels.idx1-ubyte";
constexpr auto c_mnist_test_size = 10000;
constexpr auto c_mnist_imgh = 28;
constexpr auto c_mnist_imgw = 28;

typedef uint8_t mnist_img_t[c_mnist_imgh * c_mnist_imgw];
typedef uint8_t mnist_label_t;

constexpr auto c_cifar10_dir = "/media/data/cifar-10-batches-bin";
constexpr auto c_cifar10_train_fmt = "data_batch_%d.bin";
constexpr auto c_cifar10_test_file = "test_batch.bin";
constexpr auto c_cifar10_batch_size = 10000;
constexpr auto c_cifar10_train_size = 50000;
constexpr auto c_cifar10_test_size = 10000;
constexpr auto c_cifar10_imgh = 32;
constexpr auto c_cifar10_imgw = 32;
constexpr auto c_cifar10_imgc = 3;

struct cifar_datapoint_t {
  uint8_t label;
  uint8_t image[c_cifar10_imgc][c_cifar10_imgw][c_cifar10_imgh];
};
typedef uint8_t cifar_label_t;

template<typename T>
struct dataset_t {
  VectorXtvar<T> *imgs;
  VectorXtvar<T> *labels;
  int size;
  int nchannel;
  int width;
  int height;
  int nclass;

  dataset_t(int n, int c, int h, int w, int label_classes) 
    : size(n), nchannel(c), width(w), height(h), nclass(label_classes)
  { 
    imgs = new VectorXtvar<T>[n];
    labels = new VectorXtvar<T>[n];
  }

  std::vector<int> shuffle() const {
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    return idx;
  }
};

template<typename T, int sz>
const dataset_t<T>* load_mnist_data(const char* img_file, const char* label_file) {
  char buf[256];
  sprintf(buf, "%s/%s", c_mnist_dir, img_file);
  FILE* img_fp = fopen(buf, "rb");
  sprintf(buf, "%s/%s", c_mnist_dir, label_file);
  FILE* label_fp = fopen(buf, "rb");

  if (!img_fp || !label_fp) {
    printf("cannot read data.\n");
    exit(-1);
  }

  dataset_t<T>* p = new dataset_t<T>(sz, 1, c_mnist_imgh, c_mnist_imgw, 10);

  fread(buf, 16, 1, img_fp);
  fread(buf, 8, 1, label_fp);

  std::vector<mnist_img_t> imgs(sz);
  std::vector<mnist_label_t> labels(sz);

  fread(imgs.data(), sizeof(mnist_img_t), sz, img_fp);
  fread(labels.data(), sizeof(mnist_label_t), sz, label_fp);

  fclose(img_fp);
  fclose(label_fp);

  std::vector<int> idx(sz);
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(idx.begin(), idx.end(), [&](auto i) {
    p->imgs[i].resize(c_mnist_imgh * c_mnist_imgw);
    p->labels[i].resize(10);
    for (int j = 0; j < 10; ++j) {
      if (labels[i] == j) {
        p->labels[i][j] = autodiff::reverse::constant<T>(T(1.0));
      }
      else {
        p->labels[i][j] = autodiff::reverse::constant<T>(T(0.0));
      }
    }
    for (int j = 0; j < c_mnist_imgh * c_mnist_imgw; ++j) {
      p->imgs[i][j] = autodiff::reverse::constant<T>(T(imgs[i][j] / 255.0));
    }
  });

  return p;
}

std::vector<cifar_datapoint_t> load_cifar_batch(const char* filename) {
  char buf[256];
  sprintf(buf, "%s/%s", c_cifar10_dir, filename);
  FILE* fp = fopen(buf, "rb");
  if (!fp) {
    printf("cifar: cannot read data.\n");
    exit(-1);
  }
  std::vector<cifar_datapoint_t> data(c_cifar10_batch_size);
  fread(data.data(), sizeof(cifar_datapoint_t), c_cifar10_batch_size, fp);
  fclose(fp);
  return data;
}

template<typename T, int sz>
const dataset_t<T>* load_cifar_data(bool train) {

  dataset_t<T>* p = new dataset_t<T>(sz, c_cifar10_imgc, c_cifar10_imgh, c_cifar10_imgw, 10);
  p->height = c_cifar10_imgh;
  p->width = c_cifar10_imgw;
  p->nchannel = c_cifar10_imgc;

  int offset = 0;
  std::vector<cifar_datapoint_t> data;

  auto load_batch = [&](){
    for(int i=0; i<c_cifar10_batch_size; ++i) {
      p->imgs[offset].resize(c_cifar10_imgc * c_cifar10_imgw * c_cifar10_imgh);
      p->labels[offset].resize(10);
      for (int j = 0; j < 10; ++j) {
        if (data[i].label == j) {
          p->labels[offset][j] = autodiff::reverse::constant<T>(T(1.0));
        }
        else {
          p->labels[offset][j] = autodiff::reverse::constant<T>(T(0.0));
        }
      }
      for(int c = 0; c < c_cifar10_imgc; ++c) {
        for(int h = 0; h < c_cifar10_imgh; ++h) {
          for(int w = 0; w < c_cifar10_imgw; ++w) {
            p->imgs[offset][w + h * c_cifar10_imgw + c * c_cifar10_imgw * c_cifar10_imgh] = autodiff::reverse::constant<T>(T(data[i].image[c][h][w] / 255.0));
          }
        }
      }
      ++offset;
    }
  };

  if(train) {
    for(int batch = 0; batch < c_cifar10_train_size / c_cifar10_batch_size; ++batch) {
      char buf[256];
      sprintf(buf, c_cifar10_train_fmt, batch + 1);
      data = load_cifar_batch(buf);
      load_batch();
    }
  } else {
    data = load_cifar_batch(c_cifar10_test_file);
    load_batch();
  }

  return p;
}


template<typename T>
const dataset_t<T>* load_mnist_train() {
  return load_mnist_data<T, c_mnist_train_size>(c_mnist_train_img_file, c_mnist_train_label_file);
}

template<typename T>
const dataset_t<T>* load_mnist_test() {
  return load_mnist_data<T, c_mnist_test_size>(c_mnist_test_img_file, c_mnist_test_label_file);
}

template<typename T>
const dataset_t<T>* load_cifar10_train() {
  return load_cifar_data<T, c_cifar10_train_size>(true);
}

template<typename T>
const dataset_t<T>* load_cifar10_test() {
  return load_cifar_data<T, c_cifar10_test_size>(false);
}
