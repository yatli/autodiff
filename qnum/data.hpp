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

template<typename T, int sz>
struct mnist_t {
  VectorXtvar<T> imgs[sz];
  VectorXtvar<T> labels[sz];
  vector<int> shuffle() const {
    vector<int> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idx.begin(), idx.end(), g);
    return idx;
  }
  constexpr auto size() const { return sz; }
};

template<typename T, int sz>
const mnist_t<T, sz>* load_data(const char* img_file, const char* label_file) {
  char buf[256];
  sprintf(buf, "%s/%s", c_mnist_dir, img_file);
  FILE* img_fp = fopen(buf, "rb");
  sprintf(buf, "%s/%s", c_mnist_dir, label_file);
  FILE* label_fp = fopen(buf, "rb");

  if (!img_fp || !label_fp) {
    printf("cannot read data.\n");
    exit(-1);
  }

  mnist_t<T, sz>* p = new mnist_t<T, sz>();

  fread(buf, 16, 1, img_fp);
  fread(buf, 8, 1, label_fp);

  vector<mnist_img_t> imgs(sz);
  vector<mnist_label_t> labels(sz);

  fread(imgs.data(), sizeof(mnist_img_t), sz, img_fp);
  fread(labels.data(), sizeof(mnist_label_t), sz, label_fp);

  fclose(img_fp);
  fclose(label_fp);

  vector<int> idx(sz);
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

template<typename T>
const mnist_t<T, c_mnist_train_size>* load_train() {
  return load_data<T, c_mnist_train_size>(c_mnist_train_img_file, c_mnist_train_label_file);
}

template<typename T>
const mnist_t<T, c_mnist_test_size>* load_test() {
  return load_data<T, c_mnist_test_size>(c_mnist_test_img_file, c_mnist_test_label_file);
}
