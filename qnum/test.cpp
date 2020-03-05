#include <iostream>
#include "qnum.hpp"

using namespace qnum;

int main(int argc, char** argv) {
  //std::cout << std::numeric_limits<int32_t>::digits << std::endl;
  //std::cout << qnum::qnum32_t::T_max() << std::endl;
  //std::cout << qnum::qnum32_t::ext_bits() << std::endl;
  //std::cout << qnum::qnum32_t::ext_max() << std::endl;
  //std::cout << qnum::qnum32_t::frac_bits() << std::endl;
  //std::cout << qnum::qnum32_t::frac_max() << std::endl;

  //std::cout << qnum32_t(0.1) << std::endl;
  //std::cout << qnum32_t(0.1).prev() << std::endl;
  //std::cout << qnum32_t(0.1).next() << std::endl;
  //std::cout << qnum32_t(0.01) << std::endl;
  //std::cout << qnum32_t(0.001) << std::endl;
  //std::cout << qnum32_t(0.0001) << std::endl;
  //std::cout << qnum32_t(0.00001) << std::endl;

  for(qnum32_t i = 0.15; ; i = i.next()){
    std::cout << (i * i) + i << std::endl;
  }
  return 0;
}
