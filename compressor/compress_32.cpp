#include <iostream>
#include <fstream>
#include <cstring> 
#include <algorithm>
#include <vector>
#include <chrono>
#include "la_vector.hpp"
#include <vector>
#include <cstdlib>
//#include <format>


std::ofstream outfile("result.txt", std::ios::app); 

void load_code(char* filename, uint8_t*& data, unsigned& num, unsigned& m) { 
    std::ifstream in(filename, std::ios::binary);    
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        return;
    }

    in.seekg(0, std::ios::end);    
    std::ios::pos_type ss = in.tellg();    
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / m / sizeof(uint8_t)); 
    data = new uint8_t[num * m];
    
    in.seekg(0, std::ios::beg);    
    in.read(reinterpret_cast<char*>(data), fsize);    
    in.close();
}

void pack32(const uint8_t* uint8Array, uint32_t* uint32Array, size_t uint8Count) {  
    size_t uint32Count = uint8Count / 4;  
    for (size_t i = 0; i < uint32Count; ++i) {  
        uint32_t value = 0;  
        memcpy(&value, &uint8Array[i * 4], 4);  
        uint32Array[i] = value;  
    }  
}

template <typename T>
void pack(const uint8_t* uint8Array, T* uintArray, size_t uint8Count) {
  size_t uintCount = uint8Count / sizeof(T);
  for (size_t i = 0; i < uintCount; ++i) {
    T value = 0;
    memcpy(&value, &uint8Array[i * sizeof(T)], sizeof(T));
    uintArray[i] = value;
  }
}


template <int EPSILON_BITS_RUN=16>
void run_exp(std::vector<uint32_t>& code_ints_vec) {
    auto start_phase3 = std::chrono::high_resolution_clock::now();
    la_vector<uint32_t, EPSILON_BITS_RUN> v1(code_ints_vec); //构建linear  model
    std::cout << "EPSILON_BITS: " << EPSILON_BITS_RUN << std::endl;
    outfile << "EPSILON_BITS: " << EPSILON_BITS_RUN << std::endl;
    auto end_phase3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase3_duration = end_phase3 - start_phase3;
    std::cout << "Time taken in phase 3 (compression: linear model): " << phase3_duration.count() << " seconds" << std::endl;
    std::cout << "segs: " << v1.segments_count() << std::endl;
    std::cout << "bytes:" << v1.size_in_bytes() << std::endl; 
    outfile << "Time taken in phase 3 (compression: linear model): " << phase3_duration.count() << " seconds" << std::endl;
    outfile << "segs: " << v1.segments_count() << std::endl;
    outfile << "bytes: " << v1.size_in_bytes() << std::endl; 
  
    uint32_t* out = new uint32_t[code_ints_vec.size()]; 
    auto start_phase4 = std::chrono::high_resolution_clock::now();
    v1.decode(out);
    auto end_phase4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase4_duration = end_phase4 - start_phase4;
    std::cout << "Time taken in phase 4 (decode): " << phase4_duration.count() << " seconds" << std::endl;
    outfile << "Time taken in phase 4 (decode): " << phase4_duration.count() << " seconds" << std::endl;
    
    delete[] out;
}


int main(int argc, char** argv) {
  if (argc < 4) {
      std::cerr << "Usage: " << argv[0] << " <filename> <m> <nbits>" << std::endl;
      return 1;
  }
  // points_num = 1000000000;
  // m = 4
  unsigned points_num, m, nbits; 
  m = atoi(argv[2]);
  nbits = atoi(argv[3]);
  uint8_t* codes = NULL;
  load_code(argv[1], codes, points_num, m); 
 
  if (codes == nullptr) {
      return 1;
  }


  std::cout << "code vector num:"<< points_num << std::endl;
  std::cout << "construct PQ(m=" << m << ", nbits=" << nbits << ")." << std::endl;
  
  if (!outfile.is_open()) {
      std::cerr << "Failed to open output file: results1.txt" << std::endl;
      delete[] codes;
      return 1;
  }
  
  auto word_size = m * nbits; 
  if (word_size == 32) {
    // Phase 1: project/ binary conjection,     
    auto start_phase1 = std::chrono::high_resolution_clock::now();
    uint32_t* code_ints = new uint32_t[points_num];
    pack32(codes, code_ints, points_num * m);
    auto end_phase1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase1_duration = end_phase1 - start_phase1;
    std::cout << "Time taken in phase 1 (project/binary contaction): " << phase1_duration.count() << " seconds" << std::endl;
    outfile << "Time taken in phase 1 (project/binary contaction): " << phase1_duration.count() << " seconds" << std::endl;

    // Phase 2: Sorting
    auto start_phase2 = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> code_ints_vec(code_ints, code_ints+points_num);
    std::sort(code_ints_vec.begin(), code_ints_vec.end());
    auto end_phase2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase2_duration = end_phase2 - start_phase2;
    std::cout << "Time taken in phase 2 (sorting): " << phase2_duration.count() << " seconds" << std::endl;
    outfile << "Time taken in phase 2 (sorting): " << phase2_duration.count() << " seconds" << std::endl;

    run_exp<4>(code_ints_vec);
    run_exp<6>(code_ints_vec);
    run_exp<8>(code_ints_vec);
    run_exp<10>(code_ints_vec);
    run_exp<12>(code_ints_vec);
    run_exp<14>(code_ints_vec);
    run_exp<16>(code_ints_vec);
    

    delete[] codes;
    delete[] code_ints;
  }
  else {
    delete[] codes;
    std::cout << "unsupport size: " << word_size << std::endl;
  }

  outfile.close();
 
  return 0;

}
