# BitTuner

## Core compressor

Compile: 

`g++ compress_32.cpp --std=c++17 -I./include -I./lib/sdsl-lite/include -o compress_32 -fopenmp`

Run: 

`./compress_32 <fname> 4 8`


## BitTuner GUI

`python ./gui/main.py`
