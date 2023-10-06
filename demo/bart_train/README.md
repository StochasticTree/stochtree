BART CLI Example
================

To run this example, you must first build the project from the command line by navigating to 
the main project directory and running the following commands. 
(See [here](https://cmake.org/install/) for details on installing cmake.)

```
rm -rf build
mkdir build
cmake -S . -B build
cmake --build build
```

If you have multiple cores available, you can enable a multi-core build by adding `-j [num_cores]` to the build command above. 
For example, `cmake --build build -j 10` will build the project using 10 cores.

BART training
--------------

Run the following commands from the main project folder:

```bash
./build/stochtree config=demo/bart_train/bart_train.conf
```
