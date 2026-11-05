# Minimal C3v2-1 targets. The full C2+C3 CMake build remains unchanged.
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
INCLUDES = -Iinclude -Ithird_party

.PHONY: all c3v2_1 test demo clean
all: c3v2_1

c3v2_1: build/c3v2_1_demo build/c3v2_1_test

build:
	mkdir -p build reports

build/c3v2_1_demo: src/c3v2_1.cpp src/c3v2_1_demo.cpp include/c3v2_1.h | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/c3v2_1.cpp src/c3v2_1_demo.cpp -o $@

build/c3v2_1_test: src/c3v2_1.cpp tests/c3v2_1_test.cpp include/c3v2_1.h | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/c3v2_1.cpp tests/c3v2_1_test.cpp -o $@

test: build/c3v2_1_test
	./build/c3v2_1_test

demo: build/c3v2_1_demo
	./build/c3v2_1_demo reports/c3v2_1_demo.json

clean:
	rm -f build/c3v2_1_demo build/c3v2_1_test
