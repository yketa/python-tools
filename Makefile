SHELL:=/bin/bash

# COMPILATION PARAMETERS

# -- python executable --
PYTHON=python
# -- compiler flags --
# -g flag should not affect performance (https://stackoverflow.com/questions/10988318)
CFLAGS=-std=c++20 -O3 -Wall -g
# -- linker flags --
LDFLAGS=

# find actual python executable by resolving symlinks
# (https://stackoverflow.com/a/42918/7385044)
PYTHON:=$(shell perl -MCwd -le 'print Cwd::abs_path(shift)' "`which $(PYTHON)`")
# python libraries
PYTHONFLAGS=$(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON)-config --includes)

.PHONY: clean

all: bind.so

clean:
	rm -rf *.o *.so *.d

# LIBRARIES

bind.so: CFLAGS+=-fPIC $(PYTHONFLAGS)
bind.so: bind.cpp
	$(CXX) -o $@ -shared $^ $(CFLAGS) $(LDFLAGS)

