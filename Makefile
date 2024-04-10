SHELL:=/bin/bash

# COMPILATION PARAMETERS

# -- python executable --
PYTHON=python
# -- compiler --
# default C++ compiler (g++, clang++, etc.) should already be available in Makefile as $(CXX)
# CXX=g++

# -- compiler flags --
# -g flag should not affect performance (https://stackoverflow.com/questions/10988318/g-is-using-the-g-flag-for-production-builds-a-good-idea)
CFLAGS=-std=c++20 -O3 -Wall -g
# -- linker flags --
# linker flags may be OS-dependent (https://stackoverflow.com/questions/714100/os-detecting-makefile)
# macOS needs specific linker flag (https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually)
LDFLAGS=$(shell [[ `uname -s` == Darwin ]] && echo -undefined dynamic_lookup)
# -- python flags --
# find actual python executable by resolving symlinks
# (https://stackoverflow.com/questions/7665/how-to-resolve-symbolic-links-in-a-shell-script/42918#42918)
PYTHON:=$(shell perl -MCwd -le 'print Cwd::abs_path(shift)' "`which $(PYTHON)`")
PYTHONFLAGS=
# python libraries
# PYTHONFLAGS+=$(shell $(PYTHON)-config --includes)
PYTHONFLAGS+=$(shell $(PYTHON) -m pybind11 --includes)

.PHONY: clean

all: bind.so

clean:
	rm -rf *.o *.so

# LIBRARIES

%.so: CFLAGS+=-fPIC $(PYTHONFLAGS)
%.so: %.cpp
	$(CXX) -o $@ -shared $^ $(CFLAGS) $(LDFLAGS)

