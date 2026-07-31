AMREX_HOME ?= $(HOME)/amrex
HDF5_HOME ?= /usr/lib/x86_64-linux-gnu/hdf5/serial

DEBUG = FALSE
COMP = gnu
CXXSTD = c++23
DIM = 3
USE_PARTICLES = TRUE
USE_MPI = FALSE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_HDF5 = TRUE
PRECISION = DOUBLE
TINY_PROFILE = TRUE
PROFILE = FALSE

include Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

test_dir = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
test_name = $(shell basename $(test_dir))

t: test
test: aPICtest
aPICtest: ok

ok: $(executable) Makefile
	@cd $(test_dir) && echo -n RUN $(test_name)"...  " && ./$< > /dev/null && touch ok && echo ok

c: aPICclean
aPICclean: clean
	rm -f $(test_dir)/ok
