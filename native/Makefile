# Build the jolt_llama shim against a llama.cpp build.
#
# LLAMA_SRC   llama.cpp source tree (for headers)
# LLAMA_LIB   directory holding libllama.so / libggml*.so
#
# Both default to the coordinates recorded in config/coordinates.edn; override
# on the command line for a different llama.cpp build.

LLAMA_SRC ?= $(error set LLAMA_SRC to a llama.cpp source tree)
LLAMA_LIB ?= $(error set LLAMA_LIB to a directory containing libllama.so)

CC      ?= gcc
CFLAGS  ?= -O2 -fPIC -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra

# llama.cpp replaced llama_model_params' use_mmap/use_mlock booleans with a
# load_mode enum. Probed from the header rather than gated on a version, since
# llama.cpp exposes no API version macro. The shim keeps the old two-boolean
# surface either way, so nothing above it changes.
HAS_LOAD_MODE := $(shell grep -qs 'enum llama_load_mode' $(LLAMA_SRC)/include/llama.h && echo -DJL_HAS_LOAD_MODE)
CFLAGS  += $(HAS_LOAD_MODE)
INCS     = -I. -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include
LIBS     = -L$(LLAMA_LIB) -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_LIB)

all: libjolt_llama.so smoke

libjolt_llama.so: jolt_llama.c jolt_llama.h
	$(CC) $(CFLAGS) -shared -o $@ jolt_llama.c $(INCS) $(LIBS)

# C smoke test: the independent reference the Jolt tests are compared against.
smoke: smoke.c libjolt_llama.so
	$(CC) $(CFLAGS) -o $@ smoke.c $(INCS) -L. -ljolt_llama -Wl,-rpath,$(CURDIR) -Wl,-rpath,$(LLAMA_LIB)

clean:
	rm -f libjolt_llama.so smoke *.o

.PHONY: all clean
