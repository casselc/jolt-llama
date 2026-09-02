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

# The NATIVE RUNTIME IDENTITY, embedded at build time.
#
# Saved native state is not portable across llama.cpp builds -- the same model
# serialised to 20263632 bytes on one and 20263652 on another -- and the shim
# ABI cannot stand in for that, since the shim can be byte-identical across two
# different llama.cpp trees. So the blob has to name the runtime that wrote it.
#
# Read from LLAMA_SRC's git checkout here, at BUILD time, and never by shelling
# out to git during inference. A tree that is not a git checkout yields
# "unknown:...", which the compatibility check treats as never matching --
# honest, and loud enough to notice.
LLAMA_SHA   := $(shell git -C $(LLAMA_SRC) rev-parse HEAD 2>/dev/null)
LLAMA_DIRTY := $(shell git -C $(LLAMA_SRC) status --porcelain 2>/dev/null | head -1)
ifeq ($(LLAMA_SHA),)
  JL_BUILD_ID := unknown:$(notdir $(LLAMA_SRC))
else ifeq ($(LLAMA_DIRTY),)
  JL_BUILD_ID := llama.cpp:$(LLAMA_SHA):clean
else
  JL_BUILD_ID := llama.cpp:$(LLAMA_SHA):dirty
endif
CFLAGS  += -DJL_LLAMA_BUILD_ID=\"$(JL_BUILD_ID)\"

# A promoted build must not ship an unknown coordinate. `make promote` fails
# rather than producing a library whose saved state cannot be attributed.
.PHONY: promote
promote:
	@case "$(JL_BUILD_ID)" in \
	  llama.cpp:*:clean) echo "build id: $(JL_BUILD_ID)";; \
	  *) echo "REFUSING: build id is '$(JL_BUILD_ID)', not a clean pinned llama.cpp" >&2; exit 1;; \
	esac
	@$(MAKE) all
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
