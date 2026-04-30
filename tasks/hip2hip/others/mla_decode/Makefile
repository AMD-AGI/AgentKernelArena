# Makefile

# Compiler
HIPCC = hipcc

# Source and target
SRC = mla_decode.hip
TARGET = applications_mla_decode

# Compiler flags. Targets gfx950 primary; gfx942 also covered.
# OpenMP parallelizes only the host fp32 reference (correctness check).
CFLAGS = -O3 -ffast-math --offload-arch=gfx950 --offload-arch=gfx942 -munsafe-fp-atomics -std=c++17 -fopenmp

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
