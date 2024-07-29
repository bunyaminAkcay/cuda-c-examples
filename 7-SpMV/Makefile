# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11

# Target executable
TARGET = SpMV

# Source files
CU_SOURCES = SpMV.cu
CPP_SOURCES = MatrixReader.cpp

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Rules
all: $(TARGET)

$(TARGET): $(CU_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) -o $@ $^
	rm -f $(CU_OBJECTS) $(CPP_OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJECTS) $(CPP_OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
