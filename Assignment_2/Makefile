CXX = icc
CXXFLAGS = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dot_product

all : $(TARGET1)
    
$(TARGET1) : dot_product_openmp.c kernel.c dot_product_openmp.h
	$(CXX) $(CXXFLAGS) -o $(TARGET1) dot_product_openmp.c kernel.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
