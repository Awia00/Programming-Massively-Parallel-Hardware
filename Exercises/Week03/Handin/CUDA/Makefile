CXX        = nvcc
CCC        = gcc -fopenmp -lm

SOURCES_CPP =Main.cu
HELPERS     =Host.cu.h Kernels.cu.h
EXECUTABLEGPU  =week3GPU
EXECUTABLEOMP  =week3OMP


default: compile

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLEGPU)
$(EXECUTABLEGPU): 
	$(CXX) -o $(EXECUTABLEGPU) $(SOURCES_CPP)

omp: $(EXECUTABLEOMP)
$(EXECUTABLEOMP): 
	cp $(SOURCES_CPP) $(SOURCES_CPP).cpp
	$(CCC) -o $(EXECUTABLEOMP) $(SOURCES_CPP).cpp

run: $(EXECUTABLEGPU)
	./$(EXECUTABLEGPU)

runOMP: $(EXECUTABLEOMP)
	./$(EXECUTABLEOMP)

clean:
	rm -f $(EXECUTABLEGPU)
	rm -f $(EXECUTABLEOMP)

