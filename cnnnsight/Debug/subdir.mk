################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cnn.cu \
../cnnDriver.cu \
../dataTrans.cu \
../dense.cu \
../denseDriver.cu \
../maxpoolDriver.cu \
../maxpooling.cu \
../relu.cu \
../reluDriver.cu \
../softmax.cu \
../softmaxDriver.cu \
../writeOut.cu 

CPP_SRCS += \
../ReadIn_test.cpp \
../cnn_layer_test.cpp \
../read1D.cpp \
../read2D.cpp \
../readIn.cpp \
../writeOutDriver.cpp 

O_SRCS += \
../ReadIn_test.o 

OBJS += \
./ReadIn_test.o \
./cnn.o \
./cnnDriver.o \
./cnn_layer_test.o \
./dataTrans.o \
./dense.o \
./denseDriver.o \
./maxpoolDriver.o \
./maxpooling.o \
./read1D.o \
./read2D.o \
./readIn.o \
./relu.o \
./reluDriver.o \
./softmax.o \
./softmaxDriver.o \
./writeOut.o \
./writeOutDriver.o 

CU_DEPS += \
./cnn.d \
./cnnDriver.d \
./dataTrans.d \
./dense.d \
./denseDriver.d \
./maxpoolDriver.d \
./maxpooling.d \
./relu.d \
./reluDriver.d \
./softmax.d \
./softmaxDriver.d \
./writeOut.d 

CPP_DEPS += \
./ReadIn_test.d \
./cnn_layer_test.d \
./read1D.d \
./read2D.d \
./readIn.d \
./writeOutDriver.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


