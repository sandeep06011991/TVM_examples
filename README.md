# TVM_examples
Writing operators and optimizations for TVM
References:
	a. Tensor Expression and Schedules(https://tvm.apache.org/docs/tutorials/index.html#tensor-expression-and-schedules)
	b. Optimize Tensor Operators(https://tvm.apache.org/docs/tutorials/index.html#optimize-tensor-operators)


1. Tutorial 1: 
	a. Create a custom operator a(OP)b.
	b. Generate naive code for in CUDA. 

2. Tutorial 2: (DONE)
	Generate 2 kernels to evaluate a composition. 
	Transform a(OP)b(OP)c  into OP(a,b,c)
	Try Fusion (Is there an optimization which can do this)

3. Tutorial 3: 
	Naive Matrix Transpose

5. Final Deliverable: Use TVM to generate cache effienet matrix transpose schedule. 

## Running Instructions

export PYTHONPATH={Path to directory containing tvm built}	
