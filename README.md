# TVM_examples
Writing operators and optimizations for TVM
Chapter 1:
	a. Tensor Expression and Schedules(https://tvm.apache.org/docs/tutorials/index.html#tensor-expression-and-schedules)
	b. Optimize Tensor Operators(https://tvm.apache.org/docs/tutorials/index.html#optimize-tensor-operators)


1. Tutorial 1: (DONE)
	Create a custom operator a(OP)b.
	Generate naive code for in CUDA. 

2. Tutorial 2: (DONE)
	Generate 2 kernels to evaluate a composition. 
	Transform a(OP)b(OP)c  into OP(a,b,c)
	Try Fusion (Is there an optimization which can do this)

3. Tutorial 3: 
	Matrix Transpose

5. Final Deliverable:
	
	def dot_product(A,B):
		N,F = A.shape
		N,F = B.shape
		for i in N:
			C[i] = sum_{for all j}(A[i,j]*B[i,j])

	def spmv(A,H):
		A = scipy_sparse_matrix(N,N)
		H = FeatureVector(N,F)
		V[i,:] = A[i,j] * H[j,:]

## Running Instructions

export PYTHONPATH={Path to directory containing tvm built}	
