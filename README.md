# TVM_examples
Writing operators and optimizations for TVM
Chapter 1:
	a. Tensor Expression and Schedules(https://tvm.apache.org/docs/tutorials/index.html#tensor-expression-and-schedules)
	b. Optimize Tensor Operators(https://tvm.apache.org/docs/tutorials/index.html#optimize-tensor-operators)


1. Tutorial 1:
	Create a custom operator a(OP)b.
	Generate code for it in CUDA and cpu. 

2. Tutorial 2:
	Transform a(OP)b(OP)c  into OP(a,b,c)

3. Tutorial 3:
	Transform code with tiling and spliting
	Lowering with code gen


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
