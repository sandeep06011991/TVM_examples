'''
Task 1:
OP = + 
D = (A + B) + C
A,B,C,D are 2D tensors
Generate a kernel which is a composition of 2 steps.
Launch 2 kernels 
'''

from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

n = te.var("n")
m = te.var("m")

A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.placeholder((m, n), name="C")
D_1 = te.placeholder((m,n), name="D_1")
D_2 = te.placeholder((m,n), name="D_2")

D_1 = te.compute((m, n), lambda i, j: A[i, j] + B[i, j], name="D_1")
D_2 = te.compute((m,n), lambda i, j: D_1[i,j] + C[i,j], name= "D_2")
s = te.create_schedule([D_1.op,D_2.op])


s[D_1].bind(D_1.op.axis[0],te.thread_axis("blockIdx.x"))

s[D_2].bind(D_2.op.axis[0],te.thread_axis("blockIdx.x"))
s[D_2].bind(D_2.op.axis[1],te.thread_axis("threadIdx.x"))
#assert(False)
#print(tvm.lower(s, [A, B, C], simple_mode=True))
#assert(False)
func1 = tvm.build(s,[A,B,C,D_2],target="cuda",name="add")
#assert(False)

size = 1000
ctx = tvm.context('gpu')
A_1 = tvm.nd.array(np.ones((size,size),dtype=np.float32),ctx)
B_1 = tvm.nd.array(np.ones((size,size),dtype=np.float32),ctx)
C_1 = tvm.nd.array(np.ones((size,size),dtype = np.float32),ctx)
D_1 = tvm.nd.array(np.zeros((size,size),dtype = np.float32),ctx)

import time
s = time.time()
func(A_1,B_1,C_1,D_1)
e = time.time()
print(D_1)
print("TVM CUDA runtime :",e-s)






