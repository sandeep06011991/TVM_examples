'''
Task 1:
OP = + 
C = A + B
A,B,C are 2D tensors
Generate a naive GPU runnable kernel and measure performamce with numpy. 
'''

from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

n = te.var("n")
m = te.var("m")

A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] + B[i, j], name="C")

s = te.create_schedule(C.op)

s[C].bind(C.op.axis[0],te.thread_axis("blockIdx.x"))
s[C].bind(C.op.axis[1],te.thread_axis("threadIdx.x"))

#print(tvm.lower(s, [A, B, C], simple_mode=True))
func = tvm.build(s,[A,B,C],target="cuda",name="add")

size = 1000
ctx = tvm.context('gpu')
A_1 = tvm.nd.array(np.ones((size,size),dtype=np.float32),ctx)
B_1 = tvm.nd.array(np.ones((size,size),dtype=np.float32),ctx)
C_1 = tvm.nd.array(np.zeros((size,size),dtype = np.float32),ctx)

import time
s = time.time()
func(A_1,B_1,C_1)
e = time.time()
print("TVM CUDA runtime :",e-s)

A_1 = np.ones((size,size))
B_1 = np.ones((size,size))
s = time.time()
C_1 = A_1 + B_1
e = time.time()
print("Numpy runtime:",e-s)





