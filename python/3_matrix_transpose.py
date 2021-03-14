'''
Task 1:
Transpose Naive 
A.T
A are 2D tensors
Generate a naive kernel which does transpose. 
Start to do tiling and print the final speed up. 

'''

from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
import time 


def tiled_and_shared():
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((m,n),name = "A")
    B = te.compute((m,n), lambda i,j: A[j,i], name = "B")
    s = te.create_schedule([B.op])
    AA = s.cache_read(A,"shared",[B])
        
    x,y,f1,f2 = s[AA].tile(AA.op.axis[0],AA.op.axis[1],x_factor=32,y_factor=32)
    block = s[AA].fuse(x,y)
    s[AA].bind(block,te.thread_axis("blockIdx.x"))
    #s[AA].bind(f1,te.thread_axis("threadIdx.x"))
     
    x,y,f1,f2 = s[B].tile(B.op.axis[0],B.op.axis[1],x_factor=32,y_factor=32)
    block = s[B].fuse(x,y)
    s[AA].compute_at(s[B],block)
    s[B].bind(block,te.thread_axis("blockIdx.x"))
    #s[B].bind(f1,te.thread_axis("threadIdx.x"))
     
    
    #s[AA].bind(AA.op.axis[1],te.thread_axis("blockIdx.x"))
    
    print(tvm.lower(s,[A,B],simple_mode = True))
    F = tvm.build(s,[A,B], target = "cuda", name = "tled_shared")
    return F
    pass

def tiled_but_not_shared():
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((m,n),name = "A")
    B = te.compute((m,n), lambda i,j: A[j,i], name = "B")
    s = te.create_schedule([B.op])
    x,y,f1,f2 = s[B].tile(B.op.axis[0],B.op.axis[1],x_factor=32,y_factor=32)
    block = s[B].fuse(x,y)
    s[B].bind(block,te.thread_axis("blockIdx.x"))
    s[B].bind(f2,te.thread_axis("threadIdx.x"))
    F = tvm.build(s,[A,B], target = "cuda", name = "tiled_but_not_shared")
    return F

def get_shared_but_not_tiled():
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((m, n), name="A")
    B = te.compute((m, n), lambda i, j: A[j, i], name="B")
    s = te.create_schedule([B.op])
    AA = s.cache_read(A,"shared",[B])
    s[AA].bind(AA.op.axis[1],te.thread_axis("threadIdx.x"))
    s[B].bind(B.op.axis[1],te.thread_axis("threadIdx.x"))
    F = tvm.build(s,[A,B],target = "cuda",name = "transposeShared")
    return F

#x0,y0,xi,yi = s[B].tile(B.op.axis[0],B.op.axis[1],x_factor=32,y_factor=32)
#print(s[A])
#print(s[B])
#s[A].tile(A.op.axis,32,32)
#s[B].reorder(x0,y0,yi,xi)
#block = s[B].fuse(x0,y0)
#s[B].bind(block,te.thread_axis("blockIdx.x"))
#s[B].bind(yi,te.thread_axis("threadIdx.x"))        

#s[B].bind(yi,te.thread_axis("threadIdx.y"))       
#AA = s.cache_read(A,"shared",[B])
#s[AA].compute_at(s[B],block)
#thread_x = te.thread_axis((0, 32), "threadIdx.x")
#x,y = AA.op.axis
#s[AA].reorder(y,x)
#print(AA.op.axis[1])
#s[AA].bind(AA.op.axis[1],te.thread_axis("threadIdx.x"))
#s[B].bind(B.op.axis[0],te.thread_axis("threadIdx.x"))
#print(tvm.lower(s,[A,B],simple_mode = True))


def naive_approach():
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((m, n), name="A")
    B = te.compute((m, n), lambda i, j: A[j, i], name="B")
    s = te.create_schedule([B.op])
    s[B].bind(B.op.axis[0],te.thread_axis("blockIdx.x"))
    s[B].bind(B.op.axis[1],te.thread_axis("threadIdx.x"))
    naive = tvm.build(s,[A,B],target="cuda",name="transpose")
    return naive


def measure_performance(size,transpose_function,string):
    ctx = tvm.context('gpu')
    M = np.array([[i for i in range(size)] for i in range(size)], dtype = np.float32)
    A_1 = tvm.nd.array(M,ctx)
    B_1 = tvm.nd.array(np.ones((size,size),dtype=np.float32),ctx)
    s = time.time()
    for i in range(10):
        transpose_function(A_1,B_1)
        ctx.sync()
    e = time.time()
    #print(A_1)
    #print(B_1)
    print(string,(e-s)/10)

#F = naive_approach()
#measure_performance(32,F,"Naive Appraoch")

#F = get_shared_but_not_tiled()
#measure_performance(32,F,"Shared but not Tiled")

#F = tiled_but_not_shared()
#measure_performance(8192,F,"Tiled but not shared")

F = tiled_and_shared()
#measure_performance(8192,F,"Tiled Shared")

