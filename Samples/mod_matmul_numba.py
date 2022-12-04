import numpy as np
from numba import cuda, types as numba_types

# Naive version
@cuda.jit
def naive_mul(A, B, out):
  i, j = cuda.grid(2)

  if i < out.shape[0] and j < out.shape[1]:
    temp = 0
    for k in range(A.shape[1]):
      temp += A[i, k] * B[k, j]
    out[i, j] = temp  




tpg = (32, 32) #threads per block


# Shared memory version
@cuda.jit
def shared_mul(A, B, out):
    # Define an array in the shared memory
    tile_A = cuda.shared.array(tpg, numba_types.float32)
    tile_B = cuda.shared.array(tpg, numba_types.float32)

    i, j = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.gridDim.x    # blocks per grid

    if i >= out.shape[0] and j >= out.shape[1]:
        return

    temp = 0.
    for k in range(bx):
        # Preload data into shared memory with coalesced access memory
        tile_A[ty, tx] = A[j, tx + k * tpg[0]]
        tile_B[ty, tx] = B[ty + k * tpg[0], i]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Compute block product on the shared memory
        for l in range(4):
            temp += tile_A[ty, l] * tile_B[l, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    # Write coalescing acess to global memory 
    out[j, i] = temp    
    
    
def numba_wrap_matmul(nb, tpb, A, B):
    out   = np.zeros_like(A).astype(np.float32)
    d_A   = cuda.to_device(A)
    d_B   = cuda.to_device(B)
    d_out = cuda.to_device(out)

    cuda.synchronize()
    shared_mul[nb,tpb](d_A, d_B, d_out)
    cuda.synchronize()
        
    out = d_out.copy_to_host()
    cuda.synchronize()
        
    return out
