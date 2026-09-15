// MIT License; copyright AMD. See ../main.hip for the full license.
#pragma once
template<unsigned int BlockSize>
__global__ void matrix_multiplication_kernel(const float*       A,
                                             const float*       B,
                                             float*             C,
                                             const unsigned int a_cols)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    // b_cols must match the number of output matrix columns.
    const unsigned int b_cols = blockDim.x * gridDim.x;

    // The number of tiles is determined by A's columns (which is equal to B's rows).
    const unsigned int steps = a_cols / BlockSize;

    // thread_result is the accumulation variable.
    float thread_result = 0.0F;
    for(unsigned int step = 0; step < steps; step++)
    {
        // Shared memory is used to cache the tile from both input matrices.
        // The tile is a square of BlockSize*BlockSize.
        __shared__ float a_values[BlockSize][BlockSize];
        __shared__ float b_values[BlockSize][BlockSize];

        // Index of the top-left element of the tile in A.
        // "BlockSize * a_cols * by" is the number of elements to move "down".
        // "BlockSize * step" is the number of elements to move "right".
        const unsigned int a_idx = BlockSize * (a_cols * by + step);

        // Index of the top-left element of the tile in B.
        // "BlockSize * b_cols * step" is the number of elements to move "down".
        // "BlockSize * bx" is the number of elements to move "right".
        const unsigned int b_idx = BlockSize * (b_cols * step + bx);

        // Load each element in the tile to shared memory.
        a_values[ty][tx] = A[a_idx + a_cols * ty + tx];
        b_values[ty][tx] = B[b_idx + b_cols * ty + tx];

        // Synchronization is needed to make sure that all elements are loaded before
        // starting the calculation.
        __syncthreads();

        // Each thread calculates the scalar product of the tile and increments the
        // thread-individual thread_result.
        for(unsigned int i = 0; i < BlockSize; i++)
        {
            thread_result += a_values[ty][i] * b_values[i][tx];
        }

        // Synchronize to ensure that the calculation is finished before the next tile's
        // elements start to load.
        __syncthreads();
    }

    // Calculate the index of the top-left element of the output block.
    const unsigned block_offset = b_cols * BlockSize * by + BlockSize * bx;

    // Every thread stores the final result to global memory.
    C[block_offset + b_cols * ty + tx] = thread_result;
}
