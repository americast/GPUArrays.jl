function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{IntTS²}) where {T, TS, TS², IntTS²}
    # Thread identifiers
    row = threadidx_x(state) # Local row ID (max: TS)
    col = threadidx_y(state)
    # col = row
    # Local memory to fit a tile of TS*TS elements of A and B
    Asub = @LocalMemory(state, T, TS²)
    Bsub = @LocalMemory(state, T, TS²)

    # Initialise the accumulation register
    acc = T(0.0)

    # Loop over all tiles
    numTiles = div(Asize[2], TS)
    for t in UInt32(1):UInt32(numTiles)
        # Perform the computation for a single tile
        # Commenting this inner loop doesn't give an error
        for k in UInt32(1):UInt32(TS)
            @inbounds acc += Asub[(k - 1)*TS + (row - 1 ) + 1] * Bsub[(col - 1) * TS + (k - 1) + 1]
        end
        # Synchronise before loading the next tile
        synchronize_threads(state)
    end

    # Store the final result in out
    @inbounds out[row] = acc

    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    TS = UInt32(32)
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    config = ((div(Asize[1], TS), div(Bsize[2], TS)), (TS, TS))
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{UInt32(32)}(), Val{UInt32(1024)}(), Val{1024}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
