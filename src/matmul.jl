using CUDAnative

function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{IntTS²}) where {T, TS, TS², IntTS²}
    row = CUDAnative.threadIdx().x
    Asub = CUDAnative.@cuStaticSharedMem(T, TS²)
    acc = T(0.0)
    for t in UInt32(1):UInt32(div(Asize[2], TS))
        for k in UInt32(1):UInt32(TS)
            @inbounds acc += Asub[(k - 1)+ (row - 1 ) + 1] * Asub[(row - 1) * TS + (k - 1) + 1]
        end
        synchronize_threads(state)
    end
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
    print("div: $(div(Asize[2], TS))\n")
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{UInt32(32)}(), Val{UInt32(1024)}(), Val{1024}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
