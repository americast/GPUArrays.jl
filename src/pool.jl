function maxpool2d_kernel(state, A::AbstractArray{T}, out, Asize, pool, stride, outSize) where T
    ilin = linear_index(state)
    @show ilin
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    @show idx
    if (idx[1] > Asize[1] || idx[2] > Asize[2])
        return
    end
    temp_max = A[(idx[1] - 1) + Asize[1] * (idx[2] -1 ) + 1]
    max_pos = (idx[1] - 1) + Asize[1] * (idx[2] -1 ) + 1

    for p in 1:(pool - 1)
        m = A[(idx[1] - 1) + stride * p + Asize[1] * (idx[2] - 1) + 1]
        if (m > temp_max)
                temp_max = m
                max_pos = (idx[1] - 1) + stride * p + Asize[1] * (idx[2] - 1) + 1
        end
        out[(idx[1] - 1) + outSize[1] * (idx[2] - 1) + 1] = temp_max
    end
    return
end


function maxpool2d(a, pool; stride = 1)
    Asize = UInt32.(size(a))
    out = similar(a)
    out = out[1:(div(Asize[1] - pool, stride) + 1), 1:(div(Asize[2] - pool, stride) + 1)]
    outSize = UInt32.(size(out))
    gpu_call(maxpool2d_kernel, a, (a, out, Asize, pool, stride, outSize))
    GPUArrays.synchronize(out)
    out
end


