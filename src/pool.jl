function maxpool2d_kernel(state, A::AbstractArray{T}, out, Asize, pool, stride, outSize) where T
    ilin = linear_index(state)
    idx = GPUArrays.gpu_ind2sub(Asize, ilin)
    if (idx[1] >= Asize[1] - Ksize[1] || idx[2] >= Asize[2] - Ksize[2])
        return
    end
    temp_max = input[idx[1] + Asize[1] * idx[2]]
    max_pos = idx[1] + Asize[1] * idx[2]

    for p in 1:pool
        m = input[idx[1] + stride * p + Asize[1] * idx[2]]
        if (m > max)
                temp_max = m
                max_pos = idx[1] + stride * p + Asize[1] * idx[2]
        end
        out[idx[1] + outSize[1] * idx[2]] = temp_max
    end
    return
end


function maxpool2d(a, pool; stride = 1)
    Asize = UInt32.(size(a))
    out = similar(a)
    out = out[1:(div(Asize[1] - pool[1], stride) + 1), 1:(div(Asize[2] - pool[2], stride) + 1)]
    outSize = UInt32.(size(out))
    gpu_call(maxpool2d_kernel, a, (a, out, Asize, pool, stride, outSize))
    GPUArrays.synchronize(out)
    out
end

# struct FFTKernel{T}
#     kernel::T
#     irfftplan
#     rfftplan
# end

# function fftkernel(A, kernel)
#     plan_rfft!(A)

# end
