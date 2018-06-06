@generated function ntuple_args(f, ::Val{N}, args::Vararg{<: Any, Nargs}) where {N, Nargs}
    expr = Expr(:tuple)
    for i = 1:N
        call = Expr(:call, :f, i)
        for j = 1:Nargs
            push!(call.args, :(args[$j]))
        end
        push!(expr.args, call)
    end
    quote
        Base.@_inline_meta
        $expr
    end
end

function matmul_kernel(state, A::AbstractArray{T}, B::AbstractArray{T}, out, Asize, Bsize, outSize, ::Val{TS}, ::Val{TS²}, ::Val{WPT}, ::Val{numTiles}, ::Val{RTS}) where {T, TS, TS², WPT, numTiles, RTS}
    # Thread identifiers
    row = threadidx_x(state) # Local row ID (max: TS)
    Asub = @LocalMemory(state, T, TS²)

    acc = ntuple(Val{WPT}) do i 
     0.0f0
    end

    for k in UInt32(1):UInt32(TS)
     acc = ntuple(Val{WPT}) do i
       @inbounds return (Asub[k])
     end
    end
    return

end


function matmul!(dest::GPUArray, a::GPUArray{T, 2}, b::GPUArray{T, 2}) where T
    Asize = size(a)
    Bsize = size(b)
    device = GPUArrays.device(a)
    thr = GPUArrays.threads(device)
    TS = ceil(Int,sqrt(thr))
    # print("TS: ", TS)
    WPT = 8
    outSize = UInt32.(size(dest))
    Asize = UInt32.(Asize)
    Bsize = UInt32.(Bsize)
    acc = zeros(typeof(a), WPT)
    config = (UInt32.((div(Asize[1], TS), div(Bsize[2], TS))), UInt32.((TS, div(TS, WPT))))
    # print("config: ", config)
    gpu_call(matmul_kernel, dest, (a,b, dest, Asize, Bsize, outSize, Val{TS}(), Val{TS^2}(), Val{WPT}(), Val{div(Asize[2], TS)}(), Val{div(TS, WPT)}()), config)
    dest
end
#
# A = JLArray(rand(10, 10))
# B = JLArray(rand(10, 10))
# out = JLArray(zeros(size(A, 1), size(B, 2)))
# matmul!(out, A, B)
