struct LKCache
    squared::Vector{Matrix{Gray{Float64}}}
    filtered::Vector{Matrix{Gray{Float64}}}
    gaussian_filtered::Vector{Matrix{Gray{Float64}}}
end

function Base.copy!(dst::LKCache, src::LKCache)
    for (dl, sl) in zip(dst.squared, src.squared) copy!(dl, sl) end
    for (dl, sl) in zip(dst.filtered, src.filtered) copy!(dl, sl) end
    for (dl, sl) in zip(dst.gaussian_filtered, src.gaussian_filtered)
        copy!(dl, sl)
    end
    dst
end

struct LKPyramid{G, C}
    layers::Vector{Matrix{Gray{Float64}}}
    Iy::G
    Ix::G
    Iyy::G
    Ixx::G
    Iyx::G
    cache::C
end
has_cache(lk::LKPyramid{G, C}) where {G, C} = C ≢ Nothing
has_gradients(lk::LKPyramid{G, C}) where {G, C} = G ≢ Nothing

function Base.copy!(dst::LKPyramid{G, C}, src::LKPyramid{G, C}) where {G, C}
    for (dl, sl) in zip(dst.layers, src.layers) copy!(dl, sl) end
    G ≡ Nothing && return dst

    for (dl, sl) in zip(dst.Iy, src.Iy) copy!(dl, sl) end
    for (dl, sl) in zip(dst.Ix, src.Ix) copy!(dl, sl) end
    for (dl, sl) in zip(dst.Iyy, src.Iyy) copy!(dl, sl) end
    for (dl, sl) in zip(dst.Ixx, src.Ixx) copy!(dl, sl) end
    for (dl, sl) in zip(dst.Iyx, src.Iyx) copy!(dl, sl) end
    dst
end

function LKPyramid(image, levels; downsample = 2, σ = 1.0, gradients = true, reusable = false)
    pyramid = gaussian_pyramid(image, levels, downsample, σ)
    gradients || return LKPyramid(pyramid, nothing, nothing, nothing, nothing, nothing, nothing)

    total_levels = levels + 1
    Iy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ix = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Iyy = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Ixx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
    Iyx = Vector{Matrix{Gray{Float64}}}(undef, total_levels)

    filling = Fill(zero(eltype(pyramid[1])))
    if reusable
        squared = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
        filtered = Vector{Matrix{Gray{Float64}}}(undef, total_levels)
        gaussian_filtered = Vector{Matrix{Gray{Float64}}}(undef, total_levels - 1)
        cache = LKCache(squared, filtered, gaussian_filtered)

        for (i, layer) in enumerate(pyramid)
            # TODO in-place imgradients
            Iy[i], Ix[i] = imgradients(layer, KernelFactors.scharr, filling)
            level_size, level_type = size(Iy[i]), typeof(Iy[i])

            squared[i] = level_type(undef, level_size)
            filtered[i] = level_type(undef, level_size)
            if i < total_levels
                gaussian_filtered[i] = level_type(undef, level_size)
            end

            Iyy[i], Ixx[i], Iyx[i] = compute_partial_derivatives(
                Iy[i], Ix[i]; squared=squared[i], filtered=filtered[i])
        end
        return LKPyramid(pyramid, Iy, Ix, Iyy, Ixx, Iyx, cache)
    end

    for (i, layer) in enumerate(pyramid)
        Iy[i], Ix[i] = imgradients(layer, KernelFactors.scharr, filling)
        Iyy[i], Ixx[i], Iyx[i] = compute_partial_derivatives(Iy[i], Ix[i])
    end
    LKPyramid(pyramid, Iy, Ix, Iyy, Ixx, Iyx, nothing)
end

function update!(lk::LKPyramid{G, C}, img; σ = 1.0) where {G <: AbstractVector, C}
    filling = Fill(zero(eltype(lk.layers[begin])))
    gaussian_pyramid!(lk, img, σ)
    @inbounds for (i, layer) in enumerate(lk.layers)
        lk.Iy[i], lk.Ix[i] = imgradients(layer, KernelFactors.scharr, filling)
        if C ≡ Nothing
            compute_partial_derivatives!(
                lk.Iyy[i], lk.Ixx[i], lk.Iyx[i], lk.Iy[i], lk.Ix[i])
        else
            compute_partial_derivatives!(
                lk.Iyy[i], lk.Ixx[i], lk.Iyx[i], lk.Iy[i], lk.Ix[i];
                squared=lk.cache.squared[i], filtered=lk.cache.filtered[i])
        end
    end
    lk
end
update!(lk::LKPyramid{Nothing, Nothing}, img; σ = 1.0) = gaussian_pyramid!(lk.layers, img, σ)

@inline function get_kernel(σ, N)
    kerng = KernelFactors.IIRGaussian(σ)
    ntuple(_ -> kerng, Val(N))
end

gaussian_pyramid!(lk::LKPyramid{G, Nothing}, img::AbstractArray{T, N}, σ) where {G, T, N} =
    gaussian_pyramid!(lk.layers, img, get_kernel(σ, N))
gaussian_pyramid!(lk::LKPyramid{G, LKCache}, img::AbstractArray{T, N}, σ) where {G, T, N} =
    gaussian_pyramid!(lk.layers, lk.cache.gaussian_filtered, img, get_kernel(σ, N))

function gaussian_pyramid!(pyramid::P, img, kernel) where P <: AbstractVector
    scale = 2
    @inbounds copy!(pyramid[1], img)
    @inbounds for _ in 1:(length(pyramid) - 1)
        tmp = imfilter(pyramid[scale - 1], kernel, NA())
        ImageTransformations.imresize!(
            pyramid[scale], interpolate!(tmp, BSpline(Linear())))
        scale += 1
    end
    pyramid
end
function gaussian_pyramid!(pyramid::P, filtered, img, kernel) where P <: AbstractVector
    scale = 2
    @inbounds copy!(pyramid[1], img)
    @inbounds for _ in 1:(length(pyramid) - 1)
        tmp = filtered[scale - 1]
        imfilter!(tmp, pyramid[scale - 1], kernel, NA())
        ImageTransformations.imresize!(
            pyramid[scale], interpolate!(tmp, BSpline(Linear())))
        scale += 1
    end
    pyramid
end
