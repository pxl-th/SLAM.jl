Base.@kwdef struct LucasKanade
    iterations::Int64 = 30
    window_size::Int64 = 9
    pyramid_levels::Int64 = 3
    eigenvalue_threshold::Float64 = 1e-4
    ϵ::Float64 = 1e-2
end

function optflow!(
    displacement, first_pyramid, second_pyramid, points, algorithm::LucasKanade,
)
    has_enough_layers =
        length(first_pyramid.layers) > algorithm.pyramid_levels &&
        length(second_pyramid.layers) > algorithm.pyramid_levels
    has_enough_layers || throw("Not enough layers in pyramids.")

    n_points = length(points)
    status = trues(n_points)
    n_good = n_points

    window = algorithm.window_size
    mode = BSpline(Linear())

    for level in (algorithm.pyramid_levels + 1):-1:1
        level_resolution = axes(first_pyramid.layers[level])
        # Interpolate layer to get sub-pixel precision.
        # We never go out-of-bound, so there is no need to extrapolate.
        # We can safely use `interpolate!` function (thus saving some memory), because
        # when doing linear interpolation, there is no modification to the input array.
        # See: https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/prefiltering.jl#L33
        interploated_layer = interpolate!(@inbounds(second_pyramid.layers[level]), mode)

        Threads.@threads for n in 1:n_points
            @inbounds status[n] || continue

            point = get_pyramid_coordinate(@inbounds(points[n]), level)
            offsets = get_offsets(point, point, window, level_resolution)
            grid = get_grid(point, offsets)

            G_inv, min_eigenvalue = compute_spatial_gradient(first_pyramid, grid, level)
            if min_eigenvalue < algorithm.eigenvalue_threshold
                @inbounds status[n] = false
                n_good -= 1
                continue
            end

            pyramid_contribution = SVector{2, Float64}(0.0, 0.0)
            for _ in 1:algorithm.iterations
                putative_flow = @inbounds displacement[n] + pyramid_contribution
                putative_correspondence = point + putative_flow
                if !lies_in(level_resolution, putative_correspondence)
                    @inbounds status[n] = false
                    n_good -= 1
                    break
                end

                new_offsets = get_offsets(
                    point, putative_correspondence, window, level_resolution)
                # Recalculate gradient only if the offset changes.
                if new_offsets != offsets
                    offsets = new_offsets
                    grid = get_grid(point, offsets)
                    G_inv, min_eigenvalue = compute_spatial_gradient(
                        first_pyramid, grid, level)
                    if min_eigenvalue < algorithm.eigenvalue_threshold
                        @inbounds status[n] = false
                        n_good -= 1
                        break
                    end
                else
                    offsets = new_offsets
                    grid = get_grid(point, offsets)
                end

                estimated_flow = compute_flow_vector(
                    putative_correspondence,
                    first_pyramid, interploated_layer, level,
                    grid, offsets, G_inv)

                # Epsilon termination criteria.
                abs(estimated_flow[1]) < algorithm.ϵ &&
                    abs(estimated_flow[2]) < algorithm.ϵ && break

                pyramid_contribution += estimated_flow
                in_bounds = lies_in(
                    level_resolution, putative_correspondence + estimated_flow)
                if !in_bounds
                    @inbounds status[n] = false
                    n_good -= 1
                    break
                end
            end
            @inbounds status[n] || continue
            @inbounds displacement[n] += pyramid_contribution
            level > 1 && (@inbounds displacement[n] *= 2.0;)
        end
    end

    displacement, status, n_good
end

function compute_partial_derivatives(Iy, Ix; kwargs...)
    Iyy = typeof(Iy)(undef, size(Iy))
    Ixx = typeof(Iy)(undef, size(Iy))
    Iyx = typeof(Iy)(undef, size(Iy))
    compute_partial_derivatives!(Iyy, Ixx, Iyx, Iy, Ix; kwargs...)
end

function compute_partial_derivatives!(
    Iyy, Ixx, Iyx, Iy, Ix;
    squared = typeof(Iy)(undef, size(Iy)),
    filtered = typeof(Iy)(undef, size(Iy)), σ = 4.0,
)
    kernel_factors = get_kernel(σ, 2)

    squared .= Iy .* Iy
    imfilter!(filtered, squared, kernel_factors)
    integral_image!(Iyy, filtered)

    squared .= Ix .* Ix
    imfilter!(filtered, squared, kernel_factors)
    integral_image!(Ixx, filtered)

    squared .= Iy .* Ix
    imfilter!(filtered, squared, kernel_factors)
    integral_image!(Iyx, filtered)

    Iyy, Ixx, Iyx
end

function integral_image!(integral, img)
    sd = coords_spatial(img)
    cumsum!(integral, img; dims=sd[1])
    for i ∈ 2:length(sd)
        cumsum!(integral, integral; dims=sd[i])
    end
    integral
end

function _compute_spatial_gradient(
    grid, Iyy_integral, Iyx_integral, Ixx_integral,
)
    sum_Iyy = boxdiff(Iyy_integral, grid[1], grid[2])
    sum_Ixx = boxdiff(Ixx_integral, grid[1], grid[2])
    sum_Iyx = boxdiff(Iyx_integral, grid[1], grid[2])
    SMatrix{2, 2, Float64}(sum_Iyy, sum_Iyx, sum_Iyx, sum_Ixx)
end

function compute_spatial_gradient(pyramid::LKPyramid, grid, level)
    G = _compute_spatial_gradient(
        grid, pyramid.Iyy[level], pyramid.Iyx[level], pyramid.Ixx[level])
    U, S, V = svd2x2(G)
    G_inv = pinv2x2(U, S, V)
    min_eigenvalue = min(S[1, 1], S[2, 2]) / prod(length.(grid))

    G_inv, min_eigenvalue
end

function prepare_linear_system(corresponding_point, A, Iy, Ix, offsets, B)
    P, Q = size(A)
    by, bx = 0.0, 0.0

    @inbounds for q in 1:Q, p in 1:P
        r = corresponding_point[1] + offsets[1][p]
        c = corresponding_point[2] + offsets[2][q]

        δI = A[p, q] - B(r, c)
        by += δI * Iy[p, q]
        bx += δI * Ix[p, q]
    end

    SVector{2, Float64}(by, bx)
end

function compute_flow_vector(
    corresponding_point,
    first_pyramid::LKPyramid, etp, level,
    grid, offsets, G_inv,
)
    b = prepare_linear_system(
        corresponding_point,
        view(first_pyramid.layers[level], grid[1], grid[2]),
        view(first_pyramid.Iy[level], grid[1], grid[2]),
        view(first_pyramid.Ix[level], grid[1], grid[2]),
        offsets, etp)
    G_inv * b
end

@inline lies_in(area, point) = @inbounds(
    first(area[1]) ≤ point[1] ≤ last(area[1]) &&
    first(area[2]) ≤ point[2] ≤ last(area[2]))

"""
# Arguments
- `level`: Level of the pyramid in `[1, levels]` range.
"""
@inline get_pyramid_coordinate(point, level) = floor.(Int, point ./ 2 ^ (level - 1))

function get_offsets(point, new_point, window, image_axes)
    rows, cols = image_axes

    up = floor(Int, min(window, min(point[1], new_point[1]) - first(rows)))
    down = floor(Int, min(window, last(rows) - max(point[1], new_point[1])))
    left = floor(Int, min(window, min(point[2], new_point[2]) - first(cols)))
    right = floor(Int, min(window, last(cols) - max(point[2], new_point[2])))

    (-up:down, -left:right)
end

@inline get_grid(point, offsets) = @inbounds (
    (point[1] + offsets[1][begin]):(point[1] + offsets[1][end]),
    (point[2] + offsets[2][begin]):(point[2] + offsets[2][end]))
