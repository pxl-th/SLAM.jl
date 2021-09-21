using SparseArrays
using SparseDiffTools
using LeastSquaresOptim
using Rotations
using BenchmarkTools
using FiniteDiff

function project(camera_params, point)
    projection = RotationVec(camera_params[1:3]...) * point
    projection += camera_params[4:6]
    projection = -projection[1:2] ./ projection[3]

    f, k1, k2 = camera_params[7:9]
    n = sum(projection.^2)
    r = 1 + k1 * n + k2 * n^2

    projection .* (f * r)
end

function parse_file(ba_file)
    io = open(ba_file, "r")
    n_cameras, n_points, n_observations = map(i -> parse(Int, i), split(readline(io)))

    camera_ids = Vector{Int64}(undef, n_observations)
    point_ids = Vector{Int64}(undef, n_observations)
    pixels = Matrix{Float64}(undef, 2, n_observations)
    points = Matrix{Float64}(undef, 3, n_observations)
    camera_params = Matrix{Float64}(undef, 9, n_cameras)

    for i in 1:n_observations
        camera_idx, point_idx, x, y = split(readline(io))
        camera_ids[i] = parse(Int64, camera_idx) + 1
        point_ids[i] = parse(Int64, point_idx) + 1
        pixels[1, i] = parse(Float64, x)
        pixels[2, i] = parse(Float64, y)
    end
    for i in 1:n_cameras, j in 1:9
        camera_params[j, i] = parse(Float64, readline(io))
    end
    for i in 1:n_points, j in 1:3
        points[j, i] = parse(Float64, readline(io))
    end

    close(io)
    camera_params, points, pixels, point_ids, camera_ids
end

function main(ba_file)
    camera_params, points, pixels, point_ids, camera_ids = parse_file(ba_file)

    n_observations = size(pixels, 2)
    n_cameras = size(camera_params, 2)
    n_points = size(points, 2)
    n_parameters = n_cameras * 9 + n_points * 3

    X0 = vcat(
        reshape(camera_params, length(camera_params)),
        reshape(points, length(points)),
    )
    Y = zeros(Float64, n_observations * 2)
    dx = zeros(Float64, n_observations * 2)

    function residue!(Y, X)
        cp = reshape(X[1:(n_cameras * 9)], 9, n_cameras)
        pts = reshape(X[(n_cameras * 9 + 1):end], 3, n_points)
        for i in 1:n_observations
            projection = project(
                @view(cp[:, camera_ids[i]]),
                @view(pts[:, point_ids[i]]),
            )
            id = (i - 1) * 2
            Y[(id + 1):(id + 2)] .= @view(pixels[:, i]) .- projection
        end
    end

    sparsity = spzeros(Float64, n_observations * 2, n_parameters)
    for i in 1:n_observations
        id = 2 * (i - 1)
        for j in 1:9
            sparsity[id + 1, (camera_ids[i] - 1) * 9 + j] = 1.0
            sparsity[id + 2, (camera_ids[i] - 1) * 9 + j] = 1.0
        end
        for j in 1:3
            sparsity[id + 1, n_cameras * 9 + (point_ids[i] - 1) * 3 + j] = 1.0
            sparsity[id + 2, n_cameras * 9 + (point_ids[i] - 1) * 3 + j] = 1.0
        end
    end

    colorvec = matrix_colors(sparsity)
    # g! = (j, x) -> forwarddiff_color_jacobian!(
    #     j, residue!, x; colorvec, sparsity,
    # )

    cache = ForwardColorJacCache(residue!, X0; dx, colorvec, sparsity)
    g! = (j, x) -> forwarddiff_color_jacobian!(j, residue!, x, cache)

    t1 = time()
    result = optimize!(
        LeastSquaresProblem(X0, Y, residue!, sparsity, g!),
        LevenbergMarquardt(LeastSquaresOptim.LSMR());
        iterations=15, show_trace=true,
    )
    t2 = time()
    @show t2 - t1

    @show result.x_converged
    @show result.f_converged
    @show result.iterations
    @show result.ssr
end
main("/home/pxl-th/Downloads/problem-21-11315-pre.txt")
