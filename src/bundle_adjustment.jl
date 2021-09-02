"""
Perform full Bundle-Adjustment.

Minimize error function over camera extrinsics and points positions.
Leaving camera's intrinsics and pixel coordinates intact.

Function also computes Jacobian sparsity structure,
since the size of the Jacobian is `N×P`,
where `N = 2 * NPoints` and `P = NCameras * 9 + NPoints * 3`.
However only small amount of elements are non-zero.

# Arguments:

- `pixels::Matrix{Float64}`:
    Target pixel coordinates of the `2×N` in `(y, x)` format.
- `extrinsics::Matrix{Float64}`:
    Matrix of extrinsics of the `P×N` size.
    `P=6` is the number of parameters for each KeyFrame
    and `N` is the number of KeyFrames.
    **Note**, these are the parameters that will be optimized.
    Each extrinsic is parametrized by: `(αx, αy, αz, tx, ty, tz)`,
    3 Euler ZYX angles and XYZ translation vector.
- `points_ids::Vector{Int}`:
    Vector of indices, where i-th element specifies id of a 3D point
    in `points` array, that corresponds to the i-th pixel.
    E.g. `pixels[i] ↔ points[points_ids[i]]`.
- `extrinsic_ids::Vector{Int}`:
    Vector of indices, where i-th element specifies id of an extrinsic
    in `extrinsics` array, that corresponds to the i-th pixel.
    E.g. `pixels[i] ↔ extrinsics[extrinsic_ids[i]]`.

# Returns:

- New extrinsic parameters.
- New points coordinates.
- Error after minimization.
"""
function full_bundle_adjustment(
    camera::Camera,
    extrinsics::Matrix{Float64},
    points::Matrix{Float64},
    pixels::Matrix{Float64},
    points_ids, extrinsic_ids;
)
    n_observations = size(pixels, 2)
    n_poses = size(extrinsics, 2)
    n_points = size(points, 2)
    poses_shift = n_poses * 6
    n_parameters = poses_shift + n_points * 3

    X0 = vcat(
        reshape(extrinsics, length(extrinsics)),
        reshape(points, length(points)),
    )
    Y = zeros(Float64, n_observations * 2)

    function residue!(Y, X)
        ext = reshape(@view(X[1:poses_shift]), 6, n_poses)
        pts = reshape(@view(X[(poses_shift + 1):end]), 3, n_points)
        for i in 1:n_observations
            pt = @view(pts[:, points_ids[i]])
            T = @view(ext[:, extrinsic_ids[i]])
            pt = RotZYX(T[1:3]...) * pt .+ T[4:6]
            id = (i - 1) * 2
            Y[(id + 1):(id + 2)] .= pixels[i] .- project(camera, pt)
        end
    end

    J_sparsity = spzeros(Float64, n_observations * 2, n_parameters)
    for i in 1:n_observations
        id = 2 * (i - 1)
        eid = (extrinsic_ids[i] - 1) * 6
        pid = poses_shift + (points_ids[i] - 1) * 3
        for j in 1:6
            J_sparsity[id + 1, eid + j] = 1.0
            J_sparsity[id + 2, eid + j] = 1.0
        end
        for j in 1:3
            J_sparsity[id + 1, pid + j] = 1.0
            J_sparsity[id + 2, pid + j] = 1.0
        end
    end

    colors = matrix_colors(J_sparsity)
    g! = (J, x) -> forwarddiff_color_jacobian!(J, residue!, x; colorvec=colors)
    result = optimize!(
        LeastSquaresProblem(X0, Y, residue!, J_sparsity, g!),
        LevenbergMarquardt(LeastSquaresOptim.LSMR());
        iterations, show_trace,
    )

    new_extrinsics = reshape(
        @view(result.minimizer[1:poses_shift]), 6, n_poses,
    )
    new_points = reshape(
        @view(result.minimizer[(poses_shift + 1):end]), 3, n_points,
    )
    new_extrinsics, new_points, result.ssr
end

function pnp_bundle_adjustment(
    camera::Camera, pose::SMatrix{4, 4, Float64},
    pixels::Vector{Point2f}, points::Vector{Point3f};
    iterations::Int = 10, show_trace::Bool = false,
)
    R = RotZYX(pose[1:3, 1:3])
    X0 = [R.theta1, R.theta2, R.theta3, pose[1:3, 4]...]
    Y = zeros(Float64, length(pixels) * 2)

    function residue!(Y, X)
        for i in 1:length(points)
            id = (i - 1) * 2
            pt = RotZYX(@view(X[1:3])...) * points[i] .+ @view(X[4:6])
            Y[(id + 1):(id + 2)] .= pixels[i] .- project(camera, pt)
        end
    end

    residue!(Y, X0)
    initial_error = mapreduce(abs2, +, Y)
    Y .= 0.0

    result = optimize!(
        LeastSquaresProblem(;x=X0, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace,
    )

    new_pose = to_4x4(RotZYX(result.minimizer[1:3]...), result.minimizer[4:6])
    new_pose, initial_error, result.ssr
end

function look_at(position, target, up)
    z_axis = (position - target) |> normalize
    x_axis = (up × z_axis) |> normalize
    y_axis = (z_axis × x_axis) |> normalize

    SMatrix{4, 4, Float64}(
        x_axis[1], y_axis[1], z_axis[1], 0,
        x_axis[2], y_axis[2], z_axis[2], 0,
        x_axis[3], y_axis[3], z_axis[3], 0,
        0, 0, 0, 1,
    ) * SMatrix{4, 4, Float64}(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -position..., 1,
    )
end

function rot_at(position, target, up)
    z_axis = (position - target) |> normalize
    x_axis = (up × z_axis) |> normalize
    y_axis = (z_axis × x_axis) |> normalize

    SMatrix{4, 4, Float64}(
        x_axis[1], y_axis[1], z_axis[1], 0,
        x_axis[2], y_axis[2], z_axis[2], 0,
        x_axis[3], y_axis[3], z_axis[3], 0,
        0, 0, 0, 1,
    )
end

function fibbonachi_sphere(samples::Int)
    points = Vector{GLMakie.Point3f}(undef, samples)
    ϕ = π * (3.0 - √5)
    for i in 1:samples
        y = 1.0 - ((i - 1) / (samples - 1)) * 2
        r = √(1 - y^2)
        θ = ϕ * i

        x = cos(θ) * r
        z = sin(θ) * r
        points[i] = Point3f(x, y, z)
    end
    points
end

function ba_main()
    f = 910
    resolution = 1024
    pp = resolution ÷ 2
    camera = SLAM.Camera(
        f, f, pp, pp,
        0, 0, 0, 0,
        resolution, resolution,
    )
    mp = backproject(camera, SLAM.Point2(512, 512))

    samples = 5
    positions = fibbonachi_sphere(samples)[2:end-1]
    cw = SMatrix{4, 4, Float64}[] # world → camera
    directions = GLMakie.Vec3f[]

    target = Point3f(0, 0, 0)
    up = Point3f(0, 1, 0)
    forward = Point4f(0, 0, -1, 1)
    for position in positions
        push!(cw, inv(SE3, look_at(position, target, up)))

        d = inv(SE3, rot_at(position, target, up)) * forward
        d = normalize(d[1:3] ./ d[4])
        push!(directions, d)
    end

    figure = Figure(;resolution=(800, 800))
    axis = Axis3(figure[1, 1])

    meshscatter!(axis, positions; markersize=0.01, color=:black)
    meshscatter!(axis, [target]; markersize=0.01, color=:red)
    arrows!(
        axis, positions, directions;
        lengthscale = 0.1, arrowsize=0.05, linewidth=0.01,
        color=:black, quality=4,
    )
end

function p3p_ba_main()
    n_points = 10
    f = 910
    resolution = 1024
    pp = resolution ÷ 2
    camera = SLAM.Camera(
        f, f, pp, pp,
        0, 0, 0, 0,
        resolution, resolution,
    )

    θ = π / 8
    R = RotXYZ(rand() * θ, rand() * θ, rand() * θ)
    t = SVector{3, Float64}(rand(), rand() * 2, rand() * 3)
    P_target = inv(SE3, to_4x4(R, t))

    pmin = SVector{2}(1, 1)
    pmax = SVector{2}(resolution, resolution)
    δ = pmax - pmin

    pixels_yx = [floor.(rand(SVector{2, Float64}) .* δ .+ pmin) for i in 1:n_points]
    pixels_xy = [SVector{2, Float64}(p[2], p[1]) for p in pixels_yx]
    points = [R * backproject(camera, p) .+ t for p in pixels_yx]

    n_inliers, (KP, inliers, error) = p3p_ransac(points, pixels_xy, camera.K)
    P = to_4x4(camera.iK * KP)

    new_P, initial_error, resulting_error = pnp_bundle_adjustment(
        camera, P, pixels_yx, points,
    )
    @show initial_error, resulting_error
    display(P_target); println()
    display(P); println()
    display(new_P); println()
end
