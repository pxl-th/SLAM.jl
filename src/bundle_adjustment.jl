"""
Perform Bundle-Adjustment.

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
- Outliers array, which shows for each observation,
  whether the observation is and outlier.
"""
function bundle_adjustment(
    camera::Camera,
    extrinsics::Matrix{Float64},
    points::Matrix{Float64},
    pixels::Matrix{Float64},
    points_ids, extrinsic_ids;
    constant_extrinsics::Union{Nothing, Vector{Bool}} = nothing,
    iterations = 10, show_trace::Bool = false, repr_ϵ = 5.0,
)
    check_constants = constant_extrinsics ≢ nothing
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    n_observations = size(pixels, 2)
    n_poses = size(extrinsics, 2)
    n_points = size(points, 2)
    poses_shift = n_poses * 6
    n_parameters = poses_shift + n_points * 3

    # Outlier is a mappoint that has either a negative depth in camera view
    # or big reprojection error.
    ignore_outliers = false
    outliers = fill(false, n_observations)
    X0 = vcat(
        reshape(extrinsics, length(extrinsics)),
        reshape(points, length(points)),
    )
    Y = zeros(Float64, n_observations * 2)
    dx = zeros(Float64, n_observations * 2)

    function residue!(Y, X)
        ext = reshape(@view(X[1:poses_shift]), 6, n_poses)
        pts = reshape(@view(X[(poses_shift + 1):end]), 3, n_points)
        @inbounds @simd for i in 1:n_observations
            id = (i - 1) * 2
            if ignore_outliers && outliers[i]
                Y[id + 1] = 0.0
                Y[id + 2] = 0.0
            else
                pt = @view(pts[:, points_ids[i]])
                T = @view(ext[:, extrinsic_ids[i]])

                pt = RotZYX(T[1:3]...) * pt .+ T[4:6] # (x, y, z) format.
                px = @view(pixels[:, i]) # (y, x) format.

                inv_z = 1.0 / pt[3]
                Y[id + 1] = px[1] - (fy * pt[2] * inv_z + cy)
                Y[id + 2] = px[2] - (fx * pt[1] * inv_z + cx)
            end
        end
    end

    optimizer = LevenbergMarquardt(LeastSquaresOptim.LSMR())

    # Fast run, to detect outliers.
    sparsity, g! = _get_jacobian_sparsity(
        X0, dx, residue!,
        points_ids, extrinsic_ids, constant_extrinsics,
        outliers, ignore_outliers,
        n_observations, n_parameters, poses_shift,
    )
    outliers_result = optimize!(
        LeastSquaresProblem(X0, Y, residue!, sparsity, g!), optimizer;
        iterations=5, show_trace,
    )
    X1 = outliers_result.minimizer

    n_outliers = _ba_detect_outliers!(
        outliers,
        reshape(@view(X1[1:poses_shift]), 6, n_poses),
        reshape(@view(X1[(poses_shift + 1):end]), 3, n_points),
        pixels, extrinsic_ids, points_ids, camera, n_observations; repr_ϵ,
    )
    @info "BA N Outliers $n_outliers."

    # Recompute Jacobian sparsity ignoring outliers and perform full run.
    ignore_outliers = true
    sparsity, new_g! = _get_jacobian_sparsity(
        X0, dx, residue!,
        points_ids, extrinsic_ids, constant_extrinsics,
        outliers, ignore_outliers,
        n_observations, n_parameters, poses_shift,
    )
    result = optimize!(
        LeastSquaresProblem(X1, Y, residue!, sparsity, new_g!), optimizer;
        iterations, show_trace,
    )
    X2 = result.minimizer

    extrinsics = reshape(@view(X2[1:poses_shift]), 6, n_poses)
    points = reshape(@view(X2[(poses_shift + 1):end]), 3, n_points)
    extrinsics, points, result.ssr, outliers
end

function _get_jacobian_sparsity(
    X0, dx, residue_f,
    points_ids, extrinsic_ids, constant_extrinsics,
    outliers, ignore_outliers::Bool,
    n_observations, n_parameters, poses_shift,
)
    check_constants = constant_extrinsics ≢ nothing
    sparsity = spzeros(Float64, n_observations * 2, n_parameters)

    @inbounds @simd for i in 1:n_observations
        # If mappoint is outlier for that observation,
        # then leave zero Jacobian for it, e.g. no updates.
        if !(ignore_outliers && outliers[i])
            id = 2 * (i - 1)

            # Set Jacobians for point's coordinates.
            pid = poses_shift + (points_ids[i] - 1) * 3
            sparsity[(id+1):(id+2), (pid+1):(pid+3)] .= 1.0

            # Set Jacobians for extrinsics if they are not constant.
            if !(check_constants && constant_extrinsics[extrinsic_ids[i]])
                eid = (extrinsic_ids[i] - 1) * 6
                sparsity[(id+1):(id+2), (eid+1):(eid+6)] .= 1.0
            end
        end
    end

    colorvec = matrix_colors(sparsity)
    cache = ForwardColorJacCache(residue_f, X0; dx, colorvec, sparsity)
    grad! = (j, x) -> forwarddiff_color_jacobian!(j, residue_f, x, cache)
    sparsity, grad!
end

function _ba_detect_outliers!(
    outliers,
    extrinsics, points, pixels,
    extrinsic_ids, points_ids, camera::Camera, n_observations::Int64;
    repr_ϵ::Float64, depth_ϵ::Float64 = 1e-6,
)
    n_outliers = 0
    @inbounds @simd for i in 1:n_observations
        pt = @view(points[:, points_ids[i]])
        T = @view(extrinsics[:, extrinsic_ids[i]])
        pt = RotZYX(T[1:3]...) * pt .+ T[4:6]
        r = @view(pixels[:, i]) .- project(camera, pt) # (y, x) format

        outlier = pt[3] < depth_ϵ || (r[1]^2 + r[2]^2) > repr_ϵ
        outliers[i] = outlier
        outlier && (n_outliers += 1;)
    end
    n_outliers
end

function pnp_bundle_adjustment(
    camera::Camera, pose::SMatrix{4, 4, Float64}, pixels, points;
    iterations::Int = 10, show_trace::Bool = false,
    depth_ϵ::Real = 1e-6, repr_ϵ::Real = 5.0,
)
    R = RotZYX(pose[1:3, 1:3])
    X0 = [R.theta1, R.theta2, R.theta3, pose[1:3, 4]...]

    Y = zeros(Float64, length(pixels) * 2)
    outliers = fill(false, length(points))
    ignore_outliers = false

    function residue!(Y, X)
        @inbounds @simd for i in 1:length(points)
            id = (i - 1) * 2
            if ignore_outliers && outliers[i]
                Y[id + 1] = 0.0
                Y[id + 2] = 0.0
            else
                pt = RotZYX(@view(X[1:3])...) * points[i] .+ @view(X[4:6])
                Y[(id + 1):(id + 2)] .= pixels[i] .- project(camera, pt)
            end
        end
    end

    residue!(Y, X0)
    initial_error = mapreduce(abs2, +, Y)
    Y .= 0.0

    fast_result = optimize!(
        LeastSquaresProblem(;x=X0, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace,
    )
    X1 = fast_result.minimizer

    # Detect outliers using `fast_result` minimizer.
    n_outliers = 0
    @inbounds @simd for i in 1:length(points)
        pt = RotZYX(@view(X1[1:3])...) * points[i] .+ @view(X1[4:6])
        r = pixels[i] .- project(camera, pt)
        outlier = pt[3] < depth_ϵ || (r[1]^2 + r[2]^2) > repr_ϵ
        outliers[i] = outlier
        outlier && (n_outliers += 1;)
    end

    if length(points) - n_outliers < 5
        return (
            SMatrix{4, 4, Float64}(I), initial_error, fast_result.ssr,
            outliers, n_outliers,
        )
    end

    Y .= 0.0
    ignore_outliers = true
    result = optimize!(
        LeastSquaresProblem(;x=X1, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace,
    )

    new_pose = to_4x4(RotZYX(result.minimizer[1:3]...), result.minimizer[4:6])
    new_pose, initial_error, result.ssr, outliers, n_outliers
end
