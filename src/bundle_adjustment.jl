function bundle_adjustment!(
    cache, camera::Camera; iterations::Int = 10,
    show_trace::Bool = false, repr_ϵ::Real = 5.0,
)
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    n_observations = length(cache.observations)
    n_poses, n_points = length(cache.poses_remap), length(cache.points_remap)
    poses_shift = n_poses * 6

    ignore_outliers = false
    Y = zeros(Float64, n_observations * 2)

    function residue!(Y, X)
        poses = reshape(@view(X[1:poses_shift]), 6, n_poses)
        points = reshape(@view(X[(poses_shift + 1):end]), 3, n_points)

        @simd for i in 1:n_observations
            id = (i - 1) * 2
            if ignore_outliers && cache.outliers[i]
                Y[id + 1] = 0.0
                Y[id + 2] = 0.0
            else
                pt = @view(points[:, cache.points_ids[i]])
                T = @view(poses[:, cache.poses_ids[i]])
                pt = RotZYX(T[1:3]...) * pt .+ T[4:6] # (x, y, z) format.
                px = @view(cache.pixels[:, i]) # (y, x) format.

                inv_z = 1.0 / pt[3]
                Y[id + 1] = px[1] - (fy * pt[2] * inv_z + cy)
                Y[id + 2] = px[2] - (fx * pt[1] * inv_z + cx)
            end
        end
    end

    optimizer = LevenbergMarquardt(LeastSquaresOptim.LSMR())

    # Fast run, to detect outliers.
    θ0 = cache.θ
    sparsity, g! = _get_jacobian_sparsity(cache, residue!, ignore_outliers)

    Y1 = optimize!(
        LeastSquaresProblem(θ0, Y, residue!, sparsity, g!), optimizer;
        iterations=5, show_trace)
    θ1 = Y1.minimizer
    n_outliers = _ba_detect_outliers!(cache, θ1, camera; repr_ϵ)
    @debug "BA N Outliers $n_outliers."

    ignore_outliers = true
    sparsity2, g2! = _get_jacobian_sparsity(cache, residue!, ignore_outliers)

    Y2 = optimize!(
        LeastSquaresProblem(θ1, Y, residue!, sparsity2, g2!), optimizer;
        iterations, show_trace)
    copy!(cache.θ, Y2.minimizer)
end

function _get_jacobian_sparsity(cache, residue_f, ignore_outliers)
    n_observations = length(cache.observations)
    n_poses = length(cache.poses_remap)
    n_points = length(cache.points_remap)
    poses_shift = n_poses * 6
    n_parameters = poses_shift + n_points * 3

    sparsity = spzeros(Float64, n_observations * 2, n_parameters)
    @simd for i in 1:n_observations
        # If mappoint is outlier for that observation,
        # then leave zero Jacobian for it, e.g. no updates.
        if !(ignore_outliers && cache.outliers[i])
            id = 2 * (i - 1)

            # Set Jacobians for point's coordinates.
            pid = poses_shift + (cache.points_ids[i] - 1) * 3
            sparsity[(id+1):(id+2), (pid+1):(pid+3)] .= 1.0

            # Set Jacobians for poses if they are not constant.
            eid = cache.poses_ids[i]
            if !cache.θconst[eid]
                eid = (eid - 1) * 6
                sparsity[(id+1):(id+2), (eid+1):(eid+6)] .= 1.0
            end
        end
    end

    colorvec = matrix_colors(sparsity)
    grad! = (j, x) -> forwarddiff_color_jacobian!(
        j, residue_f, x; colorvec, sparsity)
    sparsity, grad!
end

function _ba_detect_outliers!(cache, θ, camera::Camera; repr_ϵ, depth_ϵ = 1e-6)
    n_observations = length(cache.observations)
    n_poses = length(cache.poses_remap)
    n_points = length(cache.points_remap)
    poses_shift = n_poses * 6

    poses = reshape(@view(θ[1:poses_shift]), 6, n_poses)
    points = reshape(@view(θ[(poses_shift + 1):end]), 3, n_points)

    n_outliers = 0
    @simd for i in 1:n_observations
        T = @view(poses[:, cache.poses_ids[i]])
        pt = @view(points[:, cache.points_ids[i]])
        pt = RotZYX(T[1:3]...) * pt .+ T[4:6]
        Δ = @view(cache.pixels[:, i]) .- project(camera, pt) # (y, x) format

        outlier = pt[3] < depth_ϵ || (Δ[1]^2 + Δ[2]^2) > repr_ϵ
        cache.outliers[i] = outlier
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
        @simd for i in 1:length(points)
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
        LevenbergMarquardt(); iterations=5, show_trace)
    X1 = fast_result.minimizer

    # Detect outliers using `fast_result` minimizer.
    n_outliers = 0
    @simd for i in 1:length(points)
        pt = RotZYX(@view(X1[1:3])...) * points[i] .+ @view(X1[4:6])
        r = pixels[i] .- project(camera, pt)
        outlier = pt[3] < depth_ϵ || (r[1]^2 + r[2]^2) > repr_ϵ
        outliers[i] = outlier
        outlier && (n_outliers += 1;)
    end

    if length(points) - n_outliers < 5
        return (
            SMatrix{4, 4, Float64}(I), initial_error, fast_result.ssr,
            outliers, n_outliers)
    end

    Y .= 0.0
    ignore_outliers = true
    result = optimize!(
        LeastSquaresProblem(;x=X1, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace)

    new_pose = to_4x4(RotZYX(result.minimizer[1:3]...), result.minimizer[4:6])
    new_pose, initial_error, result.ssr, outliers, n_outliers
end
