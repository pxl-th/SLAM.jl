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
    iterations::Int = 10, show_trace::Bool = false,
    depth_ϵ::Real = 1e-6, repr_ϵ::Real = 5.0,
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

    function residue!(Y, X)
        ext = reshape(@view(X[1:poses_shift]), 6, n_poses)
        pts = reshape(@view(X[(poses_shift + 1):end]), 3, n_points)
        @inbounds for i in 1:n_observations
            id = (i - 1) * 2

            if ignore_outliers && outliers[i]
                Y[id + 1] = 0.0
                Y[id + 2] = 0.0
                continue
            end

            pt = @view(pts[:, points_ids[i]])
            T = @view(ext[:, extrinsic_ids[i]])
            pt = RotZYX(T[1:3]...) * pt .+ T[4:6]

            px = @view(pixels[:, i]) # (y, x) format
            inv_z = 1.0 / pt[3]
            Y[id + 1] = px[1] - (fy * pt[2] * inv_z + cy)
            Y[id + 2] = px[2] - (fx * pt[1] * inv_z + cx)
        end
    end

    function compute_jacobian_sparsity!()
        J_sparsity = spzeros(Float64, n_observations * 2, n_parameters)
        @inbounds for i in 1:n_observations
            # If mappoint is outlier for that observation,
            # then leave zero Jacobian for it, e.g. no updates.
            ignore_outliers && outliers[i] && continue
            id = 2 * (i - 1)
            # Set Jacobians for point's coordinates.
            pid = poses_shift + (points_ids[i] - 1) * 3
            for j in 1:3
                J_sparsity[id + 1, pid + j] = 1.0
                J_sparsity[id + 2, pid + j] = 1.0
            end
            # Set Jacobians for extrinsics if they are not constant.
            check_constants && constant_extrinsics[extrinsic_ids[i]] && continue
            eid = (extrinsic_ids[i] - 1) * 6
            for j in 1:6
                J_sparsity[id + 1, eid + j] = 1.0
                J_sparsity[id + 2, eid + j] = 1.0
            end
        end

        colorvec = matrix_colors(J_sparsity)
        grad! = (j, x) -> forwarddiff_color_jacobian!(j, residue!, x; colorvec)
        J_sparsity, grad!
    end

    residue!(Y, X0)
    initial_error = mapreduce(abs2, +, Y)
    Y .= 0.0

    # Fast run, to detect outliers.
    J_sparsity, g! = compute_jacobian_sparsity!()
    outliers_result = optimize!(
        LeastSquaresProblem(X0, Y, residue!, J_sparsity, g!),
        LevenbergMarquardt(LeastSquaresOptim.LSMR());
        iterations=5, show_trace,
    )
    outliers_minimizer = outliers_result.minimizer

    # Detect outliers using `outliers_minimizer` result.
    n_outliers = 0
    ext = reshape(@view(outliers_minimizer[1:poses_shift]), 6, n_poses)
    pts = reshape(@view(outliers_minimizer[(poses_shift + 1):end]), 3, n_points)
    for i in 1:n_observations
        pt = @view(pts[:, points_ids[i]])
        T = @view(ext[:, extrinsic_ids[i]])
        pt = RotZYX(T[1:3]...) * pt .+ T[4:6]
        r = @view(pixels[:, i]) .- project(camera, pt) # (y, x) format

        outlier = pt[3] < depth_ϵ || (r[1]^2 + r[2]^2) > repr_ϵ
        outliers[i] = outlier
        outlier && (n_outliers += 1;)
    end

    # Recompute Jacobian sparsity ignoring outliers and perform full run.
    Y .= 0.0
    ignore_outliers = true
    J_sparsity, g! = compute_jacobian_sparsity!()
    result = optimize!(
        LeastSquaresProblem(outliers_minimizer, Y, residue!, J_sparsity, g!),
        LevenbergMarquardt(LeastSquaresOptim.LSMR());
        iterations, show_trace,
    )

    new_extrinsics = reshape(
        @view(result.minimizer[1:poses_shift]), 6, n_poses,
    )
    new_points = reshape(
        @view(result.minimizer[(poses_shift + 1):end]), 3, n_points,
    )
    new_extrinsics, new_points, initial_error, result.ssr, outliers
end

function pnp_bundle_adjustment(
    camera::Camera, pose::SMatrix{4, 4, Float64},
    pixels::Vector{Point2f}, points::Vector{Point3f};
    iterations::Int = 10, show_trace::Bool = false,
    depth_ϵ::Real = 1e-6, repr_ϵ::Real = 5.0,
)
    R = RotZYX(pose[1:3, 1:3])
    X0 = [R.theta1, R.theta2, R.theta3, pose[1:3, 4]...]

    Y = zeros(Float64, length(pixels) * 2)
    outliers = fill(false, length(points))
    ignore_outliers = false

    function residue!(Y, X)
        @inbounds for i in 1:length(points)
            id = (i - 1) * 2
            if ignore_outliers && outliers[i]
                Y[id + 1] = 0.0
                Y[id + 2] = 0.0
                continue
            end

            pt = RotZYX(@view(X[1:3])...) * points[i] .+ @view(X[4:6])
            Y[(id + 1):(id + 2)] .= pixels[i] .- project(camera, pt)
        end
    end

    residue!(Y, X0)
    initial_error = mapreduce(abs2, +, Y)
    Y .= 0.0

    fast_result = optimize!(
        LeastSquaresProblem(;x=X0, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace,
    )
    X0 = fast_result.minimizer

    # Detect outliers using `fast_result`.
    n_outliers = 0
    for i in 1:length(points)
        pt = RotZYX(@view(X0[1:3])...) * points[i] .+ @view(X0[4:6])
        r = pixels[i] .- project(camera, pt)
        outlier = pt[3] < depth_ϵ || (r[1]^2 + r[2]^2) > repr_ϵ
        outliers[i] = outlier
        outlier && (n_outliers += 1;)
    end

    @info "PNP BA N Outliers $n_outliers / $(length(points))."
    if length(points) - n_outliers < 5
        return (
            SMatrix{4, 4, Float64}(I), initial_error, fast_result.ssr,
            outliers, n_outliers,
        )
    end

    Y .= 0.0
    ignore_outliers = true
    result = optimize!(
        LeastSquaresProblem(;x=fast_result.minimizer, y=Y, f! = residue!),
        LevenbergMarquardt(); iterations, show_trace,
    )

    new_pose = to_4x4(RotZYX(result.minimizer[1:3]...), result.minimizer[4:6])
    new_pose, initial_error, result.ssr, outliers, n_outliers
end

"""
Perform Bundle-Adjustment on the new frame and its covisibility graph.

Minimize error function over all KeyFrame's extrinsic parameters
in the covisibility graph and their corresponding MapPoint's positions.
Afterwards, update these parameters.
"""
function local_bundle_adjustment!(
    map_manager::MapManager, new_frame::Frame, params::Params,
)
    new_frame.nb_3d_kpts < params.min_cov_score && return
    # Fix extrinsics of the two KeyFrames, to make them act as anchors.
    n_fixed_keyframes = 2
    # Get `new_frame`'s covisible KeyFrames.
    map_cov_kf = new_frame.covisible_kf |> copy
    map_cov_kf[new_frame.kfid] = new_frame.nb_3d_kpts
    @debug "[LBA] Initial covisibility size $(length(map_cov_kf))."

    bad_keypoints = Set{Int64}() # kpid/mpid
    local_keyframes = Dict{Int64, Frame}() # kfid → Frame
    # {mpid → {kfid → pixel}}
    map_points = OrderedDict{Int64, OrderedDict{Int64, Point2f}}()
    extrinsics = Dict{Int64, NTuple{6, Float64}}() # kfid → extrinsics
    kp_ids_optimize = Set{Int64}() # kpid

    constant_extrinsics = Dict{Int64, Bool}() # kfid → is constant
    n_constants = 0

    # Specifies maximum KeyFrame id in the covisibility graph.
    # To avoid adding observer to the BA problem,
    # that is more recent than the `new_frame`.
    max_kfid = new_frame.kfid

    # Go through all KeyFrames in covisibility graph, get their extrinsics,
    # mark them constant/non-constant, get their 3D Keypoints.
    for (kfid, cov_score) in map_cov_kf
        kfid in keys(map_manager.frames_map) ||
            (remove_covisible_kf!(new_frame, kfid); continue)

        kf = map_manager.frames_map[kfid]
        local_keyframes[kfid] = kf
        extrinsics[kfid] = kf |> get_cw_ba

        (cov_score < params.min_cov_score || kfid == 0) &&
            (constant_extrinsics[kfid] = true; n_constants += 1; continue)
        constant_extrinsics[kfid] = false

        # Add ids of the 3D Keypoints to optimize.
        for (kpid, kp) in kf.keypoints
            kp.is_3d && push!(kp_ids_optimize, kpid)
        end
    end
    @debug "[LBA] N 3D Keypoints to optimize $(length(kp_ids_optimize))."
    @debug "[LBA] Max KF id $max_kfid."

    n_pixels = 0

    # Go through all 3D Keypoints to optimize.
    # Link MapPoint with the observer KeyFrames
    # and their corresponding pixel coordinates.
    for kpid in kp_ids_optimize
        kpid in keys(map_manager.map_points) || continue
        mp = map_manager.map_points[kpid]
        is_bad!(mp) && (push!(bad_keypoints, kpid); continue)

        # Link observer KeyFrames with the MapPoint.
        # Add observer KeyFrames as constants, if not yet added.
        mplink = Dict{Int64, Point2f}()
        for observer_id in mp.observer_keyframes_ids
            observer_id > max_kfid && continue
            # Get observer KeyFrame.
            # If not in the local map,
            # then add it from the global FramesMap as a constant.
            if observer_id in keys(local_keyframes)
                observer_kf = local_keyframes[observer_id]
            else
                if !(observer_id in keys(map_manager.frames_map))
                    remove_mappoint_obs!(map_manager, kpid, observer_id)
                    continue
                end
                observer_kf = map_manager.frames_map[observer_id]
                local_keyframes[observer_id] = observer_kf
                extrinsics[observer_id] = observer_kf |> get_cw_ba

                constant_extrinsics[observer_id] = true
                n_constants += 1
            end
            # Get corresponding pixel coordinate and link it to the MapPoint.
            if !(kpid in keys(observer_kf.keypoints))
                remove_mappoint_obs!(map_manager, kpid, observer_id)
                continue
            end
            observer_kp = observer_kf.keypoints[kpid]
            mplink[observer_id] = observer_kp.undistorted_pixel
            n_pixels += 1
        end
        map_points[mp.id] = mplink
    end

    # Ensure there are at least 2 fixed Keyframes.
    if (n_constants < 2 && length(extrinsics) > 2)
        for kfid in keys(constant_extrinsics)
            constant_extrinsics[kfid] && continue
            constant_extrinsics[kfid] = true
            n_constants += 1
            n_constants == 2 && break
        end
    end

    @debug "[LBA] Covisibility size with observers $(length(local_keyframes))."
    @debug "[LBA] N Pixels $n_pixels."

    # Convert data to the Bundle-Adjustment format.
    constants_matrix = Vector{Bool}(undef, length(extrinsics))
    extrinsics_matrix = Matrix{Float64}(undef, 6, length(extrinsics))
    points_matrix = Matrix{Float64}(undef, 3, length(map_points))
    pixels_matrix = Matrix{Float64}(undef, 2, n_pixels)

    points_ids, extrinsics_ids = Int64[], Int64[]
    extrinsics_order = Dict{Int64, Int64}() # kfid -> nkf

    extrinsic_id, pixel_id, point_id = 1, 1, 1

    # Convert to matrix form.
    for (mpid, mplink) in map_points
        points_matrix[:, point_id] .= map_manager.map_points[mpid].position

        for (kfid, pixel) in mplink
            push!(points_ids, point_id)

            pixels_matrix[:, pixel_id] .= pixel
            pixel_id += 1

            if kfid in keys(extrinsics_order)
                push!(extrinsics_ids, extrinsics_order[kfid])
                continue
            end

            constants_matrix[extrinsic_id] = constant_extrinsics[kfid]
            extrinsics_matrix[:, extrinsic_id] .= extrinsics[kfid]
            extrinsics_order[kfid] = extrinsic_id
            push!(extrinsics_ids, extrinsic_id)
            extrinsic_id += 1
        end

        point_id += 1
    end

    @debug "[LBA] N Extrinsics $(extrinsic_id - 1)."
    @debug "[LBA] N Pixels $(pixel_id - 1)."
    @debug "[LBA] N Point id $(point_id - 1)."
    @debug "[LBA] N Constant KeyFrames $(sum(constants_matrix))."

    new_extrinsics, new_points, initial_error, final_error, outliers = bundle_adjustment(
        new_frame.camera, extrinsics_matrix, points_matrix,
        pixels_matrix, points_ids, extrinsics_ids;
        constant_extrinsics=constants_matrix,
        iterations=10, show_trace=true,
    )
    @debug "[LBA] N Outliers $(sum(outliers))."
    @debug "[LBA] BA error $initial_error → $final_error."

    # Select outliers and prepare them for removal.
    kfmp_outliers = Dict{Int64, Int64}()
    pixel_id = 1
    for (mpid, mplink) in map_points
        for (kfid, _) in mplink
            outliers[pixel_id] || (pixel_id += 1; continue)
            pixel_id += 1

            kfid in keys(map_cov_kf) &&
                remove_mappoint_obs!(map_manager, mpid, kfid)
            kfid == map_manager.current_frame.kfid &&
                remove_obs_from_current_frame!(map_manager, mpid)

            push!(bad_keypoints, mpid)
        end
    end

    @debug "[LBA] N Bad Keypoints $(length(bad_keypoints))"

    # Update KeyFrame poses.
    for (kfid, nkfid) in extrinsics_order
        constant_extrinsics[kfid] && continue
        set_cw_ba!(
            map_manager.frames_map[kfid], @view(new_extrinsics[:, nkfid]),
        )
    end

    for (pid, mpid) in enumerate(keys(map_points))
        if !(mpid in keys(map_manager.map_points))
            mpid in bad_keypoints && pop!(bad_keypoints, mpid)
            continue
        end

        mp = map_manager.map_points[mpid]
        if is_bad!(mp)
            remove_mappoint!(map_manager, mpid)
            mpid in bad_keypoints && pop!(bad_keypoints, mpid)
            continue
        end

        # MapPoint culling.
        # Remove MapPoint, if it has < 3 observers,
        # not observed by the current frames_map
        # and was added less than 3 keyframes ago.
        # Meaning it was unrealiable.
        if length(mp.observer_keyframes_ids) < 3
            if mp.kfid < new_frame.kfid - 3 && !mp.is_observed
                remove_mappoint!(map_manager, mpid)
                mpid in bad_keypoints && pop!(bad_keypoints, mpid)
                continue
            end
        end

        # MapPoint is good, update its position.
        new_position = @view(new_points[:, pid])
        set_position!(
            map_manager.map_points[mpid], new_position, 1.0 / new_position[3],
        )
    end

    # MapPoint culling for bad observations.
    for mpid in bad_keypoints
        mpid in keys(map_manager.map_points) && mpid in keys(map_points) ||
            continue

        mp = map_manager.map_points[mpid]
        is_bad!(mp) && (remove_mappoint!(map_manager, mpid); continue)

        if length(mp.observer_keyframes_ids) < 3
            if mp.kfid < new_frame.kfid - 3 && !mp.is_observed
                remove_mappoint!(map_manager, mpid)
            end
        end
    end
end
