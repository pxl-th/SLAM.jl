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
function bundle_adjustment(
    camera::Camera,
    extrinsics::Matrix{Float64},
    points::Matrix{Float64},
    pixels::Matrix{Float64},
    points_ids, extrinsic_ids;
    iterations::Int = 10, show_trace::Bool = false,
)
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

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

            px = @view(pixels[:, i]) # (y, x) format
            inv_z = 1.0 / pt[3]
            Y[id + 1] = px[1] - (fy * pt[2] * inv_z + cy)
            Y[id + 2] = px[2] - (fx * pt[1] * inv_z + cx)
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

    residue!(Y, X0)
    initial_error = mapreduce(abs2, +, Y)
    Y .= 0.0

    colors = matrix_colors(J_sparsity)
    g! = (j, x) -> forwarddiff_color_jacobian!(j, residue!, x; colorvec=colors)
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
    new_extrinsics, new_points, initial_error, result.ssr
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

    bad_keypoints = Set{Int64}() # kpid
    local_keyframes = Dict{Int64, Frame}() # kfid → Frame
    # {mpid → {kfid → pixel}}
    map_points = OrderedDict{Int64, OrderedDict{Int64, Point2f}}()
    extrinsics = Dict{Int64, NTuple{6, Float64}}() # kfid → extrinsics
    constant_extrinsics = Dict{Int64, Bool}() # kfid → is constant
    kp_ids_optimize = Set{Int64}() # kpid

    # Specifies maximum KeyFrame id in the covisibility graph.
    # To avoid adding observer to the BA problem,
    # that is more recent than the `new_frame`.
    max_kfid = new_frame.kfid

    # Go through all KeyFrames in covisibility graph, get their extrinsics,
    # mark them constant/non-constant, get their 3D Keypoints.
    for (kfid, cov_score) in map_cov_kf
        @debug "[LBA] Covisibility KF $kfid ↔ $cov_score"

        kfid in keys(map_manager.frames_map) ||
            (remove_covisible_kf!(new_frame, kfid); continue)

        kf = map_manager.frames_map[kfid]
        local_keyframes[kfid] = kf
        extrinsics[kfid] = kf |> get_cw_ba

        (cov_score < params.min_cov_score || kfid == 0) &&
            (constant_extrinsics[kfid] = true; continue)
        constant_extrinsics[kfid] = false
        @debug "[LBA] Covisibility KF $kfid ↔ const $(constant_extrinsics[kfid])"

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

        mp_pixels = Dict{Int64, Point2f}()
        # Link observer KeyFrames with the MapPoint.
        # Add observer KeyFrames as constants, if not yet added.
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
            end
            # Get corresponding pixel coordinate and link it to the MapPoint.
            if !(kpid in keys(observer_kf.keypoints))
                remove_mappoint_obs!(map_manager, kpid, observer_id)
                continue
            end
            observer_kp = observer_kf.keypoints[kpid]
            mp_pixels[observer_id] = observer_kp.undistorted_pixel
            n_pixels += 1
        end
        map_points[mp.id] = mp_pixels
    end

    @debug "[LBA] Covisibility size with observers $(length(local_keyframes))."
    @debug "[LBA] N Pixels $n_pixels."

    # Convert data to the Bundle-Adjustment format.
    constants_matrix = Vector{Bool}(undef, length(extrinsics))
    extrinsics_matrix = Matrix{Float64}(undef, 6, length(extrinsics))
    points_matrix = Matrix{Float64}(undef, 3, length(map_points))
    pixels_matrix = Matrix{Float64}(undef, 2, n_pixels)

    points_ids, extrinsics_ids = Int64[], Int64[]
    extrinsics_order = Dict{Int64, Int64}()

    extrinsic_id = 1
    pixel_id = 1
    point_id = 1

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

    new_extrinsics, new_points, initial_error, final_error = bundle_adjustment(
        new_frame.camera,
        extrinsics_matrix, points_matrix,
        pixels_matrix, points_ids, extrinsics_ids;
        iterations=10, show_trace=true,
    )
    """
    TODO
    - constant mask for jacobian
    - convert result back
    """

    @debug "[LBA] BA error $initial_error → $final_error."
    exit()
end
