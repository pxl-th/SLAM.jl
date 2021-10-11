"""
Front-End is responsible for tracking keypoints
and computing poses for the Frames.
It also decides when the system needs a new Keyframe in the map.

# Parameters:

- `current_frame::Frame`: Current frame that is being processed.
    This is a shared Frame between FrontEnd, MapManager, Estimator
    and SlamManager.
- `motion_model::MotionModel`: Motion model that is used to predict
    pose for the Frame before the actual pose for it was computed.
- `map_manager::MapManager`: Map manager that is responsible for the
    creation of new Keyframes in the map.
- `params::Params`: Parameters of the system.
- `current_image::Matrix{Gray{Float64}}`: Current image that is processed.
- `previous_image::Matrix{Gray{Float64}}`: Previous processed image.
- `current_pyramid::ImageTracking.LKPyramid`: Pre-computed pyramid
    that is used for optical flow tracking for the `current_image`.
- `previous_pyramid::ImageTracking.LKPyramid`: Pre-computed pyramid
    that is used for optical flow tracking for the `previous_image`.
"""
mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    map_manager::MapManager
    params::Params

    current_image::Matrix{Gray{Float64}}
    previous_image::Matrix{Gray{Float64}}

    current_pyramid::ImageTracking.LKPyramid
    previous_pyramid::ImageTracking.LKPyramid
end

function FrontEnd(params::Params, frame::Frame, map_manager::MapManager)
    empty_pyr = ImageTracking.LKPyramid(
        [Matrix{Gray{Float64}}(undef, 0, 0)],
        nothing, nothing, nothing, nothing, nothing)
    FrontEnd(
        frame, MotionModel(), map_manager, params,
        Matrix{Gray{Float64}}(undef, 0, 0), Matrix{Gray{Float64}}(undef, 0, 0),
        empty_pyr, empty_pyr)
end

"""
```julia
track!(fe::FrontEnd, image, time)
```

Given an image and time at which it was taken, track keypoints in it.
After tracking, decide if the system needs a new Keyframe added to the map.
If it is the first image to be tracked, then Keyframe is always needed.

# Returns:

`true` if the system needs a new Keyframe, otherwise `false`.
"""
function track!(fe::FrontEnd, image, time, visualizer)
    is_kf_required = false

    lock(fe.map_manager.map_lock)
    try
        is_kf_required = track_mono!(fe, image, time, visualizer)
        @debug "Pose $(fe.current_frame.id) WC: $(fe.current_frame.wc[1:3, 4])"

        # vimage = RGB{Float64}.(fe.current_image)
        # draw_keypoints!(vimage, fe.current_frame)
        # save("/home/pxl-th/projects/slam-data/images/frame-$(fe.current_frame.id).png", vimage)

        is_kf_required && create_keyframe!(fe.map_manager, image)
    catch e
        showerror(stdout, e)
        display(stacktrace(catch_backtrace()))
    finally
        unlock(fe.map_manager.map_lock)
    end
    is_kf_required
end

function track_mono!(fe::FrontEnd, image, image_time, visualizer)
    preprocess!(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    set_wc!(
        fe.current_frame, fe.motion_model(fe.current_frame.wc, image_time),
        visualizer)

    t1 = time()
    klt_tracking!(fe)
    t2 = time()
    @debug "[FE] KLT Tracking ($(t2 - t1) sec)."

    if !fe.params.vision_initialized
        if fe.current_frame.nb_keypoints < 50
            @warn "[Front End] NB KP < 50. Reset required."
            fe.params.reset_required = true
            return false
        elseif check_ready_for_init!(fe, visualizer)
            @debug "[Front End] System ready for initialization."
            fe.params.vision_initialized = true
            return true
        else
            @debug "[Front End] System not ready for initialization."
            return false
        end
    end

    # Epipolar filtering to remove outliers.
    # In case P3P fails this pose will be used.
    t1 = time()
    pose_5pt = compute_pose_5pt!(fe; min_parallax=5.0, use_motion_model=true)
    t2 = time()
    @debug "[FE] 5PT Pose ($(t2 - t1) sec)."

    fe.map_manager.nb_keyframes > 2 && pose_5pt ≢ nothing &&
        set_cw!(fe.current_frame, pose_5pt, visualizer)

    t1 = time()
    compute_pose!(fe, visualizer)
    t2 = time()
    @debug "[FE] Compute Pose ($(t2 - t1) sec)."

    # Update motion model from estimated pose.
    update!(fe.motion_model, fe.current_frame.wc, image_time)
    check_new_kf_required(fe)
end

"""
```julia
compute_pose!(fe::FrontEnd)
```

Compute pose of a current Frame using P3P Ransac algorithm.
Pose is computed from the triangulated Keypoints (MapPoints) that are visible
in this frame.

# Returns:

`true` if the pose was successfully computed and applied to current Frame,
otherwise `false`.
"""
function compute_pose!(fe::FrontEnd, visualizer)
    if fe.current_frame.nb_3d_kpts < 5
        @warn "[Front-End] Not enough 3D keypoints to compute P3P $(fe.current_frame.nb_3d_kpts)."
        return false
    end

    p3p_pixels = Vector{Point2f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_pdn_positions = Vector{Point3f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_3d_points = Vector{Point3f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_kpids = Vector{Int64}(undef, fe.current_frame.nb_3d_kpts)
    i = 1

    for kp in values(fe.current_frame.keypoints)
        kp.is_3d || continue
        mp = get(fe.map_manager.map_points, kp.id, nothing)
        mp ≡ nothing && continue

        p3p_pdn_positions[i] = normalize(kp.position)
        # Convert pixel to `(x, y)` format, expected by P3P.
        p3p_pixels[i] = kp.undistorted_pixel[[2, 1]]
        p3p_3d_points[i] = mp.position
        p3p_kpids[i] = kp.id
        i += 1
    end

    i -= 1
    p3p_pdn_positions = @view(p3p_pdn_positions[1:i])
    p3p_pixels = @view(p3p_pixels[1:i])
    p3p_3d_points = @view(p3p_3d_points[1:i])
    p3p_kpids = @view(p3p_kpids[1:i])

    # P3P computes world → camera projection.
    res = p3p_ransac(
        p3p_3d_points, p3p_pixels, p3p_pdn_positions,
        fe.current_frame.camera.K; threshold=fe.params.max_reprojection_error,
    )
    if res ≡ nothing
        @warn "[FE] P3P Ransac returned Nothing."
        fe |> reset_frame!
        return false
    end

    n_inliers, model = res
    if n_inliers < 5 || model ≡ nothing
        @warn "[FE] P3P too few inliers - resetting!"
        fe |> reset_frame!
        return false
    end

    KP, inliers, error = model
    set_cw!(
        fe.current_frame, to_4x4(fe.current_frame.camera.iK * KP), visualizer)
    # Remove outliers after P3P.
    for (kpid, inlier) in zip(p3p_kpids, inliers)
        inlier || remove_obs_from_current_frame!(fe.map_manager, kpid)
    end

    # Prepare data for BA refinement.
    p3p_3d_points = @view(p3p_3d_points[inliers])
    p3p_kpids = @view(p3p_kpids[inliers])

    c = 1
    for i in 1:length(p3p_pixels)
        inliers[i] || continue
        p = p3p_pixels[i]
        p3p_pixels[c] = Point2f(p[2], p[1])
        c += 1
    end
    p3p_pixels = @view(p3p_pixels[1:n_inliers])

    # Refinement.
    new_T, init_error, res_error, outliers, n_outliers = pnp_bundle_adjustment(
        fe.current_frame.camera, fe.current_frame.cw,
        p3p_pixels, p3p_3d_points;
        iterations=10, show_trace=false,
        repr_ϵ=fe.params.max_reprojection_error,
    )
    if length(p3p_3d_points) - n_outliers < 5 || res_error > init_error
        @warn "[FE] P3P BA too few inliers - resetting!"
        fe |> reset_frame!
        return false
    end

    for (kpid, outlier) in zip(p3p_kpids, outliers)
        outlier && remove_obs_from_current_frame!(fe.map_manager, kpid)
    end

    set_cw!(fe.current_frame, new_T, visualizer)
    true
end

"""
```julia
compute_pose_5pt!(fe::FrontEnd; min_parallax, use_motion_model)
```

Copmute pose for pixel correspondences using 5-point algorithm
to recover essential matrix, then pose from it.

# Arguments:

- `min_parallax`: Minimum parallax required between pixels in the current Frame
    and previous Keyframe to compute pose.
    Note, that parallax is rotation-compensated, meaning a rotation from
    current to previous frame is computed and applied to every pixel.
- `use_motion_model`: If `true`, then use constant-velocity motion model
    to predict next pose from previous frame.
    Otherwise, the computed pose will be "local".

# Returns:

If successfull, 4x4 pose matrix, that transforms points
from previous Keyframe to current Frame.
Otherwise `nothing`.
"""
function compute_pose_5pt!(fe::FrontEnd; min_parallax, use_motion_model)
    if fe.current_frame.nb_keypoints < 8
        @debug "[FE] Not enough keypoints for initialization: " *
            "$(fe.current_frame.nb_keypoints)"
        return nothing
    end

    previous_keyframe = get(
        fe.map_manager.frames_map, fe.current_frame.kfid, nothing)
    previous_keyframe ≡ nothing && return nothing
    R_compensation = get_Rcw(previous_keyframe) * get_Rwc(fe.current_frame)

    n_parallax = 0
    avg_parallax = 0.0

    previous_points = Vector{Point2f}(undef, fe.current_frame.nb_keypoints)
    current_points = Vector{Point2f}(undef, fe.current_frame.nb_keypoints)
    kp_ids = Vector{Int64}(undef, fe.current_frame.nb_keypoints)
    i = 1

    # Get all Keypoint's positions and compute rotation-compensated parallax.
    @inbounds for keypoint in values(fe.current_frame.keypoints)
        pkf_keypoint = get(previous_keyframe.keypoints, keypoint.id, nothing)
        pkf_keypoint ≡ nothing && continue

        # Convert points to `(x, y)` format as expected by five points.
        previous_points[i] = pkf_keypoint.undistorted_pixel[[2, 1]]
        current_points[i] = keypoint.undistorted_pixel[[2, 1]]
        kp_ids[i] = keypoint.id
        i += 1

        # Compute rotation-compensated parallax.
        rot_position = R_compensation * keypoint.position
        rot_px = project(fe.current_frame.camera, rot_position)
        avg_parallax += norm(rot_px - pkf_keypoint.undistorted_pixel)
        n_parallax += 1
    end
    if n_parallax < 8
        @warn "[FE] Not enough keypoints in previous KF " *
            "to compute 5pt Essential Matrix."
        return nothing
    end

    avg_parallax /= n_parallax
    if avg_parallax < min_parallax
        @warn "[FE] Not enough parallax ($avg_parallax) " *
            "to compute 5pt Essential Matrix."
        return nothing
    end

    i -= 1
    previous_points = @view(previous_points[1:i])
    current_points = @view(current_points[1:i])
    kp_ids = @view(kp_ids[1:i])

    # `P` is `cw`: transforms from world (previous frame) to current frame.
    n_inliers, (_, P, inliers, _) = five_point_ransac(
        previous_points, current_points,
        fe.current_frame.camera.K, fe.current_frame.camera.K;
        max_repr_error=fe.params.max_reprojection_error)
    if n_inliers < 5
        @warn "[FE] Not enough inliers ($n_inliers) for the " *
            "5pt Essential Matrix."
        return nothing
    end

    if n_inliers != n_parallax
        for (i, inlier) in enumerate(inliers)
            inlier || remove_obs_from_current_frame!(fe.map_manager, kp_ids[i])
        end
    end

    if use_motion_model
        # Get motion-model translation scale from last KeyFrame.
        prev_cw = get_cw(previous_keyframe)
        current = prev_cw * get_wc(fe.current_frame)
        scale = norm(current[1:3, 4])

        R, t = P[1:3, 1:3], P[1:3, 4]
        t = scale .* normalize(t)
        return to_4x4(R, t) * prev_cw
    end
    to_4x4(P) # cw pose
end

"""
```julia
check_ready_for_init!(fe::FrontEnd)
```

Check if there is enough average rotation compensated parallax
between current Frame and previous KeyFrame.

Additionally, compute Essential matrix using 5-point Ransac algorithm
to filter out outliers and check if there is enough inliers to proceed.
"""
function check_ready_for_init!(fe::FrontEnd, visualizer)
    avg_parallax = compute_parallax(
        fe, fe.current_frame.kfid;
        compensate_rotation=false, median_parallax=false)
    @debug "[FE] Initial parallax $avg_parallax vs $(fe.params.initial_parallax)."
    avg_parallax ≤ fe.params.initial_parallax && return false
    pose = compute_pose_5pt!(
        fe; min_parallax=fe.params.initial_parallax, use_motion_model=false)
    ready = pose ≢ nothing
    ready && set_cw!(fe.current_frame, pose, visualizer)
    ready
end

"""
```julia
check_new_kf_required(fe::FrontEnd)
```

Check if we need to insert a new KeyFrame into the Map.
"""
function check_new_kf_required(fe::FrontEnd)
    prev_kf = get(fe.map_manager.frames_map, fe.current_frame.kfid, nothing)
    prev_kf ≡ nothing && return false

    # Id difference since last KeyFrame.
    frames_δ = fe.current_frame.id - prev_kf.id
    if fe.current_frame.nb_occupied_cells < 0.33 * fe.params.max_nb_keypoints &&
        frames_δ ≥ 5 && !fe.params.local_ba_on
        return true
    end
    if fe.current_frame.nb_3d_kpts < 20 && frames_δ ≥ 2
        return true
    end
    if fe.current_frame.nb_3d_kpts > 0.5 * fe.params.max_nb_keypoints &&
        (fe.params.local_ba_on || frames_δ < 2)
        return false
    end

    # Time difference since last KeyFrame.
    time_δ = fe.current_frame.time - prev_kf.time
    # TODO option for stereo
    median_parallax = compute_parallax(
        fe, prev_kf.kfid; compensate_rotation=true, only_2d=false,
    )
    cx = median_parallax ≥ fe.params.initial_parallax / 2.0 # TODO || stereo
    c0 = median_parallax ≥ fe.params.initial_parallax
    c1 = fe.current_frame.nb_3d_kpts < 0.75 * prev_kf.nb_3d_kpts
    c2 = fe.current_frame.nb_occupied_cells < 0.5 * fe.params.max_nb_keypoints &&
        fe.current_frame.nb_3d_kpts < 0.85 * prev_kf.nb_3d_kpts &&
        !fe.params.local_ba_on

    cx && (c0 || c1 || c2)
end

"""
```julia
compute_parallax(
    fe::FrontEnd, current_frame_id;
    compensate_rotation = true, only_2d = true, median_parallax = true,
)
```

Compute parallax in pixels between current Frame
and the provided `current_frame_id` Frame.

# Arguments:

- `compensate_rotation::Bool`:
    Compensate rotation by computing relative rotation between
    current Frame and previous Keyframe if `true`. Default is `true`.
- `only_2d::Bool`: Consider only 2d keypoints. Default is `true`.
- `median_parallax::Bool`:
    Instead of the average, compute median parallax. Default is `true`.
"""
function compute_parallax(
    fe::FrontEnd, current_frame_id;
    compensate_rotation = true, only_2d = true, median_parallax = true,
)
    frame = get(fe.map_manager.frames_map, current_frame_id, nothing)
    if frame ≡ nothing
        @warn "[Front-End] Error in `compute_parallax`! " *
            "Keyframe $current_frame_id does not exist."
        return 0.0
    end

    current_rotation = compensate_rotation ?
        get_Rcw(frame) * get_Rwc(fe.current_frame) :
        SMatrix{3, 3, Float64}(I)

    avg_parallax = 0.0
    n_parallax = 0
    parallax_set = Float64[]

    for keypoint in values(fe.current_frame.keypoints)
        only_2d && keypoint.is_3d && continue
        keypoint.id in keys(frame.keypoints) || continue

        upx = keypoint.undistorted_pixel
        compensate_rotation && (upx =
            project(frame.camera, current_rotation * keypoint.position);)

        parallax = norm(upx - get_keypoint_unpx(frame, keypoint.id))
        avg_parallax += parallax
        n_parallax += 1

        median_parallax && push!(parallax_set, parallax)
    end
    n_parallax == 0 && return 0.0

    if median_parallax
        avg_parallax = parallax_set |> median
    else
        avg_parallax /= n_parallax
    end
    avg_parallax
end

function preprocess!(fe::FrontEnd, image)
    fe.previous_image = fe.current_image
    fe.previous_pyramid = fe.current_pyramid

    fe.current_pyramid = ImageTracking.LKPyramid(
        image, fe.params.pyramid_levels; σ=fe.params.pyramid_σ)
    fe.current_image = image
end

"""
```julia
klt_tracking!(fe::FrontEnd)
```

Track keypoints from previous frame to current frame.
"""
function klt_tracking!(fe::FrontEnd)
    optical_flow_matching!(
        fe.map_manager, fe.current_frame,
        fe.previous_pyramid, fe.current_pyramid, false)
end

"""
```julia
reset_frame!(fe::FrontEnd)
```

Reset current Frame in Front-End.
"""
function reset_frame!(fe::FrontEnd)
    for kpid in keys(fe.current_frame.keypoints)
        remove_obs_from_current_frame!(fe.map_manager, kpid)
    end
    fe.current_frame.keypoints |> empty!
    fe.current_frame.keypoints_grid .|> empty!

    fe.current_frame.nb_2d_kpts = 0
    fe.current_frame.nb_3d_kpts = 0
    fe.current_frame.nb_keypoints = 0
    fe.current_frame.nb_occupied_cells = 0
end

"""
```julia
reset!(fe::FrontEnd)
```

Reset Front-End.
"""
function reset!(fe::FrontEnd)
    empty_pyr = ImageTracking.LKPyramid(
        [Matrix{Gray{Float64}}(undef, 0, 0)],
        nothing, nothing, nothing, nothing, nothing)
    fe.previous_pyramid = empty_pyr
    fe.current_pyramid = empty_pyr
end
