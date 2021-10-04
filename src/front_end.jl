mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    map_manager::MapManager
    params::Params

    current_image::Matrix{Gray{Float64}}
    previous_image::Matrix{Gray{Float64}}

    current_pyramid::ImageTracking.LKPyramid
    previous_pyramid::ImageTracking.LKPyramid

    p3p_required::Bool
end

function FrontEnd(params::Params, frame::Frame, map_manager::MapManager)
    empty_pyr = ImageTracking.LKPyramid(
        [Matrix{Gray{Float64}}(undef, 0, 0)],
        nothing, nothing, nothing, nothing, nothing)
    FrontEnd(
        frame, MotionModel(), map_manager, params,
        Matrix{Gray{Float64}}(undef, 0, 0), Matrix{Gray{Float64}}(undef, 0, 0),
        empty_pyr, empty_pyr, false)
end

function track!(fe::FrontEnd, image, time)
    is_kf_required = false

    lock(fe.map_manager.map_lock)
    try
        is_kf_required = track_mono!(fe, image, time)
        @info "Pose $(fe.current_frame.id) WC: $(fe.current_frame.wc[1:3, 4])"

        vimage = RGB{Float64}.(fe.current_image)
        draw_keypoints!(vimage, fe.current_frame)
        save("/home/pxl-th/projects/slam-data/images/frame-$(fe.current_frame.id).png", vimage)

        if is_kf_required
            create_keyframe!(fe.map_manager, image)
        end
        is_kf_required
    catch e
        showerror(stdout, e)
        display(stacktrace(catch_backtrace()))
    finally
        unlock(fe.map_manager.map_lock)
    end
    is_kf_required
end

function track_mono!(fe::FrontEnd, image, time)::Bool
    preprocess!(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    # @info "[FE] Old Pose $(fe.current_frame.id): $(get_wc_ba(fe.current_frame))"
    set_wc!(fe.current_frame, fe.motion_model(fe.current_frame.wc, time))
    # @info "[FE] New Pose $(fe.current_frame.id): $(get_wc_ba(fe.current_frame))"

    klt_tracking!(fe)

    if !fe.params.vision_initialized
        if fe.current_frame.nb_keypoints < 50
            @warn "[Front End] NB KP < 50. Reset required."
            fe.params.reset_required = true
            return false
        elseif check_ready_for_init!(fe)
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
    pose_5pt = compute_pose_5pt!(
        fe; min_parallax=5.0, #2.0 * fe.params.max_reprojection_error,
        use_motion_model=true)

    fe.map_manager.nb_keyframes > 2 && pose_5pt ≢ nothing &&
        set_cw!(fe.current_frame, pose_5pt)

    compute_pose!(fe)

    # Update motion model from estimated pose.
    update!(fe.motion_model, fe.current_frame.wc, time)
    check_new_kf_required(fe)
end

"""
Compute pose of a current Frame using P3P Ransac algorithm.
"""
function compute_pose!(fe::FrontEnd)
    if fe.current_frame.nb_3d_kpts < 5
        @warn "[Front-End] Not enough 3D keypoints to compute P3P $(fe.current_frame.nb_3d_kpts)."
        return false
    end

    do_p3p = true
    # do_p3p = fe.p3p_required && fe.params.do_p3p

    p3p_pixels = Vector{Point2f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_pdn_positions = Vector{Point3f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_3d_points = Vector{Point3f}(undef, fe.current_frame.nb_3d_kpts)
    p3p_kpids = Vector{Int64}(undef, fe.current_frame.nb_3d_kpts)
    i = 1

    for kp in values(fe.current_frame.keypoints)
        kp.is_3d || continue
        mp = get(fe.map_manager.map_points, kp.id, nothing)
        mp ≡ nothing && continue

        do_p3p && (p3p_pdn_positions[i] = normalize(kp.position);)
        # Convert pixel to `(x, y)` format, expected by P3P.
        p3p_pixels[i] = kp.undistorted_pixel[[2, 1]]
        p3p_3d_points[i] = mp.position
        p3p_kpids[i] = kp.id
        i += 1
    end

    i -= 1
    do_p3p && (p3p_pdn_positions = @view(p3p_pdn_positions[1:i]);)
    p3p_pixels = @view(p3p_pixels[1:i])
    p3p_3d_points = @view(p3p_3d_points[1:i])
    p3p_kpids = @view(p3p_kpids[1:i])

    if do_p3p
        @info "[FE] P3P Ransac..."
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
        set_cw!(fe.current_frame, to_4x4(fe.current_frame.camera.iK * KP))
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
    else
        for i in 1:length(p3p_pixels)
            p = p3p_pixels[i]
            p3p_pixels[i] = Point2f(p[2], p[1])
        end
    end

    new_T, init_error, res_error, outliers, n_outliers = pnp_bundle_adjustment(
        fe.current_frame.camera, fe.current_frame.cw,
        p3p_pixels, p3p_3d_points;
        iterations=10, show_trace=false, repr_ϵ=fe.params.max_reprojection_error,
    )
    if length(p3p_3d_points) - n_outliers < 5 || res_error > init_error
        @warn "[FE] P3P BA too few inliers - resetting!"
        fe.p3p_required = true
        fe |> reset_frame!
        return false
    end

    for (kpid, outlier) in zip(p3p_kpids, outliers)
        outlier && remove_obs_from_current_frame!(fe.map_manager, kpid)
    end

    set_cw!(fe.current_frame, new_T)
    fe.p3p_required = false
    true
end

function compute_pose_5pt!(
    fe::FrontEnd; min_parallax::Real, use_motion_model::Bool,
)::Union{Nothing, SMatrix{4, 4, Float64}}
    # Ensure there are enough keypoints for the essential matrix computation.
    if fe.current_frame.nb_keypoints < 8
        @debug "[Front-End] Not enough keypoints for initialization: " *
            "$(fe.current_frame.nb_keypoints)"
        return nothing
    end

    # Setup Essential matrix computation.
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
    for keypoint in values(fe.current_frame.keypoints)
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
        @warn "[Front-End] Not enough keypoints in previous KF " *
            "to compute 5pt Essential Matrix."
        return nothing
    end

    avg_parallax /= n_parallax
    if avg_parallax < min_parallax
        @warn "[Front-End] Not enough parallax ($avg_parallax) " *
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
        @warn "[Front-End] Not enough inliers ($n_inliers) for the " *
            "5pt Essential Matrix."
        return nothing
    end

    # Remove outliers from the current frame.
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
Check if there is enough average rotation compensated parallax
between current Frame and previous KeyFrame.

Additionally, compute Essential matrix using 5-point Ransac algorithm
to filter out outliers and check if there is enough inliers to proceed.
"""
function check_ready_for_init!(fe::FrontEnd)
    avg_parallax = compute_parallax(
        fe, fe.current_frame.kfid;
        compensate_rotation=false, median_parallax=false)
    @info "[FE] Initial parallax $avg_parallax vs $(fe.params.initial_parallax)."
    avg_parallax ≤ fe.params.initial_parallax && return false
    pose = compute_pose_5pt!(
        fe; min_parallax=fe.params.initial_parallax, use_motion_model=false)
    ready = pose ≢ nothing
    ready && set_cw!(fe.current_frame, pose)
    ready
end

"""
Check if we need to insert a new KeyFrame into the Map.
"""
function check_new_kf_required(fe::FrontEnd)::Bool
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
    fe::FrontEnd, current_frame_id::Int;
    compensate_rotation::Bool = true,
    only_2d::Bool = true, median_parallax::Bool = true,
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
    # image = adjust_histogram(image, AdaptiveEqualization())

    fe.previous_image = fe.current_image
    fe.previous_pyramid = fe.current_pyramid

    fe.current_pyramid = ImageTracking.LKPyramid(
        image, fe.params.pyramid_levels; σ=fe.params.pyramid_σ)
    fe.current_image = image
end

function klt_tracking!(fe::FrontEnd)
    optical_flow_matching!(
        fe.map_manager,
        fe.current_frame, fe.previous_pyramid, fe.current_pyramid;
        window_size=fe.params.window_size,
        max_distance=fe.params.max_ktl_distance,
        pyramid_levels=fe.params.pyramid_levels, stereo=false)
end

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

function reset!(fe::FrontEnd)
    empty_pyr = ImageTracking.LKPyramid(
        [Matrix{Gray{Float64}}(undef, 0, 0)],
        nothing, nothing, nothing, nothing, nothing,
    )
    fe.previous_pyramid = empty_pyr
    fe.current_pyramid = empty_pyr
end
