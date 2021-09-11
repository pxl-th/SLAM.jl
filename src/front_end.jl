mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    map_manager::MapManager
    params::Params

    current_image::Matrix{Gray{Float64}}
    previous_image::Matrix{Gray{Float64}}

    current_pyramid::ImageTracking.LKPyramid
    previous_pyramid::ImageTracking.LKPyramid
    keyframe_pyramid::ImageTracking.LKPyramid

    p3p_required::Bool
end

function FrontEnd(params::Params, frame::Frame, map_manager::MapManager)
    empty_pyr = ImageTracking.LKPyramid(
        [Matrix{Gray{Float64}}(undef, 0, 0)],
        nothing, nothing, nothing, nothing, nothing,
    )
    FrontEnd(
        frame, MotionModel(), map_manager, params,
        Matrix{Gray{Float64}}(undef, 0, 0), Matrix{Gray{Float64}}(undef, 0, 0),
        empty_pyr, empty_pyr, empty_pyr, false,
    )
end

function track!(fe::FrontEnd, image, time)
    is_kf_required = track_mono!(fe, image, time)
    display(fe.current_frame.wc); println()

    if is_kf_required
        create_keyframe!(fe.map_manager, image)
        fe.keyframe_pyramid = fe.current_pyramid
    end
    is_kf_required
end

function track_mono!(fe::FrontEnd, image, time)::Bool
    preprocess!(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    new_wc = fe.motion_model(fe.current_frame.wc, time)
    set_wc!(fe.current_frame, new_wc)

    klt_tracking!(fe)
    # Epipolar filtering to remove outliers.
    compute_pose_5pt!(fe; min_parallax=2.0 * fe.params.max_reprojection_error)

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
        @warn "[Front-End] Not enough 3D keypoints to compute P3P $(fe.current_frame.nb_3d_kpts) but $n3d."
        return
    end

    do_p3p = fe.p3p_required || fe.params.do_p3p

    p3p_pixels = Point2f[]
    p3p_pdn_positions = Point3f[]
    p3p_3d_points = Point3f[]
    p3p_kpids = Int64[]

    for kp in values(fe.current_frame.keypoints)
        kp.is_3d || continue
        kp.id in keys(fe.map_manager.map_points) || continue

        mp = fe.map_manager.map_points[kp.id]
        do_p3p && push!(p3p_pdn_positions, normalize(kp.position))
        # TODO do_p3p &&
        # Convert pixel to `(x, y)` format, expected by P3P.
        push!(p3p_pixels, kp.undistorted_pixel[[2, 1]])
        push!(p3p_3d_points, mp.position)
        push!(p3p_kpids, kp.id)
    end

    if do_p3p
        # P3P computes world->camera projection.
        n_inliers, model = p3p_ransac(
            p3p_3d_points, p3p_pixels, p3p_pdn_positions,
            fe.current_frame.camera.K; threshold=fe.params.max_reprojection_error,
        )
        if n_inliers < 5 || model ≡ nothing
            @warn "[Front-End] Not enough inliers for reliable P3P pose estimation."
            fe |> reset_frame!
            return
        end
        @debug "[FE] P3P $(n_inliers)/$(length(p3p_pixels)) inliers"
        KP, inliers, error = model
        set_cw!(fe.current_frame, to_4x4(fe.current_frame.camera.iK * KP))

        # Remove outliers after P3P.
        for (kpid, inlier) in zip(p3p_kpids, inliers)
            inlier || remove_obs_from_current_frame!(fe.map_manager, kpid)
        end

        # Prepare data for BA refinement.
        p3p_3d_points = p3p_3d_points[inliers]
        p3p_kpids = p3p_kpids[inliers]
        ba_pixels = [
            Point2f(p[2], p[1]) for (inlier, p) in zip(inliers, p3p_pixels)
            if inlier
        ]
    else
        ba_pixels = [Point2f(p[2], p[1]) for p in p3p_pixels]
    end

    new_T, init_error, res_error, outliers, n_outliers = pnp_bundle_adjustment(
        fe.current_frame.camera, fe.current_frame.cw,
        ba_pixels, p3p_3d_points; iterations=10, show_trace=true,
    )
    @debug "[FE] BA Pose: outliers $n_outliers, $init_error → $res_error."
    if length(p3p_3d_points) - n_outliers < 5 || res_error > init_error
        fe.p3p_required = true
        return
    end

    for (kpid, outlier) in zip(p3p_kpids, outliers)
        outlier && remove_obs_from_current_frame!(fe.map_manager, kpid)
    end
    set_cw!(fe.current_frame, new_T)
    fe.p3p_required = false
end

function compute_pose_5pt!(fe::FrontEnd; min_parallax::Real)
    fe.current_frame.kfid in keys(fe.map_manager.frames_map) || return false

     # Ensure there are enough keypoints for the essential matrix computation.
    if fe.current_frame.nb_keypoints < 8
        @debug "[Front-End] Not enough keypoints for initialization: " *
            "$(fe.current_frame.nb_keypoints)"
        return false
    end

    # Setup Essential matrix computation.
    previous_keyframe = fe.map_manager.frames_map[fe.current_frame.kfid]
    R_compensation = get_Rcw(previous_keyframe) * get_Rwc(fe.current_frame)

    n_parallax = 0
    avg_parallax = 0.0

    previous_points = Point2f[]
    current_points = Point2f[]
    kp_ids = Int[]

    # Get all Keypoint's positions and compute rotation-compensated parallax.
    for keypoint in values(fe.current_frame.keypoints)
        keypoint.id in keys(previous_keyframe.keypoints) || continue

        # Convert points to `(x, y)` format as expected by five points.
        pkf_keypoint = previous_keyframe.keypoints[keypoint.id]
        push!(previous_points, pkf_keypoint.undistorted_pixel[[2, 1]])
        push!(current_points, keypoint.undistorted_pixel[[2, 1]])
        push!(kp_ids, keypoint.id)

        # Compute rotation-compensated parallax.
        rot_position = R_compensation * keypoint.position
        rot_px = project(fe.current_frame.camera, rot_position)
        avg_parallax += norm(rot_px - pkf_keypoint.undistorted_pixel)
        n_parallax += 1
    end

    if n_parallax < 8
        @debug "[Front-End] Not enough keypoints in previous KF " *
            "to compute 5pt Essential Matrix."
        return false
    end

    avg_parallax /= n_parallax
    if (avg_parallax < min_parallax)
        @debug "[Front-End] Not enough parallax ($avg_parallax) " *
            "to compute 5pt Essential Matrix."
        return false
    end
    n_inliers, (E, P, inliers, repr_error) = five_point_ransac(
        previous_points, current_points,
        fe.current_frame.camera.K, fe.current_frame.camera.K,
    )
    if n_inliers < 5
        @debug "[Front-End] Not enough inliers ($n_inliers) for the " *
            "5pt Essential Matrix."
        return false
    end
    # Remove outliers from the current frame.
    if n_inliers != n_parallax
        for (i, inlier) in enumerate(inliers)
            inlier && continue
            remove_obs_from_current_frame!(fe.map_manager, kp_ids[i])
        end
    end
    set_wc!(fe.current_frame, inv(SE3, to_4x4(P)))
    true
end

"""
Check if there is enough average rotation compensated parallax
between current Frame and previous KeyFrame.

Additionally, compute Essential matrix using 5-point Ransac algorithm
to filter out outliers and check if there is enough inliers to proceed.
"""
function check_ready_for_init!(fe::FrontEnd)
    fe.current_frame.kfid in keys(fe.map_manager.frames_map) || return false

    avg_parallax = compute_parallax(
        fe, fe.current_frame.kfid;
        compensate_rotation=false, median_parallax=true,
    )
    avg_parallax ≤ fe.params.initial_parallax && return false

    compute_pose_5pt!(fe; min_parallax=fe.params.initial_parallax)
end

"""
Check if we need to insert a new KeyFrame into the Map.
"""
function check_new_kf_required(fe::FrontEnd)::Bool
    fe.current_frame.kfid in keys(fe.map_manager.frames_map) || return false
    prev_kf = fe.map_manager.frames_map[fe.current_frame.kfid]

    # Id difference since last KeyFrame.
    frames_δ = fe.current_frame.id - prev_kf.id
    @debug "[FE] Check new KF $(fe.current_frame.nb_3d_kpts) 3dkp, δ $frames_δ"
    fe.current_frame.nb_occupied_cells < 0.33 * fe.params.max_nb_keypoints &&
        frames_δ ≥ 5 && return true # TODO && !params.localba_is_on
    fe.current_frame.nb_3d_kpts < 20 && frames_δ ≥ 2 && return true
    fe.current_frame.nb_3d_kpts > 0.5 * fe.params.max_nb_keypoints &&
        frames_δ < 2 && return false # TODO || params.localba_is_on

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
        fe.current_frame.nb_3d_kpts < 0.85 * prev_kf.nb_3d_kpts
        # TODO && params.localba_is_on

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
    if !(current_frame_id in keys(fe.map_manager.frames_map))
        @debug "[Front-End] Error in `compute_parallax`! " *
            "Keyframe $current_frame_id does not exist."
        return 0.0
    end

    frame = fe.map_manager.frames_map[current_frame_id]
    current_rotation = compensate_rotation ?
        get_Rcw(frame) * get_Rwc(fe.current_frame) : SMatrix{3, 3, Float64}(I)

    # Compute parallax.
    avg_parallax = 0.0
    n_parallax = 0
    parallax_set = Float64[]

    # Compute parallax for all keypoints in previous KeyFrame.
    for keypoint in values(fe.current_frame.keypoints)
        only_2d && keypoint.is_3d && continue
        keypoint.id in keys(frame.keypoints) || continue

        # Compute parallax with undistorted pixel position.
        upx = keypoint.undistorted_pixel
        if compensate_rotation
            upx = project(frame.camera, current_rotation * keypoint.position)
        end

        frame_keypoint = frame.keypoints[keypoint.id]
        parallax = norm(upx - frame_keypoint.undistorted_pixel)
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

"""
Preprocess image before tracking.

Steps:
- Update previous image with current.
- TODO (Optionally) apply CLAHE.
- Update current image with `image`.
"""
function preprocess!(fe::FrontEnd, image)
    # TODO apply clahe to image

    # Update previous if tracking from Frame to Frame
    # and not from KeyFrame to Frame.
    # TODO update this, when KF->F is ready.
    fe.previous_image = fe.current_image
    fe.previous_pyramid = fe.current_pyramid

    fe.current_pyramid = ImageTracking.LKPyramid(
        image, fe.params.pyramid_levels,
    )
    fe.current_image = image
end

function klt_tracking!(fe::FrontEnd)
    priors = Point2f[]
    prior_ids = Int64[]
    prior_pixels = Point2f[]

    priors_3d = Point2f[]
    prior_3d_ids = Int64[]
    prior_3d_pixels = Point2f[]

    # Select points to track.
    for kp in values(fe.current_frame.keypoints)
        if !(fe.params.use_prior && kp.is_3d)
            # Init prior with previous pixel positions.
            push!(priors, kp.pixel)
            push!(prior_pixels, kp.pixel)
            push!(prior_ids, kp.id)
            continue
        end

        # If using prior, init pixel positions using motion model.
        # Projection in `(y, x)` format.
        projection = project_world_to_image_distort(
            fe.current_frame, fe.map_manager.map_points[kp.id].position,
        )
        in_image(fe.current_frame.camera, projection) || continue

        push!(priors_3d, projection)
        push!(prior_3d_pixels, kp.pixel)
        push!(prior_3d_ids, kp.id)
    end

    # First, track 3d keypoints, if using prior.
    if fe.params.use_prior && !isempty(priors_3d)
        # TODO allow passing displacement to tracking.
        # TODO construct displacement from priors_3d & prior_3d_pixels.
        new_keypoints, status = fb_tracking!(
            fe.previous_pyramid, fe.current_pyramid, prior_3d_pixels;
            pyramid_levels=1, window_size=fe.params.window_size,
            max_distance=fe.params.max_ktl_distance,
        )

        nb_good = 0
        for i in 1:length(new_keypoints)
            if status[i]
                update_keypoint!(
                    fe.current_frame, prior_3d_ids[i], new_keypoints[i],
                )
                nb_good += 1
            else
                # If failed, re-add keypoint to try with the full pyramid.
                push!(priors, priors_3d[i])
                push!(prior_pixels, prior_3d_pixels[i])
                push!(prior_ids, prior_3d_ids[i])
            end
        end
        # If motion model is wrong, require P3P next,
        # without using any priors.
        if nb_good < 0.33 * length(priors_3d)
            fe.p3p_required = true
            # TODO set displacements to 0 for the fb_tracking below.
            # TODO aka `priors = prior_pixels`.
        end
    end
    # Track other prior keypoints, if any.
    isempty(priors) && return

    new_keypoints, status = fb_tracking!(
        fe.previous_pyramid, fe.current_pyramid, prior_pixels;
        pyramid_levels=fe.params.pyramid_levels,
        window_size=fe.params.window_size,
        max_distance=fe.params.max_ktl_distance,
    )
    # Either update or remove current keypoints.
    nb_good = 0
    for i in 1:length(new_keypoints)
        if status[i] && in_image(fe.current_frame.camera, new_keypoints[i])
            update_keypoint!(fe.current_frame, prior_ids[i], new_keypoints[i])
            nb_good += 1
        else
            remove_obs_from_current_frame!(fe.map_manager, prior_ids[i])
        end
    end
    @debug "[FE] KLT $(nb_good)/$(length(new_keypoints)) inliers"
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
    fe.keyframe_pyramid = empty_pyr
    fe.previous_pyramid = empty_pyr
    fe.current_pyramid = empty_pyr
end
