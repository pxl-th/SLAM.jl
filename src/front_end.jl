mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    map_manager::MapManager
    params::Params

    current_image::Matrix{Gray}
    previous_image::Matrix{Gray}
end

FrontEnd(params::Params, frame::Frame, map_manager::MapManager) = FrontEnd(
    frame, MotionModel(), map_manager, params,
    Matrix{Gray}(undef, 0, 0), Matrix{Gray}(undef, 0, 0),
)

function track(fe::FrontEnd, image, time)
    is_kf_required = track_mono(fe, image, time)
    @debug "[Front-End] Is KF required $is_kf_required @ $time"
    if is_kf_required
        create_keyframe!(fe.map_manager, image)
        # TODO build optial flow pyramid and reuse it in optical flow
    end

    ni = image |> copy .|> RGB
    draw_keypoints!(ni, fe.current_frame)
    save("/home/pxl-th/projects/frame-$time.png", ni)
end

function track_mono(fe::FrontEnd, image, time)::Bool
    preprocess!(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    @debug "[Front End] Current Frame id $(fe.current_frame.id), kfid $(fe.current_frame.kfid)"
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    new_wc = fe.motion_model(fe.current_frame.wc, time)
    @debug "[Front End] New wc $new_wc"
    set_wc!(fe.current_frame, new_wc)
    # track new image
    fe |> ktl_tracking!
    # epipolar filtering
    if !fe.params.vision_initialized
        if fe.current_frame.nb_keypoints < 50
            fe.params.reset_required = true
            return false
        elseif check_ready_for_init(fe)
            @debug "[Front End] System ready for initialization."
            fe.params.vision_initialized = true
            return true
        else
            @debug "[Front End] System not ready for initialization."
            return false
        end
    end
    # compute pose 2d-3d
    # update motion model
    false # TODO check if new kf required
end

"""
Check if there is enough average rotation compensated parallax
between current Frame and previous KeyFrame.

Additionally, compute Essential matrix using 5-point Ransac algorithm
to filter out outliers and check if there is enough inliers to proceed.
"""
function check_ready_for_init(fe::FrontEnd)
    # Get previous keyframe.
    fe.current_frame.kfid in keys(fe.map_manager.frames_map) || return false

    avg_parallax = compute_parallax(
        fe, fe.current_frame.kfid; compensate_rotation=false,
    )
    @debug "[Front-End] Init Avg Parallax: $avg_parallax"
    avg_parallax ≤ fe.params.initial_parallax && return false

    previous_keyframe = fe.map_manager.frames_map[fe.current_frame.kfid]
    # Ensure there are enough keypoints for the essential matrix computation.
    if fe.current_frame.nb_keypoints < 8
        @debug "[Front-End] Not enough keypoints for initialization: " *
            "$(fe.current_frame.nb_keypoints)"
        return false
    end

    # Setup Essential matrix computation.
    R_compensation = get_Rcw(previous_keyframe) * get_Rwc(fe.current_frame)
    @debug "[Front-End] R_compensation:"
    display(R_compensation); println()

    n_parallax = 0
    avg_parallax = 0.0

    previous_points = Point2f[]
    current_points = Point2f[]
    kp_ids = Int[]

    # Get all Keypoint positions and compute rotation-compensated parallax.
    for keypoint in values(fe.current_frame.keypoints)
        keypoint.id in keys(previous_keyframe.keypoints) || continue

        pkf_keypoint = previous_keyframe.keypoints[keypoint.id]
        push!(previous_points, pkf_keypoint.undistorted_pixel)
        push!(current_points, keypoint.undistorted_pixel)
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
    if (avg_parallax < fe.params.initial_parallax)
        @debug "[Front-End] Not enough parallax ($avg_parallax) " *
            "to compute 5pt Essential Matrix."
        return false
    end
    @debug "[Front-End] 5pt parallax $avg_parallax @ $n_parallax points."

    # Pass points in `(y, x)` format and camera intrinsics.
    # TODO migrate to ransac 5pt when it is ready.
    E, P, inliers, n_inliers = five_point(
        previous_points, current_points,
        fe.current_frame.camera.K,
        fe.current_frame.camera.K,
    )
    if n_inliers < 5
        @debug "[Front-End] Not enough inliers ($n_inliers) for the " *
            "5pt Essential Matrix."
        return false
    end

    @debug "[Front-End] 5pt N inliers $n_inliers."
    @debug "[Front-End] 5pt N solutions $(length(P))"
    display(E[1]); println()
    display(P[1]); println()

    # Remove outliers from the current frame.
    for (i, inlier) in enumerate(inliers)
        inlier && continue
        remove_obs_from_current_frame!(fe.map_manager, kp_ids[i])
    end
    
    # TODO arbitrary scale for translation.
    # TODO inverse transformation.
    # TODO set transformation to current frame
    # TODO return true
    # TODO move to 3x4 pose matrices
    false
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

    current_rotation = SMatrix{3, 3, Float64}(I)
    compensate_rotation &&
        (current_rotation = frame.getRcw() * fe.current_frame.getRwc();)

    # Compute parallax.
    avg_parallax = 0.0
    n_parallax = 0
    parallax_set = Set{Float64}()

    # Compute parallax for all keypoints in previous KeyFrame.
    for keypoint in values(fe.current_frame.keypoints)
        only_2d && keypoint.is_3d && continue
        keypoint.id in keys(frame.keypoints) || continue

        # Compute parallax with undistorted pixel position.
        undistorted_pixel = keypoint.undistorted_pixel
        if compensate_rotation
            undistorted_pixel = project(
                frame.camera, current_rotation * keypoint.position,
            )
        end

        frame_keypoint = frame.keypoints[keypoint.id]
        parallax = norm(undistorted_pixel - frame_keypoint.undistorted_pixel)
        avg_parallax += parallax
        n_parallax += 1

        median_parallax && push!(parallax_set, parallax)
    end

    @debug "[Front-End] Number parallax $n_parallax"
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
    # if track frame-to-frame
    fe.previous_image = fe.current_image
    # TODO apply clahe
    fe.current_image = image
end

function ktl_tracking!(fe::FrontEnd)
    @debug "[Front-End] KTL Tracking"

    priors = Point2f[]
    prior_ids = Int64[]
    prior_pixels = Point2f[]

    priors_3d = Point2f[]
    prior_3d_ids = Int64[]
    prior_3d_pixels = Point2f[]

    # Select points to track.
    for (_, kp) in fe.current_frame.keypoints
        if !fe.params.use_prior || !kp.is_3d
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
    @debug "[Front-End] N 3D Priors: $(length(priors_3d))"
    @debug "[Front-End] N Priors: $(length(priors))"

    # First, track 3d keypoints, if using prior.
    if fe.params.use_prior && !isempty(priors_3d)
        # TODO allow passing displacement to tracking.
        # TODO construct displacement from priors_3d & prior_3d_pixels.
        new_keypoints, status = fb_tracking(
            fe.previous_image, fe.current_image, prior_3d_pixels;
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
        @debug "[Front End] Prior 3D tracking: $nb_good / $(length(priors_3d))."
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

    new_keypoints, status = fb_tracking(
        fe.previous_image, fe.current_image, prior_pixels;
        pyramid_levels=fe.params.pyramid_levels,
        window_size=fe.params.window_size,
        max_distance=fe.params.max_ktl_distance,
    )
    # Either update or remove current keypoints.
    nb_good = 0
    for i in 1:length(new_keypoints)
        if status[i]
            update_keypoint!(fe.current_frame, prior_ids[i], new_keypoints[i])
            nb_good += 1
        else
            remove_obs_from_current_frame!(fe.map_manager, prior_ids[i])
        end
    end
    @debug "[Front End] Tracking no prior: $nb_good / $(length(priors))."
end
