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
end

function track_mono(fe::FrontEnd, image, time)::Bool
    preprocess!(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    @debug "[Front End] Current Frame id $(fe.current_frame.id)"
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    set_wc!(fe.current_frame, fe.motion_model(fe.current_frame.wc, time))
    # track new image
    fe |> ktl_tracking!
    # epipolar filtering
    # compute pose 2d-3d
    # update motion model
    false # TODO check if new kf required
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

    for (_, kp) in fe.current_frame.keypoints
        if !fe.params.use_prior
            # Init prior with previous pixel positions.
            push!(priors, kp.pixel)
            push!(prior_pixels, kp.pixel)
            push!(prior_ids, kp.id)
            continue
        end

        # If using prior, init pixel positions using motion model.
        kp.is_3d || continue
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
        # If motion model is quite wrong, require P3P next,
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
