mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    # feature tracker
    map_manager::MapManager
    params::Params

    current_image::Matrix{Gray}
    previous_image::Matrix{Gray}
end

FrontEnd(frame::Frame, params::Params, map_manager::MapManager) = FrontEnd(
    frame, MotionModel(), map_manager, params,
    Matrix{Gray}(undef, 0, 0), Matrix{Gray}(undef, 0, 0),
)

function track(fe::FrontEnd, image, time)
    is_kf_required = track_mono(fe, image, time)
    if is_kf_required
        create_keyframe!(fe.map_manager, image)
        # TODO build optial flow pyramid and reuse it in optical flow
    end
end

function track_mono(fe::FrontEnd, image, time)::Bool
    preprocess(fe, image)
    # If it's the first frame, then KeyFrame is always needed.
    fe.current_frame.id == 1 && return true
    # Apply motion model & update current Frame pose.
    set_wc!(fe.current_frame, fe.motion_model(fe.current_frame.wc, time))
    # track new image
    # epipolar filtering
    # compute pose 2d-3d
    # update motion model
    # check if new kf required
end

"""
Preprocess image before tracking.

Steps:
- Update previous image with current.
- TODO (Optionally) apply CLAHE.
- Update current image with `image`.
"""
function preprocess(fe::FrontEnd, image)
    # if track frame-to-frame
    fe.previous_image = fe.current_image
    # TODO apply clahe
    fe.current_image = image
end

function ktl_tracking(fe::FrontEnd)
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

    @debug "[Front-End] N Priors: $(length(priors))"

    # First, track 3d keypoints, if using prior.
    # Track other prior keypoints, if any.
end
