mutable struct FrontEnd
    current_frame::Frame
    motion_model::MotionModel
    # feature tracker
    map_manager::MapManager

    current_image::Matrix{Gray}
    previous_image::Matrix{Gray}
end

FrontEnd(frame::Frame) = FrontEnd(
    frame, MotionModel(),
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
    set_transformation!(
        fe.current_frame, fe.motion_model(fe.current_frame.wc, time),
    )
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
