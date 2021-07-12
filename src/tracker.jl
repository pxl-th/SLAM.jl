"""
Perform Forward-Backward Lucas-Kanade optical flow tracking.

Params:
    keypoints::Vector{Point2f}
        Vector of 2d keypoints in `(row, col)` format which to track.
"""
function fb_tracking(
    previous_image, current_image, keypoints;
    nb_iterations::Int = 30, window_size::Int = 11, pyramid_levels::Int = 3,
    max_distance::Real = 0.5,
)
    new_keypoints = Vector{SVector{2, Float64}}(undef, length(keypoints))

    algorithm = LucasKanade(nb_iterations; window_size, pyramid_levels)
    previous_pyramid = ImageTracking.LKPyramid(previous_image, pyramid_levels)
    current_pyramid = ImageTracking.LKPyramid(current_image, pyramid_levels)

    fb_tracking!(
        new_keypoints, previous_pyramid, current_pyramid, keypoints,
        algorithm; max_distance,
    )
end

"""
Perform Forward-Backward tracking by first tracking keypoints from
`previous_pyramid` to `current_pyramid` and then track resuling keypoints
in reverse order.
Then filter out those forward-backward keypoints that are too far away
from the original keypoints to be consistent.

# Arguments:

- `new_keypoints`:
    Vector where to write resulting new poisitions of keypoints.
    Should be of the same size as `keypoints` vector.
- `max_distance::Real`:
    Maximum distance in pixels between Forward-Backward tracked keypoints
    to be considered the same keypoint.
"""
function fb_tracking!(
    new_keypoints::Vector{SVector{2, Float64}},
    previous_pyramid::ImageTracking.LKPyramid,
    current_pyramid::ImageTracking.LKPyramid,
    keypoints::Vector{SVector{2, Float64}},
    algorithm::LucasKanade;
    max_distance::Real = 0.5,
)
    isempty(keypoints) && return

    # Forward tracking.
    flow = fill(SVector{2}(0.0, 0.0), length(keypoints))
    flow, status = ImageTracking.optflow!(
        previous_pyramid, current_pyramid, keypoints, flow, algorithm,
    )

    valid_correspondences = Vector{SVector{2, Float64}}(undef, sum(status))
    valid_ids = Dict{Int32, Int32}() # Mapping to the original points ids.

    nb_good = 0
    for i in 1:length(status)
        status[i] || continue

        new_point = keypoints[i] .+ flow[i]
        new_keypoints[i] = new_point

        nb_good += 1
        valid_correspondences[nb_good] = new_point
        valid_ids[nb_good] = i
    end

    # Backward tracking.
    back_flow = fill(SVector{2}(0.0, 0.0), length(valid_correspondences))
    back_flow, back_status = ImageTracking.optflow!(
        current_pyramid, previous_pyramid, valid_correspondences,
        back_flow, algorithm,
    )
    for i in 1:length(back_status)
        idx = valid_ids[i]
        back_status[i] || (status[idx] = false; continue)

        new_point = valid_correspondences[i] .+ back_flow[i]
        norm(keypoints[idx] .- new_point) ≥ max_distance &&
            (status[idx] = false; continue)
    end

    new_keypoints, status
end
