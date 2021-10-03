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
    new_keypoints::AbstractVector{Point2f},
    previous_pyramid::ImageTracking.LKPyramid,
    current_pyramid::ImageTracking.LKPyramid,
    keypoints::AbstractVector{Point2f},
    algorithm::LucasKanade;
    displacement::AbstractVector{Point2f} = fill(Point2f(0.0, 0.0), length(keypoints)),
    max_distance::Real = 0.5,
)
    isempty(keypoints) && return

    # Forward tracking.
    displacement, status = ImageTracking.optflow!(
        previous_pyramid, current_pyramid, keypoints, displacement, algorithm)

    n_good = sum(status) # TODO return n_good from optflow!
    valid_ids = Vector{Int64}(undef, n_good) # Mapping to the original ids.
    valid_correspondences = Vector{Point2f}(undef, n_good)
    back_displacement = Vector{Point2f}(undef, n_good)

    scale = 1.0 / 2.0^algorithm.pyramid_levels
    c = 1
    @inbounds for i in 1:length(status)
        status[i] || continue

        old_kp, Δ = keypoints[i], displacement[i]
        new_point = old_kp .+ Δ
        new_keypoints[i] = new_point

        valid_correspondences[c] = new_point
        back_displacement[c] = -Δ .* scale
        valid_ids[c] = i
        c += 1
    end

    # Backward tracking.
    back_displacement, back_status = ImageTracking.optflow!(
        current_pyramid, previous_pyramid, valid_correspondences,
        back_displacement, algorithm)

    @inbounds for i in 1:length(back_status)
        idx = valid_ids[i]
        back_status[i] || (status[idx] = false; continue)

        new_point = valid_correspondences[i] .+ back_displacement[i]
        norm(keypoints[idx] .- new_point) ≥ max_distance &&
            (status[idx] = false; continue)
    end
    new_keypoints, status
end

function fb_tracking!(
    previous_pyramid::ImageTracking.LKPyramid,
    current_pyramid::ImageTracking.LKPyramid,
    keypoints::AbstractVector{Point2f};
    displacement::AbstractVector{Point2f} = fill(Point2f(0.0, 0.0), length(keypoints)),
    nb_iterations::Int = 30, window_size::Int = 11, pyramid_levels::Int = 3,
    max_distance::Real = 0.5,
)
    new_keypoints = Vector{Point2f}(undef, length(keypoints))
    algorithm = LucasKanade(nb_iterations; window_size, pyramid_levels)
    fb_tracking!(
        new_keypoints, previous_pyramid, current_pyramid,
        keypoints, algorithm; displacement, max_distance)
end
