"""
Parameters of the system.

# Arguments:

- `stereo::Bool`: Set to `false` if in monocular mode, otherwise `true`.
    Default is `false`.
- `max_nb_keypoints::Int64`: Maximum number of keypoints to detect in the image.
    Default is `1000`.
- `max_distance::Int64`: Cell size in pixels of the grid in Frame.
    Each frame is divided into a grid, where each cell contains certain
    number of keypoints. This grid is used when retrieving neighbouring
    keypoints from the Frame.
- `max_ktl_distance::Float64`: When doing optical flow tracking, it is done
    by first tracking keypoint from previous frame to the current frame and
    then in reverse. If the distance between keypoint in previous frame and
    reverse tracked is greater than this distance, then that keypoint is
    considered lost and discarded. Default is `1.0`.
- `pyramid_levels::Int64`: Number of pyramid levels to use during optical flow.
    Set to `0` to use only original image. Otherwise if set to `L`, then total
    number of levels is `L + 1`. Default is `3`.
- `pyramid_σ::Float64`: `σ` for the Gaussian blur, used when constructing
    pyramids. Default is `1.0`.
- `window_size::Int64`: Size of the window to use when doing optical flow
    tracking. Default is `9`.
- `initial_parallax::Float64`: Average amount of the parallax needed
    between the frames for the initialization. Default is `20.0`.
- `max_reprojection_error::Float64`: Maximum reprojection error for the
    mappoint to be considered inlier. It is used in triangulation
    and pose calculation. Default is `3.0`.
- `min_cov_score::Int64`: Minimum number of 3D MapPoints required
    to perform Bundle-Adjustment. Default is `25`.
- `filtering_ratio::Float64`:
    Number of good keypoints divided by the total number of keypoints
    in a Frame. This is used to filter out non-informative Frames.
    If frame ratio is greater than this value, then it is removed during
    map filtering. Default is `0.9`.
- `map_filtering::Bool`: Set to `true` to perform local map matching.
    The goal of this is to try to re-match lost 3D keypoints back into
    the current frame.
- `max_projection_distance::Float64`: Maximum distance between projected
    and target pixels to consider keypoints being "the same".
    Is is used during local map matching. Default is `2.0`.
- `max_descriptor_distance::Float64`: Maximum distance ratio between
    descriptors to be considered similar. It is used during local map matching.
    Default is `0.35`. The lesser the value, the more accurate matching will be.
    Should be in `(0, 1)` range.

**Below variables are part of the SLAM state and should not change manually.**

- `vision_initialized::Bool`: If `true`, then the Front-End was successfully
    initialized.
- `reset_required::Bool`: Indicates if the visual part requires reset.
    This could happen if the system is tracking too little points.
- `local_ba_on::Bool`: `true`, when Estimator is performing
    Local Bundle Adjustment.
"""
Base.@kwdef mutable struct Params
    stereo::Bool = false
    max_nb_keypoints::Int64 = 1000
    max_distance::Int64 = 35
    max_ktl_distance::Float64 = 1.0
    pyramid_levels::Int64 = 3
    pyramid_σ::Float64 = 1.0
    window_size::Int64 = 9
    initial_parallax::Float64 = 20.0
    max_reprojection_error::Float64 = 3.0
    min_cov_score::Int64 = 25
    filtering_ratio::Float64 = 0.9
    do_local_matching::Bool = false
    map_filtering::Bool = false
    max_projection_distance::Float64 = 2.0
    max_descriptor_distance::Float64 = 0.35

    # State variables.
    vision_initialized::Bool = false
    reset_required::Bool = false
    local_ba_on::Bool = false
end

"""
```julia
reset!(p::Params)
```

Reset state of the system.
"""
function reset!(p::Params)
    p.vision_initialized = false
    p.reset_required = false
end
