Base.@kwdef mutable struct Params
    stereo::Bool = false
    max_nb_keypoints::Int64 = 1000
    """
    Cell size of the grid in the Frame.
    """
    max_distance::Int64 = 35
    """
    Maximum distance a point can shift to be considered inlier in KTL tracking.
    """
    max_ktl_distance::Float64 = 1.0
    """
    Number of pyramid levels to construct.
    0 to use original image only.
    Otherwise, total number of levels is L + 1.
    """
    pyramid_levels::Int64 = 3
    pyramid_σ::Float64 = 1.0
    """
    Size of the window for the KTL tracking.
    Actual size is `2 * S + 1`.
    """
    window_size::Int64 = 9
    """
    Use prior motion model to estimate subsequent position of keypoints
    in the next frame.
    """
    use_prior::Bool = true
    """
    Whether or not to undistort each frame.
    """
    do_undistort::Bool = false
    """
    Boolean that indicated if the vision (front-end) tracking was
    successfully initialized. Initially it is `false`.
    """
    vision_initialized::Bool = false
    """
    Indicates if the visual part requires reset.
    This could happen if the system is tracking too little points.
    """
    reset_required::Bool = false
    """
    Amount of parallax (in pixels) needed for initialization system to kick in.
    """
    initial_parallax::Float64 = 20.0
    """
    Maximum allowed reprojection error.
    Used during triangulation in Mapper to discard outliers.
    """
    max_reprojection_error::Float64 = 3.0
    """
    Whether to do P3P for pose estimation
    or doing only Bundle Adjustment for pose.
    """
    do_p3p::Bool = true
    """
    Minimum number of 3D MapPoints required to perform Bundle-Adjustment.
    """
    min_cov_score::Int64 = 25
    """
    Number of good keypoints divided by the total number of keypoints in a Frame.
    This is used to filter out non-informative Frames.
    """
    filtering_ratio::Float64 = 0.9
    """
    Indicates whether system is performing Local Bundle-Adjustment now.
    """
    local_ba_on::Bool = false
    do_local_matching::Bool = false
    """
    Maximum distance between projected and target pixels.
    Used during matching to local map.
    """
    max_projection_distance::Float64 = 2.0
    max_descriptor_distance::Float64 = 0.35
end

function reset!(p::Params)
    p.vision_initialized = false
    p.reset_required = false
end
