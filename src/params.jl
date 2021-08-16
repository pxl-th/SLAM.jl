@with_kw mutable struct Params
    max_nb_keypoints::Int = 1000
    """
    Cell size of the grid in the Frame.
    """
    max_distance::Int = 50
    """
    Maximum distance a point can shift to be considered inlier in KTL tracking.
    """
    max_ktl_distance::Real = 0.5
    """
    Number of pyramid levels to construct.
    0 to use original image only.
    Otherwise, total number of levels is L + 1.
    """
    pyramid_levels::Int = 3
    """
    Size of the window for the KTL tracking.
    Actual size is `2 * S + 1`.
    """
    window_size::Int = 21
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
    initial_parallax::Real = 10.0
    """
    Maximum allowed reprojection error.
    Used during triangulation in Mapper to discard outliers.
    """
    max_reprojection_error = 3.0
end
