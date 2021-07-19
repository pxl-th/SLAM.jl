struct Params
    max_nb_keypoints::Int64
    """
    Cell size of the grid in the Frame.
    """
    max_distance::Int64
    """
    Maximum distance a point can shift to be considered inlier in KTL tracking.
    """
    max_ktl_distance::Float64
    """
    Number of pyramid levels to construct.
    0 to use original image only.
    Otherwise, total number of levels is L + 1.
    """
    pyramid_levels::Int64
    """
    Size of the window for the KTL tracking.
    Actual size is `2 * S + 1`.
    """
    window_size::Int64
    """
    Use prior motion model to estimate subsequent position of keypoints
    in the next frame.
    """
    use_prior::Bool
    """
    Whether or not to undistort each frame.
    """
    do_undistort::Bool
    # intrinsics & extrinsics for stereo
end
