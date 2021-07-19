struct Params
    max_nb_keypoints::Int64
    """
    Cell size of the grid in the Frame.
    """
    max_distance::Int64
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
