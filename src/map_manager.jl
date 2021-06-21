struct MapManager
    current_frame::Frame
    map_frames::Dict{Int64, Frame}

    nb_keyframes::Int64

    # params::Params
    # extractor::Extractor
    # tracker::Tracker

    # map_points::Dict{Int64, MapPoint}
end

function create_keyframe(m::MapManager, image)
    # prepare_frame
    # extract_keypoints
    # add_keyframe
end

function prepare_frame(m::MapManager)
    m.current_frame.kfid = m.nb_keyframes

    # Filter if there are too many keypoints.
    # if m.current_frame.nb_keypoints > m.params.max_nb_keypoints
    #     # TODO
    # end

    for kp in get_keypoints(m.current_frame)
        # Get related MapPoint.
        # if kp is not in the map (map_plms), then remove it
        # else link it to the current frame kfid = m.nb_keyframes
    end
end

function extract_keypoints(m::MapManager, image)
    keypoints = m.current_frame |> get_keypoints
    current_points = [kp.pixel for kp in keypoints]

    # describe keypoints if using brief

    nb_2_detect = m.params.max_nb_keypoints - m.current_frame.nb_occupied_cells
    if nb_2_detect > 0
        # Detect keypoints in the provided `image` using current keypoints
        # and roi to set a mask.
        new_points = detect_grid_fast(
            image, params.max_distance, current_points, roi,
        )
    end
end
