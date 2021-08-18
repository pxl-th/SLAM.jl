mutable struct MapManager
    current_frame::Frame
    """
    KeyFrame id => Frame.
    """
    frames_map::Dict{Int64, Frame}

    params::Params
    extractor::Extractor

    map_points::Dict{Int64, MapPoint}
    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_keyframes::Int64
    nb_mappoints::Int64
end

MapManager(params::Params, frame::Frame, extractor::Extractor) = MapManager(
    frame, Dict{Int64, Frame}(), params, extractor,
    Dict{Int64, MapPoint}(), 0, 0, 0, 0,
)

function create_keyframe!(m::MapManager, image)
    @debug "[Map Manager] Creating new keyframe."
    prepare_frame!(m)
    extract_keypoints!(m, image)
    add_keyframe!(m)
end

function prepare_frame!(m::MapManager)
    m.current_frame.kfid = m.nb_keyframes

    # Filter if there are too many keypoints.
    # if m.current_frame.nb_keypoints > m.params.max_nb_keypoints
    #     # TODO
    # end

    for kp in get_keypoints(m.current_frame)
        if kp.id in keys(m.map_points)
            # Link new Keyframe to the MapPoint.
            mp = m.map_points[kp.id]
            add_keyframe_observation!(mp, m.current_keyframe_id)
        else
            remove_obs_from_current_frame!(m, kp.id)
        end
    end
end

function extract_keypoints!(m::MapManager, image)
    nb_2_detect = m.params.max_nb_keypoints - m.current_frame.nb_occupied_cells
    nb_2_detect ≤ 0 && return
    # Detect keypoints in the provided `image`
    # using current keypoints to set a mask of regions
    # to avoid detecting features in.
    current_points = [kp.pixel for kp in values(m.current_frame.keypoints)]
    keypoints = detect(m.extractor, image, current_points)
    isempty(keypoints) && return

    descriptors, keypoints = describe(m.extractor, image, keypoints)
    add_keypoints_to_frame!(m, m.current_frame, keypoints, descriptors)
end

function add_keypoints_to_frame!(
    m::MapManager, frame::Frame, keypoints, descriptors,
)
    for (kp, dp) in zip(keypoints, descriptors)
        # m.current_mappoint_id is incremented in `add_mappoint!`.
        add_keypoint!(frame, Point2f(kp[1], kp[2]), m.current_mappoint_id)
        add_mappoint!(m, dp)
    end
end

function add_mappoint!(m::MapManager, descriptor)
    new_mappoint = MapPoint(
        m.current_mappoint_id, m.current_keyframe_id, descriptor,
    )
    m.map_points[m.current_mappoint_id] = new_mappoint
    m.current_mappoint_id += 1
    m.nb_mappoints += 1
end

"""
Copy current MapManager's Frame and add it to the KeyFrame map.
Increase current keyframe id & total number of keyframes.
"""
function add_keyframe!(m::MapManager)
    m.frames_map[m.current_keyframe_id] = m.current_frame |> deepcopy
    m.current_keyframe_id += 1
    m.nb_keyframes += 1
end

"""
Remove a MapPoint observation from current Frame by `id`.
"""
function remove_obs_from_current_frame!(m::MapManager, id::Int64)
    remove_keypoint!(m.current_frame, id)
    # TODO visualization related part. Point-cloud point removal.
    # Set MapPoint as not observable.
    # if !(id in keys(m.map_points))
    #     # TODO Reset point in point-cloud to origin-point.
    #     return
    # end
end

"""
Remove KeyFrame observation from MapPoint.
"""
function remove_mappoint_obs!(m::MapManager, kpid::Int, kfid::Int)
    # Remove MapPoint from KeyFrame.
    frame_exists = kfid in keys(m.frames_map)
    frame_exists && remove_keypoint!(m.frames_map[kfid], kpid)
    # Remove KeyFrame observation from MapPoint.
    kpid in keys(m.map_points) || return
    mappoint = m.map_points[kpid]
    remove_kf_observation!(mappoint, kfid)
    
    frame_exists || return
    frame = m.frames_map[kfid]
    for observer_id in mappoint.observer_keyframes_ids
        observer_id in keys(m.frames_map) || continue
        decrease_covisible_kf!(frame, observer_id)
        decrease_covisible_kf!(m.frames_map[observer_id], kfid)
    end
end

"""
Update position of a MapPoint.
"""
function update_mappoint!(m::MapManager, kpid::Int, new_position, inv_depth)
    kpid in keys(m.map_points) || return
    mp = m.map_points[kpid]
    # If MapPoint is 2D, turn it to 3D and update its observing KeyFrames.
    if !mp.is_3d
        for observer_id in mp.observer_keyframes_ids
            if observer_id in keys(m.frames_map)
                turn_keypoint_3d!(m.frames_map[observer_id], kpid)
            else
                remove_kf_observation!(mp, observer_id)
            end
        end
        mp.is_observed && turn_keypoint_3d!(m.current_frame, kpid)
    end
    # Update world position.
    set_position!(mp, new_position, inv_depth ≥ 0 ? inv_depth : -1)
end
