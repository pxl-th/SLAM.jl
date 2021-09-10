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
    @debug "[Map Manager] Creating new keyframe $(m.current_keyframe_id)."
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

    n_added = 0
    n_removed = 0
    for kp in values(m.current_frame.keypoints)
        if kp.id in keys(m.map_points)
            # Link new Keyframe to the MapPoint.
            mp = m.map_points[kp.id]
            add_keyframe_observation!(mp, m.current_keyframe_id)
            n_added += 1
        else
            remove_obs_from_current_frame!(m, kp.id)
            n_removed += 1
        end
    end
    @debug "[MM] Added total KF observs $n_added"
    @debug "[MM] Removed total KF observs $n_removed"
end

function extract_keypoints!(m::MapManager, image)
    nb_2_detect = m.params.max_nb_keypoints - m.current_frame.nb_occupied_cells
    if nb_2_detect ≤ 0
        @debug "[MM] No need to extract more KPs"
        return
    end
    # Detect keypoints in the provided `image`
    # using current keypoints to set a mask of regions
    # to avoid detecting features in.
    current_points = [kp.pixel for kp in values(m.current_frame.keypoints)]
    @debug "[MM] Before extraction $(length(current_points)) keypoints"
    keypoints = detect(m.extractor, image, current_points)
    isempty(keypoints) && return
    @debug "[MM] Extracted $(length(keypoints)) keypoints"

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
Remove MapPoint from the map given its id.

Removing a mappoint, also update covisibility scores of the observer Frames.
If MapPoint is observed by the current Frame, remove its keypoint as well.
"""
function remove_mappoint!(m::MapManager, id)
    id in keys(m.map_points) || return
    mp = m.map_points[id]
    for observer_id in mp.observer_keyframes_ids
        observer_id in keys(m.frames_map) || continue
        observer_kf = m.frames_map[observer_id]

        remove_keypoint!(observer_kf, id)
        for co_observer_id in mp.observer_keyframes_ids
            observer_id == co_observer_id && continue
            decrease_covisible_kf!(observer_kf, co_observer_id)
        end
    end

    mp.is_observed && remove_keypoint!(m.current_frame, id)
    mp.is_3d && (m.nb_mappoints -= 1;)
    pop!(m.map_points, id)
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

function remove_keyframe!(m::MapManager, kfid)
    kfid in keys(m.frames_map) || return

    kf = m.frames_map[kfid]
    for kp in get_keypoints(kf)
        kp.id in keys(m.map_points) &&
            remove_kf_observation!(m.map_points[kp.id], kfid)
    end

    for cov_kfid in keys(kf.covisible_kf)
        cov_kfid in keys(m.frames_map) &&
            remove_covisible_kf!(m.frames_map[cov_kfid], kfid)
    end

    pop!(m.frames_map, kfid)
    m.nb_keyframes -= 1
end

"""
Remove a MapPoint observation from current Frame by `id`.
"""
function remove_obs_from_current_frame!(m::MapManager, id::Int64)
    remove_keypoint!(m.current_frame, id)
    # TODO visualization related part. Point-cloud point removal.
    # Set MapPoint as not observable.
    if id in keys(m.map_points)
        m.map_points[id].is_observed = false
    else
        # TODO Reset point in visualization point-cloud to origin.
    end
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
        if mp.is_observed
            # Because we deepcopy frame before putting it to the frames_map,
            # we need to update current frame as well.
            # Which should also update current frame in the FrontEnd.
            turn_keypoint_3d!(m.current_frame, kpid)
        end
    end
    # Update world position.
    set_position!(mp, new_position, inv_depth ≥ 0 ? inv_depth : -1)
end

"""
Update MapPoints and covisible graph between KeyFrames.
"""
function update_frame_covisibility!(m::MapManager, frame::Frame)
    covisible_keyframes = Dict{Int64, Int64}()
    local_map_ids = Set{Int64}()
    # For each Keypoint in the `frame`, get its corresponding MapPoint.
    # Get the set of KeyFrames that observe this MapPoint.
    # Add them to the covisible map, which contains all KeyFrames
    # that share visibility with the `frame`.
    for kp in get_keypoints(frame)
        if !(kp.id in keys(m.map_points))
            remove_mappoint_obs!(m, kp.id, frame.kfid)
            remove_obs_from_current_frame!(m, kp.id)
            continue
        end
        mp = m.map_points[kp.id]
        # Get the set of KeyFrames observing this KeyFrame to update covisibility.
        for kfid in mp.observer_keyframes_ids
            kfid == frame.kfid && continue
            if kfid in keys(covisible_keyframes)
                covisible_keyframes[kfid] += 1
            else
                covisible_keyframes[kfid] = 1
            end
        end
    end
    # Update covisibility for covisible KeyFrames.
    # For each entry in the covisible map, get its corresponding KeyFrame.
    # Update the covisible score for the `frame` in it.
    # Add all 3D Keypoints that are not in the `frame`
    # to the local map for future tracking.
    bad_kfids = Set{Int64}()
    for (kfid, cov_score) in covisible_keyframes
        if !(kfid in keys(m.frames_map))
            push!(bad_kfids, kfid)
            continue
        end
        cov_frame = m.frames_map[kfid]
        cov_frame.covisible_kf[frame.kfid] = cov_score
        for kp in get_3d_keypoints(cov_frame)
            kp.id in keys(frame.keypoints) || push!(local_map_ids, kp.id)
        end
    end
    for bad_kfid in bad_kfids
        pop!(covisible_keyframes, bad_kfid)
    end
    # Update the set of covisible KeyFrames.
    frame.covisible_kf = covisible_keyframes
    # Update local map of unobserved MapPoints.
    if length(local_map_ids) > 0.5 * length(frame.local_map_ids)
        frame.local_map_ids = local_map_ids
    else
        union!(frame.local_map_ids, local_map_ids)
    end
end

function reset!(m::MapManager)
    m.nb_keyframes = 0
    m.nb_mappoints = 0
    m.current_keyframe_id = 0
    m.current_mappoint_id = 0

    m.map_points |> empty!
    m.frames_map |> empty!
end
