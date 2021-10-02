mutable struct MapManager
    """
    Current frame that is shared throughout all the components in the system.
    """
    current_frame::Frame
    """
    KeyFrame id → Frame.
    """
    frames_map::Dict{Int64, Frame}

    params::Params
    extractor::Extractor
    """
    MapPoint id → MapPoint.
    """
    map_points::Dict{Int64, MapPoint}
    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_keyframes::Int64
    nb_mappoints::Int64

    mappoint_lock::ReentrantLock
    keyframe_lock::ReentrantLock
    map_lock::ReentrantLock
end

function MapManager(params::Params, frame::Frame, extractor::Extractor)
    mappoint_lock = ReentrantLock()
    keyframe_lock = ReentrantLock()
    map_lock = ReentrantLock()

    MapManager(
        frame, Dict{Int64, Frame}(), params, extractor,
        Dict{Int64, MapPoint}(), 0, 0, 0, 0,
        mappoint_lock, keyframe_lock, map_lock)
end

function get_keyframe(m::MapManager, kfid)
    lock(m.keyframe_lock) do
        get(m.frames_map, kfid, nothing)
    end
end

function has_keyframe(m::MapManager, kfid)
    lock(m.keyframe_lock) do
        kfid in keys(m.frames_map)
    end
end

function get_mappoint(m::MapManager, mpid)
    lock(m.mappoint_lock) do
        get(m.map_points, mpid, nothing)
    end
end

function create_keyframe!(m::MapManager, image)
    @debug "[MM] Creating new keyframe $(m.current_keyframe_id)."
    prepare_frame!(m)
    extract_keypoints!(m, image)
    add_keyframe!(m)
end

function prepare_frame!(m::MapManager)
    m.current_frame.kfid = m.current_keyframe_id
    @info "[MM] Adding KF $(m.current_frame.kfid) to Map."

    # Filter if there are too many keypoints.
    # if m.current_frame.nb_keypoints > m.params.max_nb_keypoints
    #     # TODO
    # end

    for kp in values(m.current_frame.keypoints)
        mp = get(m.map_points, kp.id, nothing)
        if mp ≡ nothing
            remove_obs_from_current_frame!(m, kp.id)
        else
            add_keyframe_observation!(mp, m.current_keyframe_id)
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

    add_keypoints_to_frame!(m, m.current_frame, keypoints)

    vimage = RGB{Float64}.(image)
    draw_keypoints!(vimage, m.current_frame)
    save("/home/pxl-th/projects/slam-data/images/frame-$(m.current_frame.id).png", vimage)
end

function add_keypoints_to_frame!(m::MapManager, frame, keypoints)
    lock(m.mappoint_lock) do
        for kp in keypoints
            # m.current_mappoint_id is incremented in `add_mappoint!`.
            add_keypoint!(frame, Point2f(kp[1], kp[2]), m.current_mappoint_id)
            add_mappoint!(m)
        end
    end
end

@inline function add_mappoint!(m::MapManager)
    mp = MapPoint(m.current_mappoint_id, m.current_keyframe_id)
    m.map_points[m.current_mappoint_id] = mp
    m.current_mappoint_id += 1
    m.nb_mappoints += 1
end

"""
Remove MapPoint from the map given its id.

Removing a mappoint, also update covisibility scores of the observer Frames.
If MapPoint is observed by the current Frame, remove its keypoint as well.
"""
function remove_mappoint!(m::MapManager, id)
    lock(m.keyframe_lock)
    lock(m.mappoint_lock)

    if !(id in keys(m.map_points))
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

    mp = m.map_points[id]
    for observer_id in mp.observer_keyframes_ids
        observer_kf = get(m.frames_map, observer_id, nothing)
        observer_kf ≡ nothing && continue

        remove_keypoint!(observer_kf, id)
        for co_observer_id in mp.observer_keyframes_ids
            observer_id == co_observer_id && continue
            decrease_covisible_kf!(observer_kf, co_observer_id)
        end
    end

    mp.is_observed && remove_keypoint!(m.current_frame, id)
    mp.is_3d && (m.nb_mappoints -= 1;)
    pop!(m.map_points, id)

    unlock(m.keyframe_lock)
    unlock(m.mappoint_lock)
end

"""
Copy current MapManager's Frame and add it to the KeyFrame map.
Increase current keyframe id & total number of keyframes.
"""
function add_keyframe!(m::MapManager)
    new_keyframe = m.current_frame |> deepcopy

    lock(m.keyframe_lock) do
        m.frames_map[m.current_keyframe_id] = new_keyframe
        m.current_keyframe_id += 1
        m.nb_keyframes += 1
    end
end

function remove_keyframe!(m::MapManager, kfid)
    lock(m.keyframe_lock)
    lock(m.mappoint_lock)

    kf = get(m.frames_map, kfid, nothing)
    if kf ≡ nothing
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

    for kp in get_keypoints(kf)
        mp = get(m.map_points, kp.id, nothing)
        mp ≢ nothing && remove_kf_observation!(mp, kfid)
    end
    for cov_kfid in keys(kf.covisible_kf)
        cov_kf = get(m.frames_map, cov_kfid, nothing)
        cov_kf ≢ nothing && remove_covisible_kf!(cov_kf, kfid)
    end

    pop!(m.frames_map, kfid)
    m.nb_keyframes -= 1

    unlock(m.mappoint_lock)
    unlock(m.keyframe_lock)
end

"""
Remove a MapPoint observation from current Frame by `id`.
"""
function remove_obs_from_current_frame!(m::MapManager, id::Int64)
    remove_keypoint!(m.current_frame, id)
    # Set MapPoint as not observable.
    mp = get(m.map_points, id, nothing)
    mp ≢ nothing && (mp.is_observed = false;)
end

"""
Remove KeyFrame observation from MapPoint.
"""
function remove_mappoint_obs!(m::MapManager, kpid::Int, kfid::Int)
    lock(m.keyframe_lock)
    lock(m.mappoint_lock)

    mp = get(m.map_points, kpid, nothing)
    if mp ≡ nothing
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

    # Remove KeyFrame observation from MapPoint.
    remove_kf_observation!(mp, kfid)

    # Remove MapPoint from KeyFrame.
    kf = get(m.frames_map, kfid, nothing)
    if kf ≡ nothing
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end
    remove_keypoint!(kf, kpid)

    for observer_id in mp.observer_keyframes_ids
        observer_kf = get(m.frames_map, observer_id, nothing)
        observer_kf ≡ nothing && continue

        decrease_covisible_kf!(kf, observer_id)
        decrease_covisible_kf!(observer_kf, kfid)
    end

    unlock(m.keyframe_lock)
    unlock(m.mappoint_lock)
end

"""
Update position of a MapPoint.
"""
function update_mappoint!(m::MapManager, kpid, new_position)
    lock(m.keyframe_lock)
    lock(m.mappoint_lock)

    if !(kpid in keys(m.map_points))
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

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
    set_position!(mp, new_position)

    unlock(m.mappoint_lock)
    unlock(m.keyframe_lock)
end

"""
Update MapPoints and covisible graph between KeyFrames.
"""
function update_frame_covisibility!(map_manager::MapManager, frame::Frame)
    covisible_keyframes = Dict{Int64, Int64}()
    local_map_ids = Set{Int64}()
    # For each Keypoint in the `frame`, get its corresponding MapPoint.
    # Get the set of KeyFrames that observe this MapPoint.
    # Add them to the covisible map, which contains all KeyFrames
    # that share visibility with the `frame`.
    for kp in get_keypoints(frame)
        if !(kp.id in keys(map_manager.map_points))
            remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
            remove_obs_from_current_frame!(map_manager, kp.id)
            continue
        end
        mp = get_mappoint(map_manager, kp.id)
        mp_observers = get_observers(mp)
        # Get the set of KeyFrames observing this KeyFrame to update covisibility.
        for kfid in mp_observers
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
        if !(kfid in keys(map_manager.frames_map))
            push!(bad_kfids, kfid)
            continue
        end
        cov_frame = get_keyframe(map_manager, kfid)
        add_covisibility!(cov_frame, frame.kfid, cov_score)
        for kp in get_3d_keypoints(cov_frame)
            kp.id in keys(frame.keypoints) || push!(local_map_ids, kp.id)
        end
    end
    for bad_kfid in bad_kfids
        pop!(covisible_keyframes, bad_kfid)
    end
    # Update the set of covisible KeyFrames.
    set_covisible_map!(frame, covisible_keyframes)
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
