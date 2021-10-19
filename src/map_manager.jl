"""
```julia
MapManager(params::Params, frame::Frame, extractor::Extractor)
```

Map Manager is responsible for managing Keyframes in the map
as well as Mappoints.

# Arguments:

- `current_frame::Frame`: Current frame that is shared throughout
    all the components in the system.
- `frames_map::Dict{Int64, Frame}`: Map of the Keyframes (its id → Keyframe).
- `params::Params`: Parameters of the system.
- `extractor::Extractor`: Extractor for finding keypoints in the frames.
- `map_points::Dict{Int64, MapPoint}`: Map of all the map_points
    (its id → MapPoints).
- `current_mappoint_id::Int64`: Id of the current Mappoint to be created.
    It is incremented each time a new Mappoint is added to `map_points`.
- `current_keyframe_id::Int64`: Id of the current Keyframe to be created.
    It is incremented each time a new Keyframe is added to `frames_map`.
- `nb_keyframes::Int64`: Total number of keyframes.
- `nb_mappoints::Int64`: Total number of mappoints.
"""
mutable struct MapManager
    current_frame::Frame
    frames_map::Dict{Int64, Frame}

    params::Params
    extractor::Extractor
    map_points::Dict{Int64, MapPoint}

    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_keyframes::Int64
    nb_mappoints::Int64

    mappoint_lock::ReentrantLock
    keyframe_lock::ReentrantLock
    map_lock::ReentrantLock
    optimization_lock::ReentrantLock
end

function MapManager(params::Params, frame::Frame, extractor::Extractor)
    mappoint_lock = ReentrantLock()
    keyframe_lock = ReentrantLock()
    map_lock = ReentrantLock()
    optimization_lock = ReentrantLock()

    MapManager(
        frame, Dict{Int64, Frame}(), params, extractor,
        Dict{Int64, MapPoint}(), 0, 0, 0, 0,
        mappoint_lock, keyframe_lock, map_lock, optimization_lock)
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
    @debug "[MM] Adding KF $(m.current_frame.kfid) to Map."

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

    if m.params.do_local_matching
        descriptors, keypoints = describe(m.extractor, image, keypoints)
    else
        descriptors = fill(BitVector(), length(keypoints))
    end
    add_keypoints_to_frame!(m, m.current_frame, keypoints, descriptors)
end

function add_keypoints_to_frame!(m::MapManager, frame, keypoints, descriptors)
    lock(m.mappoint_lock) do
        for (kp, dp) in zip(keypoints, descriptors)
            # m.current_mappoint_id is incremented in `add_mappoint!`.
            add_keypoint!(frame, Point2f(kp[1], kp[2]), m.current_mappoint_id)
            add_mappoint!(m, dp)
        end
    end
end

@inline function add_mappoint!(m::MapManager, descriptor)
    mp = MapPoint(m.current_mappoint_id, m.current_keyframe_id, descriptor)
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
    new_keyframe = deepcopy(m.current_frame)

    lock(m.keyframe_lock) do
        m.frames_map[m.current_keyframe_id] = new_keyframe
        m.current_keyframe_id += 1
        m.nb_keyframes += 1
    end
    @debug "[MM] Added."
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

    # Remove MapPoint from KeyFrame.
    kf = get(m.frames_map, kfid, nothing)
    kf ≢ nothing && remove_keypoint!(kf, kpid)

    mp = get(m.map_points, kpid, nothing)
    if mp ≡ nothing
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

    # Remove KeyFrame observation from MapPoint.
    remove_kf_observation!(mp, kfid)

    if kf ≢ nothing
        for observer_id in mp.observer_keyframes_ids
            observer_kf = get(m.frames_map, observer_id, nothing)
            observer_kf ≡ nothing && continue

            decrease_covisible_kf!(kf, observer_id)
            decrease_covisible_kf!(observer_kf, kfid)
        end
    end

    unlock(m.keyframe_lock)
    unlock(m.mappoint_lock)
end

"""
```julia
update_mappoint!(m::MapManager, mpid, new_position)
```

Update position of a MapPoint.
"""
function update_mappoint!(m::MapManager, mpid, new_position)
    lock(m.keyframe_lock)
    lock(m.mappoint_lock)

    if !(mpid in keys(m.map_points))
        unlock(m.mappoint_lock)
        unlock(m.keyframe_lock)
        return
    end

    mp = m.map_points[mpid]
    # If MapPoint is 2D, turn it to 3D and update its observing KeyFrames.
    if !mp.is_3d
        for observer_id in mp.observer_keyframes_ids
            if observer_id in keys(m.frames_map)
                turn_keypoint_3d!(m.frames_map[observer_id], mpid)
            else
                remove_kf_observation!(mp, observer_id)
            end
        end
        if mp.is_observed
            # Because we deepcopy frame before putting it to the frames_map,
            # we need to update current frame as well.
            # Which should also update current frame in the FrontEnd.
            turn_keypoint_3d!(m.current_frame, mpid)
        end
    end
    set_position!(mp, new_position)

    unlock(m.mappoint_lock)
    unlock(m.keyframe_lock)
end

"""
```julia
update_frame_covisibility!(map_manager::MapManager, frame::Frame)
```

Update covisibility graph for the `frame`.
This is done by going through all of the keypoints in the `frame`.
Getting their corresponding mappoints. And joining sets of observers for
those mappoints.
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

"""
```julia
reset!(m::MapManager)
```

Reset map manager.
"""
function reset!(m::MapManager)
    m.nb_keyframes = 0
    m.nb_mappoints = 0
    m.current_keyframe_id = 0
    m.current_mappoint_id = 0

    m.map_points |> empty!
    m.frames_map |> empty!
end

"""
```julia
merge_mappoints(m::MapManager, prev_id, new_id)
```

Merge `prev_id` Mappoint into `new_id` Mappoint.
For "previous" observers, update mappoint and keypoint to the new.
"""
function merge_mappoints(m::MapManager, prev_id, new_id)
    lock(m.mappoint_lock)
    lock(m.keyframe_lock)
    try
        prev_mp = get(m.map_points, prev_id, nothing)
        prev_mp ≡ nothing && return
        new_mp = get(m.map_points, new_id, nothing)
        new_mp ≡ nothing && return
        new_mp.is_3d || return

        prev_observers = get_observers(prev_mp)
        new_observers = get_observers(new_mp)

        # For previous mappoint observers, update keypoint for them.
        # If successfull, then add covisibility link between old and new
        # observer keyframes.
        for prev_observer_id in prev_observers
            prev_observer_kf = get(m.frames_map, prev_observer_id, nothing)
            prev_observer_kf ≡ nothing && continue
            update_keypoint!(
                prev_observer_kf, prev_id, new_id, new_mp.is_3d) || continue

            add_keyframe_observation!(new_mp, prev_observer_id)
            for new_observer_id in new_observers
                new_observer_kf = get(m.frames_map, new_observer_id, nothing)
                new_observer_kf ≡ nothing && continue

                add_covisibility!(new_observer_kf, prev_observer_id)
                add_covisibility!(prev_observer_kf, new_observer_id)
            end
        end

        for (kfid, descriptor) in prev_mp.keyframes_descriptors
            add_descriptor!(new_mp, kfid, descriptor)
        end
        if is_observing_kp(m.current_frame, prev_id)
            update_keypoint!(m.current_frame, prev_id, new_id, new_mp.is_3d)
        end

        # Update nb mappoints and erase old mappoint.
        prev_mp.is_3d && (m.nb_mappoints -= 1;)
        pop!(m.map_points, prev_id)
    catch e
        showerror(stdout, e); println()
        display(stacktrace(catch_backtrace())); println()
    finally
        unlock(m.keyframe_lock)
        unlock(m.mappoint_lock)
    end
end

"""
```julia
optical_flow_matching!(map_manager, frame, from_pyramid, to_pyramid, stereo)
```

Match keypoints in `frame` from `from_pyramid` to `to_pyramid`.
This function is used when matching keypoints temporally from previous frame
to current frame, or when matching keypoints between stereo image.

If there are 3D keypoints in the `frame`, then try to match respectful
keypoints using displacement guess from motion model (when matching temporally)
or calibration pose (in stereo).

# Arguments

- `map_manager::MapManager`: Map manager, used for retrieving parameters info,
    3D mappoints, removing mappoints.
- `frame`: Frame for which to do matching.
- `from_pyramid::LKPyramid`: Pyramid from which to track.
- `to_pyramid::LKPyramid`: Pyramid to which to track.
- `stereo::Bool`: Set to `true` if doing stereo matching. It will retrieve data
    using mutex and update `stereo` keypoints instead of regular keypoints.
    Otherwise set to `false`.
"""
function optical_flow_matching!(
    map_manager::MapManager, frame::Frame,
    from_pyramid::LKPyramid, to_pyramid::LKPyramid, stereo,
)
    window_size = map_manager.params.window_size
    max_distance = map_manager.params.max_ktl_distance
    pyramid_levels = map_manager.params.pyramid_levels

    pyramid_levels_3d = 1
    ids = Vector{Int64}(undef, frame.nb_keypoints)
    pixels = Vector{Point2f}(undef, frame.nb_keypoints)

    ids3d = Vector{Int64}(undef, frame.nb_3d_kpts)
    pixels3d = Vector{Point2f}(undef, frame.nb_3d_kpts)
    displacements3d = Vector{Point2f}(undef, frame.nb_3d_kpts)

    i, i3d = 1, 1
    scale = 1.0 / 2.0^pyramid_levels_3d
    n_good = 0

    keypoints = stereo ? get_keypoints(frame) : values(frame.keypoints)
    for kp in keypoints
        if !kp.is_3d
            pixels[i] = kp.pixel
            ids[i] = kp.id
            i += 1
            continue
        end

        mp = stereo ?
            get_mappoint(map_manager, kp.id) :
            map_manager.map_points[kp.id]
        if mp ≡ nothing
            remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
            continue
        end

        position = get_position(mp)
        projection = stereo ?
            project_world_to_right_image_distort(frame, position) :
            project_world_to_image_distort(frame, position)

        if stereo
            if in_right_image(frame, projection)
                ids3d[i3d] = kp.id
                pixels3d[i3d] = kp.pixel
                displacements3d[i3d] = scale .* (projection .- kp.pixel)
                i3d += 1
            else
                remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
                continue
            end
        else
            if in_image(frame, projection)
                ids3d[i3d] = kp.id
                pixels3d[i3d] = kp.pixel
                displacements3d[i3d] = scale .* (projection .- kp.pixel)
                i3d += 1
            end
        end
    end

    i3d -= 1
    ids3d = @view(ids3d[1:i3d])
    pixels3d = @view(pixels3d[1:i3d])
    displacements3d = @view(displacements3d[1:i3d])

    failed_3d = true
    if !isempty(ids3d)
        new_keypoints, status = fb_tracking!(
            from_pyramid, to_pyramid, pixels3d;
            displacement=displacements3d,
            pyramid_levels=pyramid_levels_3d,
            window_size, max_distance)

        nb_good = 0
        for j in 1:length(status)
            if status[j]
                if stereo
                    succ = maybe_stereo_update!(frame, ids3d[j], new_keypoints[j])
                    succ && (n_good += 1;)
                else
                    update_keypoint!(frame, ids3d[j], new_keypoints[j])
                    nb_good += 1
                end
            else
                # If failed → add to track with 2d keypoints w/o prior.
                pixels[i] = pixels3d[j]
                ids[i] = ids3d[j]
                i += 1
            end
        end
        @debug "[MM] 3D Points tracked $nb_good. Stereo $stereo."
        failed_3d = nb_good < 0.33 * length(ids3d)
    end

    i -= 1
    pixels = @view(pixels[1:i])
    ids = @view(ids[1:i])

    isempty(pixels) && return nothing
    new_keypoints, status = fb_tracking!(
        from_pyramid, to_pyramid, pixels;
        pyramid_levels, window_size, max_distance)

    for j in 1:length(new_keypoints)
        if stereo
            status[j] && maybe_stereo_update!(frame, ids[j], new_keypoints[j]) &&
                (n_good += 1;)
        else
            status[j] ?
                update_keypoint!(frame, ids[j], new_keypoints[j]) :
                remove_obs_from_current_frame!(map_manager, ids[j])
        end
    end
    nothing
end

"""
```julia
maybe_stereo_update!(
    frame::Frame, kpid, new_position::Point2f; epipolar_error::Float64 = 2.0,
)
```

Update stereo keypoint if the vertical distance between matched keypoints is
less than `epipolar_error`. In this case, set y-coordinate equal
to the left keypoint. Otherwise do nothing.

# Returns:

`true` if successfully updated, otherwise `false`.
"""
function maybe_stereo_update!(
    frame::Frame, kpid, new_position; epipolar_error = 2.0,
)
    kp = get_keypoint(frame, kpid)
    right_pixel = undistort_point(frame.right_camera, new_position)
    abs(kp.undistorted_pixel[1] - right_pixel[1]) > epipolar_error &&
        return false
    # Make y-coordinate the same.
    corrected = Point2f(kp.pixel[1], new_position[2])
    update_stereo_keypoint!(frame, kpid, corrected)
    true
end
