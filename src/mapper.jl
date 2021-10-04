struct KeyFrame
    id::Int64
    left_pyramid::Union{Nothing, ImageTracking.LKPyramid}
    right_image::Union{Nothing, Matrix{Gray{Float64}}}
end

mutable struct Mapper
    params::Params
    map_manager::MapManager
    estimator::Estimator

    current_frame::Frame
    keyframe_queue::Vector{KeyFrame}

    exit_required::Bool
    new_kf_available::Bool

    estimator_thread
    queue_lock::ReentrantLock
end

function Mapper(params::Params, map_manager::MapManager, frame::Frame)
    estimator = Estimator(map_manager, params)
    estimator_thread = Threads.@spawn run!(estimator)
    @info "[MP] Launched estimator thread."

    Mapper(
        params, map_manager, estimator,
        frame, KeyFrame[], false, false,
        estimator_thread, ReentrantLock())
end

function run!(mapper::Mapper)
    while !mapper.exit_required
        succ, kf = get_new_kf!(mapper)
        if !succ
            sleep(1e-2)
            continue
        end

        new_keyframe = get_keyframe(mapper.map_manager, kf.id)
        new_keyframe ≡ nothing &&
            @error "[MP] Got invalid frame $(kf.id) from Map"
        @info "[MP] Get $(kf.id) KF"

        if mapper.params.stereo
            @info "[MP] Stereo Matching"
            try
                right_pyramid = ImageTracking.LKPyramid(
                    kf.right_image, mapper.params.pyramid_levels;
                    σ=mapper.params.pyramid_σ)

                optical_flow_matching!(
                    mapper.map_manager,
                    new_keyframe, kf.left_pyramid, right_pyramid;
                    window_size=mapper.params.window_size,
                    max_distance=mapper.params.max_ktl_distance,
                    pyramid_levels=mapper.params.pyramid_levels, stereo=true)

                vimage = RGB{Float64}.(kf.right_image)
                draw_keypoints!(vimage, new_keyframe; right=true)
                save("/home/pxl-th/projects/slam-data/images/frame-$(new_keyframe.id)-right.png", vimage)
            catch e
                showerror(stdout, e)
                display(stacktrace(catch_backtrace()))
            end
            @info "[MP] Stereo Matched"
        end

        if new_keyframe.nb_2d_kpts > 0 && new_keyframe.kfid > 0
            lock(mapper.map_manager.map_lock)
            try
                triangulate_temporal!(mapper, new_keyframe)
            catch e
                showerror(stdout, e)
                display(stacktrace(catch_backtrace()))
            finally
                unlock(mapper.map_manager.map_lock)
            end
        end

        # Check if reset is required.
        if mapper.params.vision_initialized
            if kf.id == 1 && new_keyframe.nb_3d_kpts < 30
                @warn "[MP] Bad initialization detected. Resetting!"
                mapper.params.reset_required = true
                mapper |> reset!
                continue
            elseif kf.id < 10 && new_keyframe.nb_3d_kpts < 3
                @warn "[MP] Reset required. Nb 3D points: $(new_keyframe.nb_3d_kpts)."
                mapper.params.reset_required = true
                mapper |> reset!
                continue
            end
        end
        # Update the map points and the covisibility graph between KeyFrames.
        update_frame_covisibility!(mapper.map_manager, new_keyframe)

        # if kf.id > 0
        #     try
        #         match_local_map!(mapper, new_keyframe)
        #     catch e
        #         showerror(stdout, e)
        #         display(stacktrace(catch_backtrace()))
        #     end
        # end

        # Send new KF to estimator for bundle adjustment.
        @debug "[MP] Sending new Keyframe to Estimator."
        add_new_kf!(mapper.estimator, new_keyframe)
    end
    mapper.estimator.exit_required = true
    @info "[MP] Exit required."
    wait(mapper.estimator_thread)
end

function triangulate_temporal!(mapper::Mapper, frame::Frame)
    keypoints = get_2d_keypoints(frame)
    @info "[MP] Triangulating $(length(keypoints)) Keypoints..."
    if isempty(keypoints)
        @warn "[MP] No 2D keypoints to triangulate."
        return
    end
    K = to_4x4(frame.camera.K)
    P1 = K * SMatrix{4, 4, Float64, 16}(I)

    good, candidates = 0, 0
    rel_kfid = -1
    # frame -> observer key frame.
    rel_pose::SMatrix{4, 4, Float64, 16} = SMatrix{4, 4, Float64, 16}(I)
    rel_pose_inv::SMatrix{4, 4, Float64, 16} = SMatrix{4, 4, Float64, 16}(I)

    cam = frame.camera
    max_error = mapper.params.max_reprojection_error

    # Go through all 2D keypoints in `frame`.
    for kp in keypoints
        @assert !kp.is_3d
        # Remove mappoints observation if not in map.
        if !(kp.id in keys(mapper.map_manager.map_points))
            remove_mappoint_obs!(mapper.map_manager, kp.id, frame.kfid)
            continue
        end
        map_point = get_mappoint(mapper.map_manager, kp.id)
        map_point.is_3d && continue

        # Get first KeyFrame id from the set of mappoint observers.
        observers = get_observers(map_point)
        length(observers) < 2 && continue
        kfid = observers[1]
        frame.kfid == kfid && continue

        # Get 1st KeyFrame observation for the MapPoint.
        observer_kf = get_keyframe(mapper.map_manager, kfid)
        if observer_kf ≡ nothing
            @error "[MP] Missing observer for triangulation."
            continue # TODO should this be possible?
        end

        # Compute relative motion between new KF & observer KF.
        # Don't recompute if the frame's ids don't change.
        if rel_kfid != kfid
            rel_pose = observer_kf.cw * frame.wc
            rel_pose_inv = inv(SE3, rel_pose)
            rel_kfid = kfid
        end

        # Get observer keypoint.
        observer_kp = get_keypoint(observer_kf, kp.id)
        observer_kp ≡ nothing && continue
        obup = observer_kp.undistorted_pixel
        kpup = kp.undistorted_pixel

        parallax = norm(obup .- project(cam, rel_pose[1:3, 1:3] * kp.position))
        candidates += 1

        # Compute 3D pose and check if it is good.
        # Note, that we use inverted relative pose.
        left_point = iterative_triangulation(
            obup[[2, 1]], kpup[[2, 1]], P1, K * rel_pose_inv)
        if left_point[4] ≈ 0.0 || left_point[4] > 1e6
            @error "[MP] Failed triangulation, singular value."
            continue
        end

        left_point *= 1.0 / left_point[4]
        # Project into the right camera (new KeyFrame).
        right_point = rel_pose_inv * left_point

        # Ensure that 3D point is in front of the both cameras.
        if left_point[3] < 0.1 || right_point[3] < 0.1
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid)
            continue
        end

        # Remove MapPoint with high reprojection error.
        lrepr = norm(project(cam, left_point[1:3]) .- obup)
        rrepr = norm(project(cam, right_point[1:3]) .- kpup)
        if lrepr > max_error || rrepr > max_error
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid)
            continue
        end
        # 3D pose is good, update MapPoint and related Frames.
        wpt = project_camera_to_world(observer_kf, left_point)[1:3]
        update_mappoint!(mapper.map_manager, kp.id, wpt)
        good += 1
    end
    @info "[MP] Temporal triangulation: $good/$candidates good KeyPoints."
end

"""
Try matching keypoints from `frame` with keypoints from frames in its
covisibility graph (aka local map).
"""
function match_local_map!(mapper::Mapper, frame::Frame)
    # Maximum number of MapPoints to track.
    max_nb_mappoints = 10 * mapper.params.max_nb_keypoints
    covisibility_map = get_covisible_map(frame)

    if length(frame.local_map_ids) < max_nb_mappoints
        # Get local map of the oldest covisible KeyFrame and add it
        # to the local map to `frame` to search for MapPoints.
        kfid = collect(keys(covisibility_map))[1]
        co_kf = get_keyframe(mapper.map_manager, kfid)
        while co_kf ≡ nothing && kfid > 0
            kfid -= 1
            co_kf = get_keyframe(mapper.map_manager, kfid)
        end

        co_kf ≢ nothing && union!(frame.local_map_ids, co_kf.local_map_ids)
        # TODO if still not enough, go for another round.
    end
    @debug "[MP] Local Map candidates $(length(frame.local_map_ids))."

    prev_new_map = do_local_map_matching(
        mapper, frame, frame.local_map_ids;
        max_projection_distance=mapper.params.max_projection_distance,
        max_descriptor_distance=mapper.params.max_descriptor_distance)
    @debug "[MP] New Matching: $(length(prev_new_map))."
    # display(prev_new_map); println()

    isempty(prev_new_map) || merge_matches(mapper, prev_new_map)
end

function merge_matches(mapper::Mapper, prev_new_map::Dict{Int64, Int64})
    lock(mapper.map_manager.optimization_lock)
    lock(mapper.map_manager.map_lock)
    try
        for (prev_id, new_id) in prev_new_map
            merge_mappoints(mapper.map_manager, prev_id, new_id);
        end
    catch e
        showerror(stdout, e); println()
        display(stacktrace(catch_backtrace())); println()
    finally
        unlock(mapper.map_manager.map_lock)
        unlock(mapper.map_manager.optimization_lock)
    end
end

"""
Given a frame and its local map of Keypoints ids (triangulated),
project respective mappoints onto the frame, find surrounding keypoints (triangulated?),
match surrounding keypoints with the projection.
Best match is the new candidate for replacement.
"""
function do_local_map_matching(
    mapper::Mapper, frame::Frame, local_map::Set{Int64};
    max_projection_distance, max_descriptor_distance,
)
    prev_new_map = Dict{Int64, Int64}()
    isempty(local_map) && return prev_new_map

    # Maximum field of view.
    vfov = 0.5 * frame.camera.height / frame.camera.fy
    hfov = 0.5 * frame.camera.width / frame.camera.fx
    max_rad_fov = vfov > hfov ? atan(vfov) : atan(hfov)
    view_threshold = cos(max_rad_fov)
    @debug "[MP] View threshold $view_threshold."

    # Define max distance from projection.
    frame.nb_3d_kpts < 30 && (max_projection_distance *= 2.0;)
    # matched kpid → [(local map kpid, distance)] TODO
    matches = Dict{Int64, Vector{Tuple{Int64, Float64}}}()

    # Go through all MapPoints from the local map in `frame`.
    for kpid in local_map
        is_observing_kp(frame, kpid) && continue
        mp = get_mappoint(mapper.map_manager, kpid)
        mp ≡ nothing && continue
        (!mp.is_3d || isempty(mp.descriptor)) && continue

        # Project MapPoint into KeyFrame's image plane.
        position = get_position(mp)
        camera_position = project_world_to_camera(frame, position)[1:3]
        camera_position[3] < 0.1 && continue

        view_angle = camera_position[3] / norm(camera_position)
        abs(view_angle) < view_threshold && continue

        projection = project_undistort(frame.camera, camera_position)
        in_image(frame.camera, projection) || continue

        surrounding_keypoints = get_surrounding_keypoints(frame, projection)

        # Find best match for the `mp` among `surrounding_keypoints`.
        best_id, best_distance = find_best_match(
            mapper.map_manager, frame, mp, projection, surrounding_keypoints;
            max_projection_distance, max_descriptor_distance)
        best_id == -1 && continue

        match = (kpid, best_distance)
        if best_id in keys(matches)
            push!(matches[best_id], match)
        else
            matches[best_id] = Tuple{Int64, Float64}[match]
        end
    end

    for (kpid, match) in matches
        best_distance = 1e6
        best_id = -1

        for (local_kpid, distance) in match
            if distance ≤ best_distance
                best_distance = distance
                best_id = local_kpid
            end
            best_id != -1 && (prev_new_map[kpid] = best_id;)
        end
    end
    prev_new_map
end

"""
For a given `target_mp` MapPoint, find best match among surrounding keypoints.

Given target mappoint from covisibility graph, its projection onto `frame`
and surrounding keypoints in `frame` for that projection,
find best matching keypoint (already triangulated?) in `frame`.
"""
function find_best_match(
    map_manager::MapManager, frame::Frame, target_mp::MapPoint,
    projection, surrounding_keypoints;
    max_projection_distance, max_descriptor_distance,
)
    target_mp_observers = get_observers(target_mp)
    target_mp_position = get_position(target_mp)

    # TODO parametrize descriptor size.
    min_distance = 256.0 * max_descriptor_distance
    best_distance, second_distance = min_distance, min_distance
    best_id, second_id = -1, -1

    for kp in surrounding_keypoints
        kp.id < 0 && continue
        distance = norm(projection .- kp.pixel)
        distance > max_projection_distance && continue

        # TODO should surrounding kp be triangulated?
        mp = get_mappoint(map_manager, kp.id)
        if mp ≡ nothing
            # TODO should remove keypoint as well?
            remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
            continue
        end
        isempty(mp.descriptor) && continue

        # Check that `kp` and `target_mp` are indeed candidates for matching.
        # They should have no overlap in their observers.
        mp_observers = get_observers(mp)
        isempty(intersect(target_mp_observers, mp_observers)) || continue

        avg_projection = 0.0
        n_projections = 0

        # Compute average projection distance for the `target_mp` projected
        # into each of the `mp` observers KeyFrame.
        for observer_kfid in mp_observers
            observer_kf = get_keyframe(map_manager, observer_kfid)
            observer_kf ≡ nothing && (remove_mappoint_obs!(
                map_manager, kp.id, observer_kfid); continue)

            observer_kp = get_keypoint(observer_kf, kp.id)
            observer_kp ≡ nothing && (remove_mappoint_obs!(
                map_manager, kp.id, observer_kfid); continue)

            observer_projection = project_world_to_image_distort(
                observer_kf, target_mp_position)
            avg_projection += norm(observer_kp.pixel .- observer_projection)
            n_projections += 1
        end
        avg_projection /= n_projections
        avg_projection > max_projection_distance && continue

        distance = mappoint_min_distance(target_mp, mp)
        if distance ≤ best_distance
            second_distance = best_distance
            second_id = best_id

            best_distance = distance
            best_id = kp.id
        elseif distance ≤ second_distance
            second_distance = distance
            second_id = kp.id
        end
    end

    # best_id != -1 && second_id != -1 &&
    #     0.9 * second_distance < best_distance && (best_id = -1;)

    best_id, best_distance
end

function get_new_kf!(mapper::Mapper)
    lock(mapper.queue_lock) do
        if isempty(mapper.keyframe_queue)
            mapper.new_kf_available = false
            return false, nothing
        end

        keyframe = popfirst!(mapper.keyframe_queue)
        @info "[MP] Popping queue $(length(mapper.keyframe_queue))"
        mapper.new_kf_available = !isempty(mapper.keyframe_queue)
        true, keyframe
    end
end

function add_new_kf!(mapper::Mapper, kf::KeyFrame)
    lock(mapper.queue_lock) do
        push!(mapper.keyframe_queue, kf)
        mapper.new_kf_available = true
    end
end

function reset!(mapper::Mapper)
    lock(mapper.queue_lock) do
        mapper.new_kf_available = false
        mapper.exit_required = false
        mapper.keyframe_queue |> empty!
    end
end
