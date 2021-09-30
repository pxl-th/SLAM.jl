struct KeyFrame
    id::Int64
    # image::Matrix{Gray}
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
        estimator_thread, ReentrantLock(),
    )
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

        # TODO match to local map
        # Send new KF to estimator for bundle adjustment.
        @debug "[MP] Sending new Keyframe to Estimator."
        add_new_kf!(mapper.estimator, new_keyframe)
        # TODO send new KF to loop closer
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
    P1 = K * SMatrix{4, 4, Float64}(I)

    good, candidates = 0, 0
    rel_kfid = -1
    # frame -> observer key frame.
    rel_pose::SMatrix{4, 4, Float64, 16} = SMatrix{4, 4, Float64, 16}(I)
    # observer key frame -> frame.
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
        observer_kf ≡ nothing && continue # TODO should this be possible?

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
        # kp.id in keys(observer_kf.keypoints) || continue
        # observer_kp = get_keypoint(observer_kf, kp.id)
        obup = observer_kp.undistorted_pixel
        kpup = kp.undistorted_pixel

        parallax = norm(obup .- project(cam, rel_pose[1:3, 1:3] * kp.position))
        candidates += 1

        # Compute 3D pose and check if it is good.
        # Note, that we use inverted relative pose.
        left_point = iterative_triangulation(
            obup[[2, 1]], kpup[[2, 1]], P1, K * rel_pose_inv,
        )
        left_point *= 1.0 / left_point[4]
        # Project into the right camera (new KeyFrame).
        right_point = rel_pose_inv * left_point

        # Ensure that 3D point is in front of the both cameras.
        if left_point[3] < 0.1 || right_point[3] < 0.1
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            continue
        end

        # Remove MapPoint with high reprojection error.
        lrepr = norm(project(cam, left_point[1:3]) .- obup)
        rrepr = norm(project(cam, right_point[1:3]) .- kpup)
        if lrepr > max_error || rrepr > max_error
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            continue
        end
        # 3D pose is good, update MapPoint and related Frames.
        wpt = project_camera_to_world(observer_kf, left_point)[1:3]
        update_mappoint!(mapper.map_manager, kp.id, wpt)
        good += 1
    end
    @info "[MP] Temporal triangulation: $good/$candidates good KeyPoints."
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
