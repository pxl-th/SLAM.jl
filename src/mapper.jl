struct KeyFrame
    id::Int64
    image::Matrix{Gray}
    # optical flow pyramid levels
    # raw image (colored)
end

mutable struct Mapper
    params::Params
    map_manager::MapManager

    current_frame::Frame
    keyframe_queue::Vector{KeyFrame}

    exit_required::Bool
    new_kf_available::Bool
end

Mapper(params::Params, map_manager::MapManager, frame::Frame) =
    Mapper(params, map_manager, frame, KeyFrame[], false, false)

function run!(mapper::Mapper)
    if mapper.exit_required
        @debug "[Mapper] Mapper is stopping - exit required."
        return
    end
    
    succ, kf = mapper |> get_new_kf!
    succ || return

    @debug "[Mapper] New keyframe to process: $(kf.id) id."
    new_keyframe = mapper.map_manager.frames_map[kf.id]
    # Triangulate temporal.
    if new_keyframe.nb_2d_kpts > 0 && new_keyframe.kfid > 0
        @debug "[Mapper] Temporal triangulation. Before triangulation:"
        @debug "\t- 2d kpts: $(new_keyframe.nb_2d_kpts)"
        @debug "\t- 3d kpts: $(new_keyframe.nb_3d_kpts)"

        triangulate_temporal!(mapper, new_keyframe)

        @debug "[Mapper] After triangulation:"
        @debug "\t- 2d kpts: $(new_keyframe.nb_2d_kpts)"
        @debug "\t- 3d kpts: $(new_keyframe.nb_3d_kpts)"
    end

    # Check if reset is required (for mono mode).
    if mapper.params.vision_initialized
        if kf.id == 1 && new_keyframe.nb_3d_kpts < 30
            @debug "[Mapper] Bad initialization detected. Resetting!"
            mapper.params.reset_required = true
            mapper |> reset!
            return
        elseif kf.id < 10 && new_keyframe.nb_3d_kpts < 3
            @debug "[Mapper] Reset required. Nb 3D points: $(new_keyframe.nb_3d_kpts)."
            mapper.params.reset_required = true
            mapper |> reset!
            return
        end
    end

    # Update the map points and the covisibility graph between KeyFrames.
    # TODO update_frame_covisibility(mapper.map_manager, new_keyframe)

    # TODO match to local map
    # TODO send new KF to estimator for bundle adjustment
end

function triangulate_temporal!(mapper::Mapper, frame::Frame)
    keypoints = frame |> get_2d_keypoints
    if isempty(keypoints)
        @debug "[Mapper] No 2D keypoints to triangulate."
        return
    end
    P1 = SMatrix{3, 4, Float64}(I)

    good = 0
    candidates = 0
    rel_kfid = -1
    rel_pose = SMatrix{4, 4, Float64}(I)

    # Go through all 2D keypoints in `frame`.
    for (i, kp) in enumerate(keypoints)
        # TODO removeMapPointsObs if not in map
        kp.id in keys(mapper.map_manager.map_points) || continue
        map_point = mapper.map_manager.map_points[kp.id]
        map_point.is_3d && continue

        if length(map_point.observer_keyframes_ids) < 2
            @debug "[Mapper] Too few observers for a MapPoint $(map_point.id): " *
                "$(length(map_point.observer_keyframes_ids))"
            continue
        else
            @debug "[Mapper] Enough observers for a MapPoint $(map_point.id): " *
                "$(length(map_point.observer_keyframes_ids))"
        end
        kfid = map_point.observer_keyframes_ids[1]
        frame.kfid == kfid && continue

        # Get 1st KeyFrame observation for the MapPoint.
        observer_kf = mapper.map_manager.frames_map[kfid]
        # Compute relative motion between new KF & observer KF.
        # Don't recompute if the frame's ids don't change.
        if rel_kfid != kfid
            rel_pose = observer_kf.cw * frame.wc
            @debug "[Mapper] Rel Pose:\n" *
                "\t- Observer KF: $(observer_kf.id), $(observer_kf.kfid) \n" *
                "\t- Frame: $(frame.id), $(frame.kfid)\n" *
                "\t- Rel KFID: $rel_kfid, KFID: $kfid"

            rel_kfid = kfid

            display(frame.wc); println()
            display(observer_kf.cw); println()
            display(rel_pose); println()
            println("--------------")
        end
        # Get observer keypoint.
        if !(kp.id in keys(observer_kf.keypoints))
            @debug "[Mapper] No observers for a Keypoint $(kp.id)"
            continue
        end
        observer_kp = observer_kf.keypoints[kp.id]
        # TODO Check rotation or normal parallax?
        rot_px = project(
            frame.camera, (rel_pose * SVector{4}(kp.position..., 1.0))[1:3],
        )
        @debug "[Mapper] px: $(observer_kp.undistorted_pixel), rot px: $rot_px"
        parallax = norm(observer_kp.undistorted_pixel .- rot_px)
        candidates += 1
        # Compute 3D pose and check if it is good.
        # Note, that we invert relative pose in triangulation.
        left_point = RecoverPose.triangulate_point(
            observer_kp.position, kp.position, P1, inv(SE3, rel_pose),
        )
        left_point *= 1.0 / left_point[4]
        # Project into the right camera (new KeyFrame).
        right_point = inv(SE3, rel_pose) * left_point

        display(rel_pose); println()
        @debug "left point $left_point"
        @debug "right point $right_point"

        # Ensure that 3D point is in front of the both cameras.
        if left_point[3] < 0.1 || right_point[3] < 0.1
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            @debug "[Mapper] Point is behind both cameras:\n" *
                "\t- mp id: $(map_point.id) \n" *
                "\t- id: $(observer_kp.id) \n" *
                "\t- left point: $left_point \n" *
                "\t- right point: $right_point \n" *
                "\t- rel pose: $rel_pose \n" *
                "\t- N observers: $(length(map_point.observer_keyframes_ids))"
            continue
        end
        # Remove MapPoint with high reprojection error.
        left_projection = project(observer_kf.camera, left_point)
        right_projection = project(frame.camera, right_point)
        left_error = norm(left_projection - observer_kp.undistorted_pixel)
        right_error = norm(right_projection - kp.undistorted_pixel)

        if left_error > mapper.params.max_reprojection_error ||
            right_error > mapper.params.max_reprojection_error
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            @debug "[Mapper] Reprojection error is too high: \n" *
                "\t- left projection: $left_projection, error: $left_error \n" *
                "\t- right projection: $right_projection, error: $right_error \n" *
                "\t- parallax: $parallax"
            continue
        end
        # 3D pose is good, update MapPoint and related Frames.
        wpt = project_camera_to_world(observer_kf, left_point)[1:3]
        update_mappoint!(mapper.map_manager, kp.id, wpt, 1.0 / left_point[3])
        good += 1
        @debug "[Mapper] Parallax $parallax"
        exit()
    end
    @debug "[Mapper] Temporal triangulation: $good good KeyPoints."
end

function get_new_kf!(mapper::Mapper)::Tuple{Bool, Union{Nothing, KeyFrame}}
    if isempty(mapper.keyframe_queue)
        mapper.new_kf_available = false
        return false, nothing
    end
    
    keyframe = mapper.keyframe_queue |> popfirst!
    mapper.new_kf_available = !isempty(mapper.keyframe_queue)
    true, keyframe
end

function add_new_kf!(mapper::Mapper, kf::KeyFrame)
    push!(mapper.keyframe_queue, kf)
    mapper.new_kf_available = true
end

function reset!(mapper::Mapper)
    mapper.new_kf_available = false
    mapper.exit_required = false
    mapper.keyframe_queue |> empty!
end
