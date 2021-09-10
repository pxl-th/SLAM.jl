struct KeyFrame
    id::Int64
    image::Matrix{Gray}
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
        @debug "\t- curr 3d kpts: $(mapper.current_frame.nb_3d_kpts)"

        triangulate_temporal!(mapper, new_keyframe)

        @debug "[Mapper] After triangulation:"
        @debug "\t- 2d kpts: $(new_keyframe.nb_2d_kpts)"
        @debug "\t- 3d kpts: $(new_keyframe.nb_3d_kpts)"
        @debug "\t- curr 3d kpts: $(mapper.current_frame.nb_3d_kpts)"
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
    update_frame_covisibility!(mapper.map_manager, new_keyframe)

    # TODO match to local map
    # Send new KF to estimator for bundle adjustment
    local_bundle_adjustment!(mapper.map_manager, new_keyframe, mapper.params)
    map_filtering!(mapper.map_manager, new_keyframe, mapper.params)
    # TODO send new KF to loop closer
end

"""
Filter out KeyFrames that share too many MapPoints with other KeyFrames
in the covisibility graph. Since they are not informative.
"""
function map_filtering!(map_manager::MapManager, new_keyframe::Frame, params)
    params.filtering_ratio ≥ 1 && return
    new_keyframe.kfid < 20 && return

    covisibility_map = new_keyframe.covisible_kf
    n_removed = 0
    for kfid in keys(covisibility_map)
        # TODO if new kf is available → break
        kfid == 0 && break
        kfid ≥ new_keyframe.kfid && continue

        if !(kfid in keys(map_manager.frames_map))
            remove_covisible_kf!(new_keyframe, kfid)
            continue
        end

        kf = map_manager.frames_map[kfid]
        if kf.nb_3d_kpts < params.min_cov_score ÷ 2
            remove_keyframe!(map_manager, kfid)
            n_removed += 1
            continue
        end

        n_good, n_total = 0, 0
        for kp in get_3d_keypoints(kf)
            if !(kp.id in keys(map_manager.map_points))
                remove_mappoint_obs!(kp.id, kfid)
                continue
            end
            mp = map_manager.map_points[kp.id]
            is_bad!(mp) && continue

            length(mp.observer_keyframes_ids) > 4 && (n_good += 1;)
            n_total += 1;
            # TODO if new kf is available → break
        end

        ratio = n_good / n_total
        ratio > params.filtering_ratio &&
            (remove_keyframe!(map_manager, kfid); n_removed += 1)
    end

    @debug "[MF] Removed $n_removed KeyFrames."
end

function triangulate_temporal!(mapper::Mapper, frame::Frame)
    keypoints = frame |> get_2d_keypoints
    if isempty(keypoints)
        @debug "[Mapper] No 2D keypoints to triangulate."
        return
    end
    K = to_4x4(frame.camera.K)
    P1 = K * SMatrix{4, 4, Float64}(I)

    good = 0
    candidates = 0
    rel_kfid = -1
    # frame -> observer key frame.
    rel_pose = SMatrix{4, 4, Float64}(I)
    # observer key frame -> frame.
    rel_pose_inv = SMatrix{4, 4, Float64}(I)

    max_error = mapper.params.max_reprojection_error
    cam = frame.camera

    # Go through all 2D keypoints in `frame`.
    for kp in keypoints
        # Remove mappoints observation if not in map.
        if !(kp.id in keys(mapper.map_manager.map_points))
            remove_mappoint_obs!(mapper.map_manager, kp.id, frame.kfid)
            @debug "[MP] No MapPoint for KP" maxlog=10
            continue
        end
        map_point = mapper.map_manager.map_points[kp.id]
        if map_point.is_3d
            @debug "[MP] Already 3d" maxlog=10
            continue
        end
        # Get first KeyFrame id from the set of mappoint observers.
        if length(map_point.observer_keyframes_ids) < 2
            @debug "[MP] Not enough observers @ $(frame.kfid): $(map_point.observer_keyframes_ids)" maxlog=10
            continue
        end
        kfid = map_point.observer_keyframes_ids[1]
        if frame.kfid == kfid
            @debug "[MP] Observer is the same as Frame" maxlog=10
            continue
        end
        # Get 1st KeyFrame observation for the MapPoint.
        observer_kf = mapper.map_manager.frames_map[kfid]
        # Compute relative motion between new KF & observer KF.
        # Don't recompute if the frame's ids don't change.
        if rel_kfid != kfid
            rel_pose = observer_kf.cw * frame.wc
            rel_pose_inv = inv(SE3, rel_pose)
            rel_kfid = kfid
        end
        # Get observer keypoint.
        if !(kp.id in keys(observer_kf.keypoints))
            @debug "[MP] Observer has no such KP" maxlog=10
            continue
        end
        observer_kp = observer_kf.keypoints[kp.id]

        obup = observer_kp.undistorted_pixel
        kpup = kp.undistorted_pixel

        parallax = norm(obup .- project(cam, rel_pose[1:3, 1:3] * kp.position))
        candidates += 1
        # Compute 3D pose and check if it is good.
        # Note, that we use inverted relative pose.
        left_point = iterative_triangulation(
            observer_kp.undistorted_pixel[[2, 1]], kp.undistorted_pixel[[2, 1]],
            P1, K * rel_pose_inv,
        )
        left_point *= 1.0 / left_point[4]
        # Project into the right camera (new KeyFrame).
        right_point = rel_pose_inv * left_point

        # Ensure that 3D point is in front of the both cameras.
        if left_point[3] < 0 || right_point[3] < 0
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            @debug "[MP] Triangulation is behind cameras." maxlog=10
            continue
        end
        # Remove MapPoint with high reprojection error.
        lrepr = norm(project(cam, left_point[1:3]) .- obup)
        rrepr = norm(project(cam, right_point[1:3]) .- kpup)
        if lrepr > max_error || rrepr > max_error
            parallax > 20 && remove_mappoint_obs!(
                mapper.map_manager, observer_kp.id, frame.kfid,
            )
            @debug "[MP] Triangulation has too big repr error $lrepr, $rrepr" maxlog=10
            continue
        end
        # 3D pose is good, update MapPoint and related Frames.
        wpt = project_camera_to_world(observer_kf, left_point)[1:3]
        update_mappoint!(mapper.map_manager, kp.id, wpt, 1.0 / left_point[3])
        good += 1
    end
    @debug "[Mapper] Temporal triangulation: $good/$candidates good KeyPoints."
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
