struct Observation
    pixel::Point2f

    point::Point3f
    pose::NTuple{6, Float64}

    point_order::Int64
    pose_order::Int64

    constant::Bool
    in_covmap::Bool

    kfid::Int64
    mpid::Int64
end

struct LocalBACache
    observations::Vector{Observation}
    outliers::Vector{Bool} # same length as observations
    bad_keypoints::Set{Int64}

    θ::Vector{Float64}
    θconst::Vector{Bool}
    pixels::Matrix{Float64}

    poses_ids::Vector{Int64}
    points_ids::Vector{Int64}

    poses_remap::Vector{Int64} # order id → kfid
    points_remap::Vector{Int64} # order id → mpid
end

mutable struct Estimator
    map_manager::MapManager
    params::Params

    frame_queue::Vector{Frame}
    new_kf_available::Bool
    exit_required::Bool

    queue_lock::ReentrantLock
end

function Estimator(map_manager::MapManager, params::Params)
    Estimator(
        map_manager, params, Frame[],
        false, false, ReentrantLock(),
    )
end

function run!(estimator::Estimator)
    while !estimator.exit_required
        new_kf = get_new_kf!(estimator)
        if new_kf ≡ nothing
            sleep(1e-2)
            continue
        end

        try
            local_bundle_adjustment!(estimator, new_kf)
            map_filtering!(estimator, new_kf)
        catch e
            showerror(stdout, e); println()
            display(stacktrace(catch_backtrace())); println()
        end
    end
    @info "[ES] Exit required."
end

function add_new_kf!(estimator::Estimator, frame::Frame)
    lock(estimator.queue_lock) do
        push!(estimator.frame_queue, frame)
        estimator.new_kf_available = true
    end
end

function get_new_kf!(estimator::Estimator)
    lock(estimator.queue_lock) do
        if isempty(estimator.frame_queue)
            estimator.new_kf_available = false
            return nothing
        end

        # TODO if more than 1 frame in queue, add them to ba anyway.
        estimator.new_kf_available = false
        kf = popfirst!(estimator.frame_queue)
        @info "[ES] Popping queue $(length(estimator.frame_queue)): $(kf.kfid)."
        kf
    end
end

function LocalBACache(
    observations, bad_keypoints, θ, θconst, pixels,
    poses_ids, points_ids, poses_remap, points_remap,
)
    outliers = fill(false, length(observations))
    # outliers = Vector{Bool}(undef, length(observations))
    LocalBACache(
        observations, outliers, bad_keypoints, θ, θconst, pixels,
        poses_ids, points_ids, poses_remap, points_remap,
    )
end

function _get_ba_parameters(
    map_manager::MapManager, frame::Frame,
    covisibility_map::Dict{Int64, Int64}, min_cov_score,
)
    # poses: kfid → (order id, θ).
    poses = Dict{Int64, Tuple{Int64, NTuple{6, Float64}}}()
    constant_poses = Set{Int64}()
    # map_points: mpid → (order id, point).
    map_points = Dict{Int64, Tuple{Int64, Point3f}}()

    processed_keypoints_ids = Set{Int64}()
    bad_keypoints = Set{Int64}()

    observations = Vector{Observation}(undef, 0)
    sizehint!(observations, 1000)

    poses_remap = Vector{Int64}(undef, 0)
    points_remap = Vector{Int64}(undef, 0)
    sizehint!(poses_remap, 10)
    sizehint!(points_remap, 1000)

    too_much = false

    for (co_kfid, score) in covisibility_map
        too_much && break

        co_frame = get_keyframe(map_manager, co_kfid)
        co_frame ≡ nothing && (remove_covisible_kf!(frame, co_kfid); continue)
        (co_kfid > frame.kfid || get_3d_keypoints_nb(co_frame) == 0 ||
            score == 0) && continue

        if !(co_kfid in keys(poses))
            pose_order_id = length(poses) + 1
            poses[co_kfid] = (pose_order_id, get_cw_ba(co_frame))
            push!(poses_remap, co_kfid)

            if !(co_kfid in constant_poses)
                is_constant = score < min_cov_score || co_kfid == 0
                is_constant && (push!(constant_poses, co_kfid); continue)
            end
        end

        for kpid in get_3d_keypoints_ids(co_frame)
            too_much && break

            kpid in processed_keypoints_ids && continue
            push!(processed_keypoints_ids, kpid)

            mp = get_mappoint(map_manager, kpid)
            mp ≡ nothing && continue
            is_bad!(mp) && (push!(bad_keypoints, kpid); continue)

            mp_order_id = length(map_points) + 1
            mp_position = get_position(mp)
            map_points[kpid] = (mp_order_id, mp_position)
            push!(points_remap, kpid)

            # For each observer, add observation: px ← (mp, pose).
            normal_observers, constant_observers = 0, 0
            for ob_kfid in get_observers(mp)
                normal_observers == 3 && constant_observers == 2 && break
                ob_kfid > frame.kfid && continue

                ob_frame = get_keyframe(map_manager, ob_kfid)
                if ob_frame ≡ nothing
                    remove_mappoint_obs!(map_manager, kpid, ob_kfid)
                    continue
                end
                ob_pixel = get_keypoint_unpx(ob_frame, kpid)
                if ob_pixel ≡ nothing
                    remove_mappoint_obs!(map_manager, kpid, ob_kfid)
                    continue
                end

                in_processed = ob_kfid in keys(poses)
                in_covmap = ob_kfid in keys(covisibility_map)
                in_constants = ob_kfid in constant_poses

                is_constant = ob_kfid == 0 || in_constants || !in_covmap
                !is_constant && in_covmap && (is_constant =
                    covisibility_map[ob_kfid] < min_cov_score;)

                # Allow only 2 constant and 3 normal observer KeyFrames
                # to reduce BA complexity.
                if in_covmap
                    if normal_observers < 2
                        normal_observers += 1
                    else continue end
                else
                    if constant_observers < 1
                        constant_observers += 1
                    else continue end
                end

                if in_processed
                    pose_order_id, ob_pose = poses[ob_kfid]
                else
                    ob_pose = get_cw_ba(ob_frame)
                    is_constant && push!(constant_poses, ob_kfid)
                    pose_order_id = length(poses) + 1
                    poses[ob_kfid] = (pose_order_id, ob_pose)
                    push!(poses_remap, ob_kfid)
                end

                push!(observations, Observation(
                    ob_pixel, mp_position, ob_pose,
                    mp_order_id, pose_order_id,
                    is_constant, in_covmap, ob_kfid, kpid,
                ))
                length(observations) ≥ 2000 && (too_much = true; break)
            end
        end
    end

    n_observations = length(observations)
    n_poses, n_points = length(poses), length(map_points)
    point_shift = n_poses * 6
    @info "[ES] BA Observations: $n_observations."
    @info "[ES] BA Poses: $n_poses."
    @info "[ES] BA Points: $n_points."

    θ = Vector{Float64}(undef, point_shift + n_points * 3)
    θconst = Vector{Bool}(undef, n_poses)
    poses_ids = Vector{Int64}(undef, n_observations)
    points_ids = Vector{Int64}(undef, n_observations)
    pixels = Matrix{Float64}(undef, 2, n_observations)

    processed_poses = fill(false, n_poses)
    processed_points = fill(false, n_points)

    @inbounds for (oi, observation) in enumerate(observations)
        pixels[:, oi] .= observation.pixel
        poses_ids[oi] = observation.pose_order
        points_ids[oi] = observation.point_order

        if !processed_poses[observation.pose_order]
            processed_poses[observation.pose_order] = true
            p = (observation.pose_order - 1) * 6
            θ[(p + 1):(p + 6)] .= observation.pose
            θconst[observation.pose_order] = observation.constant
        end
        if !processed_points[observation.point_order]
            processed_points[observation.point_order] = true
            p = point_shift + (observation.point_order - 1) * 3
            θ[(p + 1):(p + 3)] .= observation.point
        end
    end

    LocalBACache(
        observations, bad_keypoints, θ, θconst, pixels,
        poses_ids, points_ids, poses_remap, points_remap,
    )
end

function _update_ba_parameters!(
    map_manager::MapManager, cache::LocalBACache, current_kfid,
)
    points_shift = length(cache.poses_remap) * 6

    @inbounds for (i, kfid) in enumerate(cache.poses_remap)
        p = (i - 1) * 6
        kf = get_keyframe(map_manager, kfid)
        @assert kf ≢ nothing
        set_cw_ba!(kf, @view(cache.θ[(p + 1):(p + 6)]))
    end

    @inbounds for i in 1:length(cache.observations)
        cache.outliers[i] || continue

        obs = cache.observations[i]
        obs.in_covmap && remove_mappoint_obs!(map_manager, obs.mpid, obs.kfid)
        obs.kfid == current_kfid &&
            remove_obs_from_current_frame!(map_manager, obs.mpid)

        push!(cache.bad_keypoints, obs.mpid)
    end

    @inbounds for (i, mpid) in enumerate(cache.points_remap)
        mp = get_mappoint(map_manager, mpid)
        @assert mp ≢ nothing
        if is_bad!(mp)
            remove_mappoint!(map_manager, mpid)
            mpid in cache.bad_keypoints && pop!(cache.bad_keypoints, mpid)
        else
            p = points_shift + (i - 1) * 3
            set_position!(mp, @view(cache.θ[(p + 1):(p + 3)]))
        end
    end

    for bad_kpid in cache.bad_keypoints
        mp = get_mappoint(map_manager, bad_kpid)
        mp ≡ nothing && continue
        is_bad!(mp) && remove_mappoint!(map_manager, mp)
    end
end

"""
Perform Bundle-Adjustment on the new frame and its covisibility graph.

Minimize error function over all KeyFrame's extrinsic parameters
in the covisibility graph and their corresponding MapPoint's positions.
Afterwards, update these parameters.
"""
function local_bundle_adjustment!(estimator::Estimator, new_frame::Frame)
    if new_frame.nb_3d_kpts < estimator.params.min_cov_score
        @warn "[ES] Not enough 3D keypoints for BA: $(new_frame.nb_3d_kpts)."
        return
    end

    estimator.params.local_ba_on = true
    covisibility_map = get_covisible_map(new_frame)
    covisibility_map[new_frame.kfid] = new_frame.nb_3d_kpts

    t1 = time()

    cache = _get_ba_parameters(
        estimator.map_manager, new_frame, covisibility_map,
        estimator.params.min_cov_score)
    bundle_adjustment!(cache, new_frame.camera)

    lock(estimator.map_manager.map_lock)
    try
        _update_ba_parameters!(estimator.map_manager, cache, new_frame.kfid)
    catch e
        showerror(stdout, e); println()
        display(stacktrace(catch_backtrace())); println()
    finally
        unlock(estimator.map_manager.map_lock)
    end

    t2 = time()
    @info "[ES] NEW BA Time: $(t2 - t1) seconds."

    estimator.params.local_ba_on = false
end

"""
Filter out KeyFrames that share too many MapPoints with other KeyFrames
in the covisibility graph. Since they are not informative.
"""
function map_filtering!(estimator::Estimator, new_keyframe::Frame)
    estimator.params.filtering_ratio ≥ 1 && return
    new_keyframe.kfid < 20 && return

    n_removed = 0
    for kfid in keys(get_covisible_map(new_keyframe))
        estimator.new_kf_available && break
        kfid == 0 && break
        kfid ≥ new_keyframe.kfid && continue

        if !has_keyframe(estimator.map_manager, kfid)
            remove_covisible_kf!(new_keyframe, kfid)
            continue
        end

        kf = get_keyframe(estimator.map_manager, kfid)
        if kf.nb_3d_kpts < estimator.params.min_cov_score ÷ 2
            remove_keyframe!(estimator.map_manager, kfid)
            @info "[ES] Removed KeyFrame $kfid."
            n_removed += 1
            continue
        end

        n_good, n_total = 0, 0
        for kp in get_3d_keypoints(kf)
            if !(kp.id in keys(estimator.map_manager.map_points))
                remove_mappoint_obs!(estimator.map_manager, kp.id, kfid)
                continue
            end
            mp = get_mappoint(estimator.map_manager, kp.id)
            mp ≡ nothing && continue

            get_observers_number(mp) > 4 && (n_good += 1;)
            n_total += 1;

            estimator.new_kf_available && break
        end

        ratio = n_good / n_total
        if ratio > estimator.params.filtering_ratio
            remove_keyframe!(estimator.map_manager, kfid)
            @info "[ES] Removed KeyFrame $kfid."
            n_removed += 1
        end
    end
    @info "[ES] Removed $n_removed KeyFrames."
end

function reset!(estimator::Estimator)
    lock(estimator.queue_lock) do
        estimator.new_kf_available = false
        empty!(estimator.frame_queue)
    end
end
