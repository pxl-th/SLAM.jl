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
        catch e
            showerror(stdout, e); println()
            display(stacktrace(catch_backtrace())); println()
        end
        # map_filtering!(estimator, new_kf)
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

function _gather_extrinsics!(covisibility_map, map_manager, params, new_frame)
    extrinsics = Dict{Int64, NTuple{6, Float64}}() # kfid → extrinsics
    constant_extrinsics = Dict{Int64, Bool}() # kfid → is constant
    local_keyframes = Dict{Int64, Frame}() # kfid → Frame
    keypoint_ids_to_optimize = Set{Int64}()
    n_constants = 0

    # Go through all KeyFrames in covisibility graph, get their extrinsics,
    # mark them constant/non-constant, get their 3D Keypoints.
    for (kfid, cov_score) in covisibility_map
        kf = get_keyframe(map_manager, kfid)
        if kf ≡ nothing
            remove_covisible_kf!(new_frame, kfid)
            continue
        end

        local_keyframes[kfid] = kf
        extrinsics[kfid] = get_cw_ba(kf)

        if cov_score < params.min_cov_score || kfid == 0
            constant_extrinsics[kfid] = true
            n_constants += 1
            continue
        end
        constant_extrinsics[kfid] = false
        # Add ids of the 3D Keypoints to optimize.
        union!(keypoint_ids_to_optimize, get_3d_keypoints_ids(kf))
    end
    @info "[ES] OLD Constants 1: $n_constants."
    (
        extrinsics, constant_extrinsics, local_keyframes,
        keypoint_ids_to_optimize, n_constants,
    )
end

function _gather_mappoints!(
    extrinsics, constant_extrinsics,
    keypoint_ids_to_optimize, map_manager,
    local_keyframes, max_kfid,
)
    # {mpid → {kfid → pixel}}
    map_points = OrderedDict{Int64, OrderedDict{Int64, Point2f}}()
    bad_keypoints = Set{Int64}() # kpid/mpid
    n_pixels, n_constants = 0, 0

    # Go through all 3D Keypoints to optimize.
    # Link MapPoint with the observer KeyFrames
    # and their corresponding pixel coordinates.
    for kpid in keypoint_ids_to_optimize
        mp = get_mappoint(map_manager, kpid)
        if mp ≡ nothing
            continue
        end
        if is_bad!(mp)
            push!(bad_keypoints, kpid)
            continue
        end

        # Link observer KeyFrames with the MapPoint.
        # Add observer KeyFrames as constants, if not yet added.
        mplink = Dict{Int64, Point2f}()
        for observer_id in get_observers(mp)
            observer_id > max_kfid && continue
            # Get observer KeyFrame.
            # If not in the local map, then add it from the global
            # frames map as a constant.
            observer_kf = get(local_keyframes, observer_id, nothing)
            if observer_kf ≡ nothing
                observer_kf = get_keyframe(map_manager, observer_id)
                if observer_kf ≡ nothing
                    remove_mappoint_obs!(map_manager, kpid, observer_id)
                    continue
                end

                local_keyframes[observer_id] = observer_kf
                extrinsics[observer_id] = get_cw_ba(observer_kf)
                constant_extrinsics[observer_id] = true
                n_constants += 1
            end

            # Get corresponding pixel coordinate and link it to the MapPoint.
            observer_kp_unpx = get_keypoint_unpx(observer_kf, kpid)
            if observer_kp_unpx ≡ nothing
                remove_mappoint_obs!(map_manager, kpid, observer_id)
                continue
            end
            mplink[observer_id] = observer_kp_unpx
            n_pixels += 1
        end
        map_points[mp.id] = mplink
    end
    @info "[ES] OLD Total observations: $n_pixels."
    @info "[ES] OLD Poses: $(length(extrinsics))."
    @info "[ES] OLD Constants: $n_constants."
    map_points, bad_keypoints, n_pixels, n_constants
end

"""
Convert data to the Bundle-Adjustment format.
"""
function _convert_to_matrix_form(
    extrinsics, constant_extrinsics, map_points, map_manager, n_pixels,
)
    extrinsics_matrix = Matrix{Float64}(undef, 6, length(extrinsics))
    constants_matrix = Vector{Bool}(undef, length(extrinsics))
    points_matrix = Matrix{Float64}(undef, 3, length(map_points))
    pixels_matrix = Matrix{Float64}(undef, 2, n_pixels)

    i_point, i_extrinsic = 1, 1
    points_ids = Vector{Int64}(undef, n_pixels)
    extrinsics_ids = Vector{Int64}(undef, n_pixels)

    extrinsic_id, pixel_id, point_id = 1, 1, 1
    extrinsics_order = Dict{Int64, Int64}() # kfid -> nkf

    # Convert to matrix form.
    for (mpid, mplink) in map_points
        mp = get_mappoint(map_manager, mpid)
        points_matrix[:, point_id] .= get_position(mp)

        for (kfid, pixel) in mplink
            points_ids[i_point] = point_id
            i_point += 1
            pixels_matrix[:, pixel_id] .= pixel
            pixel_id += 1

            if kfid in keys(extrinsics_order)
                extrinsics_ids[i_extrinsic] = extrinsics_order[kfid]
                i_extrinsic += 1
                continue
            end

            constants_matrix[extrinsic_id] = constant_extrinsics[kfid]
            extrinsics_matrix[:, extrinsic_id] .= extrinsics[kfid]
            extrinsics_order[kfid] = extrinsic_id

            extrinsics_ids[i_extrinsic] = extrinsic_id
            i_extrinsic += 1
            extrinsic_id += 1
        end
        point_id += 1
    end
    (
        extrinsics_matrix, constants_matrix, extrinsics_order,
        points_matrix, pixels_matrix, extrinsics_ids, points_ids,
    )
end

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
    pixels::Vector{Float64}

    poses_ids::Vector{Int64}
    points_ids::Vector{Int64}

    poses_remap::Vector{Int64} # order id → kfid
    points_remap::Vector{Int64} # order id → mpid
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

    for (co_kfid, score) in covisibility_map
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
            kpid in processed_keypoints_ids && continue
            push!(processed_keypoints_ids, kpid)

            mp = get_mappoint(map_manager, kpid)
            mp ≡ nothing && continue
            is_bad!(mp) && (push!(bad_keypoints, kpid); continue)
            # TODO skip, if mp is_weak or get rid of is_weak entirelly?

            mp_order_id = length(map_points) + 1
            mp_position = get_position(mp)
            map_points[kpid] = (mp_order_id, mp_position)
            push!(points_remap, kpid)

            # For each observer, add observation: px ← (mp, pose).
            constant_observers = 0
            for ob_kfid in get_observers(mp)
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

                # Allow only two constant observer KeyFrames
                # outside of covisibility graph to reduce BA complexity.
                if !in_covmap
                    if constant_observers < 2
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
            end
        end
    end
    n_observations = length(observations)
    n_poses, n_points = length(poses), length(map_points)
    point_shift = n_poses * 6

    @info "[ES] Total Observations: $(length(observations))."
    @info "[ES] Poses: $(length(poses)), Covisibility: $(length(covisibility_map))."
    @info "[ES] Constants: $(length(constant_poses))."

    θ = Vector{Float64}(undef, point_shift + n_points * 3)
    θconst = Vector{Bool}(undef, n_poses)
    poses_ids = Vector{Int64}(undef, n_observations)
    points_ids = Vector{Int64}(undef, n_observations)
    pixels = Vector{Float64}(undef, n_observations * 2)

    processed_poses = fill(false, n_poses)
    processed_points = fill(false, n_points)

    for (oi, observation) in enumerate(observations)
        p = (oi - 1) * 2
        pixels[(p + 1):(p + 2)] .= observation.pixel
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

    @assert sum(processed_poses) == n_poses "Poses: $(sum(processed_poses)) ↔ $n_poses. \n"
    @assert sum(processed_points) == n_points "Points: $(sum(processed_points)) ↔ $n_points. \n"

    LocalBACache(
        observations, bad_keypoints, θ, θconst, pixels,
        poses_ids, points_ids, poses_remap, points_remap,
    )
end

"""
if obs is outlier
    -> remove mappoint obs for respective keyframe
    -> add mp id to bad keypoints

for mp
    -> if nothing, remove it
    -> if less than 3 observers, remove it
    -> otherwise update position

for bad kps
    -> remove mappoints
"""
function _update_ba_parameters!(
    map_manager::MapManager, cache::LocalBACache, current_kfid,
)
    n_poses = length(cache.poses_remap)
    n_point = length(cache.points_remap)
    points_shift = n_poses * 6

    for (i, kfid) in enumerate(cache.poses_remap)
        p = (i - 1) * 6
        kf = get_keyframe(map_manager, kfid)
        @assert kf ≢ nothing

        old_pose = get_cw_ba(kf)
        new_pose = cache.θ[(p + 1):(p + 6)]
        @assert all(isapprox.(new_pose, old_pose))
        set_cw_ba!(kf, new_pose)
    end

    for i in 1:length(cache.observations)
        cache.outliers[i] || continue

        obs = cache.observations[i]
        obs.in_covmap && remove_mappoint_obs!(map_manager, obs.mpid, obs.kfid)
        obs.kfid == current_kfid &&
            remove_obs_from_current_frame!(map_manager, obs.mpid)

        push!(cache.bad_keypoints, obs.mpid)
    end

    for (i, mpid) in enumerate(cache.points_remap)
        mp = get_mappoint(map_manager, mpid)
        @assert mp ≢ nothing
        if is_bad!(mp)
            remove_mappoint!(map_manager, mpid)
            mpid in cache.bad_keypoints && pop!(cache.bad_keypoints, mpid)
        else
            p = points_shift + (i - 1) * 3
            old_position = get_position(mp)
            new_position = cache.θ[(p + 1):(p + 3)]
            @assert all(isapprox.(new_position, old_position))
            set_position!(mp, new_position)
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
    # Specifies maximum KeyFrame id in the covisibility graph.
    # To avoid adding observer to the BA problem,
    # that is more recent than the `new_frame`.
    max_kfid = new_frame.kfid

    try
        t1 = time()
        cache = _get_ba_parameters(
            estimator.map_manager, new_frame, covisibility_map,
            estimator.params.min_cov_score)

        _update_ba_parameters!(estimator.map_manager, cache, new_frame.kfid)
        t2 = time()
        @info "[ES] NEW BA Time: $(t2 - t1) seconds."
    catch e
        showerror(stdout, e)
        display(stacktrace(catch_backtrace()))
    end

    t1 = time()
    (
        extrinsics, constant_extrinsics, local_keyframes,
        keypoint_ids_to_optimize, n_constants,
    ) = _gather_extrinsics!(
        covisibility_map, estimator.map_manager, estimator.params, new_frame)

    map_points, bad_keypoints, n_pixels, n_constants_tmp = _gather_mappoints!(
        extrinsics, constant_extrinsics,
        keypoint_ids_to_optimize, estimator.map_manager,
        local_keyframes, max_kfid)
    n_constants += n_constants_tmp

    # Ensure there are at least 2 fixed Keyframes.
    if n_constants < 2 && length(extrinsics) > 2
        for kfid in keys(extrinsics)
            constant_extrinsics[kfid] && continue
            constant_extrinsics[kfid] = true
            n_constants += 1
            n_constants == 2 && break
        end
    end

    (
        extrinsics_matrix, constants_matrix, extrinsics_order,
        points_matrix, pixels_matrix, extrinsics_ids, points_ids,
    ) = _convert_to_matrix_form(
        extrinsics, constant_extrinsics, map_points,
        estimator.map_manager, n_pixels)

    new_extrinsics, new_points, error, outliers = bundle_adjustment(
        new_frame.camera, extrinsics_matrix, points_matrix,
        pixels_matrix, points_ids, extrinsics_ids;
        constant_extrinsics=constants_matrix, iterations=10)

    # Select outliers and prepare them for removal.
    pixel_id = 1
    for (mpid, mplink) in map_points
        for (kfid, _) in mplink
            if !outliers[pixel_id]
                pixel_id += 1
                continue
            end

            pixel_id += 1
            kfid in keys(covisibility_map) &&
                remove_mappoint_obs!(estimator.map_manager, mpid, kfid)
            kfid == estimator.map_manager.current_frame.kfid &&
                remove_obs_from_current_frame!(estimator.map_manager, mpid)

            push!(bad_keypoints, mpid)
        end
    end

    lock(estimator.map_manager.map_lock) do
        # Update KeyFrame poses.
        for (kfid, nkfid) in extrinsics_order
            constant_extrinsics[kfid] && continue
            set_cw_ba!(
                get_keyframe(estimator.map_manager, kfid),
                @view(new_extrinsics[:, nkfid]))
        end

        for (pid, mpid) in enumerate(keys(map_points))
            if !(mpid in keys(estimator.map_manager.map_points))
                mpid in bad_keypoints && pop!(bad_keypoints, mpid)
                continue
            end

            mp = get_mappoint(estimator.map_manager, mpid)
            if mp ≡ nothing
                remove_mappoint!(estimator.map_manager, mpid)
                mpid in bad_keypoints && pop!(bad_keypoints, mpid)
                continue
            end

            # MapPoint culling.
            # Remove MapPoint, if it has < 3 observers,
            # not observed by the current frames_map
            # and was added less than 3 keyframes ago.
            # Meaning it was unrealiable.
            if get_observers_number(mp) < 3
                if mp.kfid < new_frame.kfid - 3 && !mp.is_observed
                    remove_mappoint!(estimator.map_manager, mpid)
                    mpid in bad_keypoints && pop!(bad_keypoints, mpid)
                    continue
                end
            end

            # MapPoint is good, update its position.
            set_position!(
                get_mappoint(estimator.map_manager, mpid),
                @view(new_points[:, pid]))
        end

        # MapPoint culling for bad observations.
        for mpid in bad_keypoints
            mpid in keys(estimator.map_manager.map_points) &&
                mpid in keys(map_points) || continue

            mp = get_mappoint(estimator.map_manager, mpid)
            if mp ≡ nothing
                remove_mappoint!(estimator.map_manager, mpid)
                continue
            end

            if length(mp.observer_keyframes_ids) < 3
                if mp.kfid < new_frame.kfid - 3 && !mp.is_observed
                    remove_mappoint!(estimator.map_manager, mpid)
                end
            end
        end
    end

    estimator.params.local_ba_on = false

    t2 = time()
    @info "[ES] OLD BA Time: $(t2 - t1) seconds."
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
            @debug "[ES] Removed KeyFrame $kfid."
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
            @debug "[ES] Removed KeyFrame $kfid."
            n_removed += 1
        end
    end
    @debug "[ES] Removed $n_removed KeyFrames."
end

function reset!(estimator::Estimator)
    lock(estimator.queue_lock) do
        estimator.new_kf_available = false
        empty!(estimator.frame_queue)
    end
end
