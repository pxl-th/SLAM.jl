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
            sleep(0.1)
            continue
        end

        local_bundle_adjustment!(estimator, new_kf)
        map_filtering!(estimator, new_kf)
    end
    @debug "[ES] Exit required."
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

        @debug "[ES] Popping queue $(length(estimator.frame_queue))."
        estimator.new_kf_available = false
        popfirst!(estimator.frame_queue)
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
        for kp in get_3d_keypoints(kf)
            push!(keypoint_ids_to_optimize, kp.id)
        end
    end
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
    map_points = OrderedDict{Int64, OrderedDict{Int64, Point2f}}() # {mpid → {kfid → pixel}}
    bad_keypoints = Set{Int64}() # kpid/mpid
    n_constants = 0
    n_pixels = 0

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
            # If not in the local map,
            # then add it from the global FramesMap as a constant.
            if observer_id in keys(local_keyframes)
                observer_kf = local_keyframes[observer_id]
            else
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
    map_points, bad_keypoints, n_pixels, n_constants
end

function _convert_to_matrix_form(
    extrinsics, constant_extrinsics, map_points, map_manager, n_pixels,
)
    # Convert data to the Bundle-Adjustment format.
    extrinsics_matrix = Matrix{Float64}(undef, 6, length(extrinsics))
    constants_matrix = Vector{Bool}(undef, length(extrinsics))
    points_matrix = Matrix{Float64}(undef, 3, length(map_points))
    pixels_matrix = Matrix{Float64}(undef, 2, n_pixels)

    points_ids, extrinsics_ids = Int64[], Int64[]
    extrinsics_order = Dict{Int64, Int64}() # kfid -> nkf
    extrinsic_id, pixel_id, point_id = 1, 1, 1

    # Convert to matrix form.
    for (mpid, mplink) in map_points
        mp = get_mappoint(map_manager, mpid)
        @assert mp ≢ nothing
        points_matrix[:, point_id] .= get_position(mp)

        for (kfid, pixel) in mplink
            push!(points_ids, point_id)
            pixels_matrix[:, pixel_id] .= pixel
            pixel_id += 1

            if kfid in keys(extrinsics_order)
                push!(extrinsics_ids, extrinsics_order[kfid])
                continue
            end

            constants_matrix[extrinsic_id] = constant_extrinsics[kfid]
            extrinsics_matrix[:, extrinsic_id] .= extrinsics[kfid]
            extrinsics_order[kfid] = extrinsic_id
            push!(extrinsics_ids, extrinsic_id)
            extrinsic_id += 1
        end
        point_id += 1
    end
    (
        extrinsics_matrix, constants_matrix, extrinsics_order,
        points_matrix, pixels_matrix, extrinsics_ids, points_ids,
    )
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

    covisibility_map = deepcopy(get_covisible_map(new_frame))
    covisibility_map[new_frame.kfid] = new_frame.nb_3d_kpts
    # Specifies maximum KeyFrame id in the covisibility graph.
    # To avoid adding observer to the BA problem,
    # that is more recent than the `new_frame`.
    max_kfid = new_frame.kfid

    (
        extrinsics, constant_extrinsics, local_keyframes,
        keypoint_ids_to_optimize, n_constants,
    ) = _gather_extrinsics!(
        covisibility_map, estimator.map_manager, estimator.params, new_frame,
    )

    map_points, bad_keypoints, n_pixels, n_constants_tmp = _gather_mappoints!(
        extrinsics, constant_extrinsics,
        keypoint_ids_to_optimize, estimator.map_manager,
        local_keyframes, max_kfid,
    )
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
        estimator.map_manager, n_pixels,
    )

    new_extrinsics, new_points, ie, fe, outliers = bundle_adjustment(
        new_frame.camera, extrinsics_matrix, points_matrix,
        pixels_matrix, points_ids, extrinsics_ids;
        constant_extrinsics=constants_matrix,
        iterations=10,
    )

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
                @view(new_extrinsics[:, nkfid]),
            )
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
                @view(new_points[:, pid]),
            )
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
