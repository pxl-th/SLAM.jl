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

        @debug "[ES] Starting LBA."
        local_bundle_adjustment!(estimator, new_kf)
        @debug "[ES] Starting Map filtering."
        map_filtering!(estimator, new_kf)
        @debug "[ES] Finished Map filtering."
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
            @debug "[ES] Empty queue."
            return nothing
        end

        # TODO if more than 1 frame in queue, add them to ba anyway.

        @debug "[ES] Popping queue $(length(estimator.frame_queue))."
        estimator.new_kf_available = false
        popfirst!(estimator.frame_queue)
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

    # Fix extrinsics of at least two KeyFrames.
    n_fixed_keyframes = 2
    # Get `new_frame`'s covisible KeyFrames.
    map_cov_kf = get_covisible_map(new_frame) |> deepcopy
    map_cov_kf[new_frame.kfid] = new_frame.nb_3d_kpts
    @debug "[ES] Initial covisibility size $(length(map_cov_kf))."

    bad_keypoints = Set{Int64}() # kpid/mpid
    local_keyframes = Dict{Int64, Frame}() # kfid → Frame
    map_points = OrderedDict{Int64, OrderedDict{Int64, Point2f}}() # {mpid → {kfid → pixel}}
    extrinsics = Dict{Int64, NTuple{6, Float64}}() # kfid → extrinsics
    kp_ids_optimize = Set{Int64}() # kpid
    constant_extrinsics = Dict{Int64, Bool}() # kfid → is constant
    n_constants = 0

    # Specifies maximum KeyFrame id in the covisibility graph.
    # To avoid adding observer to the BA problem,
    # that is more recent than the `new_frame`.
    max_kfid = new_frame.kfid

    # Go through all KeyFrames in covisibility graph, get their extrinsics,
    # mark them constant/non-constant, get their 3D Keypoints.
    for (kfid, cov_score) in map_cov_kf
        if !(kfid in keys(estimator.map_manager.frames_map))
            remove_covisible_kf!(new_frame, kfid)
            continue
        end

        kf = get_keyframe(estimator.map_manager, kfid)
        local_keyframes[kfid] = kf
        extrinsics[kfid] = get_cw_ba(kf)

        if cov_score < estimator.params.min_cov_score || kfid == 0
            constant_extrinsics[kfid] = true
            n_constants += 1
            continue
        end

        constant_extrinsics[kfid] = false
        # Add ids of the 3D Keypoints to optimize.
        for (kpid, kp) in kf.keypoints
            kp.is_3d && push!(kp_ids_optimize, kpid)
        end
    end
    @debug "[ES] N 3D Keypoints to optimize $(length(kp_ids_optimize))."
    @debug "[ES] Max KF id $max_kfid."

    n_pixels = 0

    # Go through all 3D Keypoints to optimize.
    # Link MapPoint with the observer KeyFrames
    # and their corresponding pixel coordinates.
    for kpid in kp_ids_optimize
        kpid in keys(estimator.map_manager.map_points) || continue
        mp = get_mappoint(estimator.map_manager, kpid)
        if mp ≡ nothing
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
                if !(observer_id in keys(estimator.map_manager.frames_map))
                    remove_mappoint_obs!(estimator.map_manager, kpid, observer_id)
                    continue
                end
                observer_kf = get_keyframe(estimator.map_manager, observer_id)
                local_keyframes[observer_id] = observer_kf
                extrinsics[observer_id] = get_cw_ba(observer_kf)

                constant_extrinsics[observer_id] = true
                n_constants += 1
            end
            # Get corresponding pixel coordinate and link it to the MapPoint.
            if !(kpid in keys(observer_kf.keypoints))
                remove_mappoint_obs!(estimator.map_manager, kpid, observer_id)
                continue
            end

            observer_kp = get_keypoint(observer_kf, kpid)
            mplink[observer_id] = observer_kp.undistorted_pixel
            n_pixels += 1
        end
        map_points[mp.id] = mplink
    end

    # Ensure there are at least 2 fixed Keyframes.
    if (n_constants < 2 && length(extrinsics) > 2)
        for kfid in keys(constant_extrinsics)
            constant_extrinsics[kfid] && continue
            constant_extrinsics[kfid] = true
            n_constants += 1
            n_constants == 2 && break
        end
    end

    @debug "[ES] Covisibility size with observers $(length(local_keyframes))."
    @debug "[ES] N Pixels $n_pixels."

    # Convert data to the Bundle-Adjustment format.
    constants_matrix = Vector{Bool}(undef, length(extrinsics))
    extrinsics_matrix = Matrix{Float64}(undef, 6, length(extrinsics))
    points_matrix = Matrix{Float64}(undef, 3, length(map_points))
    pixels_matrix = Matrix{Float64}(undef, 2, n_pixels)

    points_ids, extrinsics_ids = Int64[], Int64[]
    extrinsics_order = Dict{Int64, Int64}() # kfid -> nkf

    extrinsic_id, pixel_id, point_id = 1, 1, 1

    # Convert to matrix form.
    for (mpid, mplink) in map_points
        mp = get_mappoint(estimator.map_manager, mpid)
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

    @debug "[ES] N Extrinsics $(extrinsic_id - 1)."
    @debug "[ES] N Pixels $(pixel_id - 1)."
    @debug "[ES] N Point id $(point_id - 1)."
    @debug "[ES] N Constant KeyFrames $(sum(constants_matrix))."

    new_extrinsics, new_points, ie, fe, outliers = bundle_adjustment(
        new_frame.camera, extrinsics_matrix, points_matrix,
        pixels_matrix, points_ids, extrinsics_ids;
        constant_extrinsics=constants_matrix,
        iterations=10, show_trace=true,
    )
    @debug "[ES] N Outliers $(sum(outliers))."
    @debug "[ES] BA error $ie → $fe."

    # Select outliers and prepare them for removal.
    kfmp_outliers = Dict{Int64, Int64}()
    pixel_id = 1
    for (mpid, mplink) in map_points
        for (kfid, _) in mplink
            if !outliers[pixel_id]
                pixel_id += 1
                continue
            end

            pixel_id += 1
            kfid in keys(map_cov_kf) &&
                remove_mappoint_obs!(estimator.map_manager, mpid, kfid)
            kfid == estimator.map_manager.current_frame.kfid &&
                remove_obs_from_current_frame!(estimator.map_manager, mpid)

            push!(bad_keypoints, mpid)
        end
    end

    @debug "[ES] N Bad Keypoints $(length(bad_keypoints))"

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

        if !(kfid in keys(estimator.map_manager.frames_map))
            remove_covisible_kf!(new_keyframe, kfid)
            continue
        end

        kf = get_keyframe(estimator.map_manager, kfid)
        if kf.nb_3d_kpts < estimator.params.min_cov_score ÷ 2
            remove_keyframe!(estimator.map_manager, kfid)
            @debug "[MF] Removed KeyFrame $kfid."
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
            @debug "[MF] Removed KeyFrame $kfid."
            n_removed += 1
        end
    end
    @debug "[MF] Removed $n_removed KeyFrames."
end

function reset!(estimator::Estimator)
    lock(estimator.queue_lock) do
        estimator.new_kf_available = false
        empty!(estimator.frame_queue)
    end
end
