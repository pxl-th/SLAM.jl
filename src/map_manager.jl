struct MapPoint
    id::Int64
    """
    Anchored position: kfid + position + inv_depth.
    """
    kfid::Int64
    """
    Set of KeyFrame's ids that are visible from this MapPoint.
    """
    observer_keyframes_ids::Set{Int64}
    """
    Mean descriptor.
    """
    descriptor::BitVector
    keyframes_descriptors::Dict{Int64, BitVector}
    """
    Anchored position: kfid + position + inv_depth.
    """
    position::Point3f0
    inv_depth::Float32
    """
    True if the MapPoint has been initialized.
    """
    is_3d::Bool
    """
    True if the MapPoint is visible in the current frame.
    """
    is_observed::Bool
end

MapPoint(::Val{:invalid}) = MapPoint(
    -1, -1, Set{Int64}(), BitVector(), Dict{Int64, BitVector},
    Point3f0(), 0f0, false, false,
)

function MapPoint(id, kfid, descriptor, is_observed::Bool = true)
    observed_keyframes_ids = Set{Int64}(kfid)
    keyframes_descriptors = Dict{Int64, BitVector}(kfid => descriptor)
    position = Point3f0(0f0)
    inv_depth = 0f0
    is_3d = false

    MapPoint(
        id, kfid, observed_keyframes_ids,
        descriptor, keyframes_descriptors,
        Point3f0(0f0), 0f0,
        is_3d, is_observed,
    )
end

@inline is_valid(m::MapPoint)::Bool = m.id != -1

@inline add_keyframe_observation!(m::MapPoint, id::Int64) =
    push!(m.observer_keyframes_ids, id)

struct MapManager
    current_frame::Frame
    frames_map::Dict{Int64, Frame}

    nb_keyframes::Int64

    params::Params
    extractor::Extractor
    # tracker::Tracker

    map_points::Dict{Int64, MapPoint}
    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_mappoints::Int64
end

function create_keyframe!(m::MapManager, image)
    prepare_frame!(m)
    extract_keypoints!(m, image)
    add_keyframe!(m)
end

function prepare_frame!(m::MapManager)
    m.current_frame.kfid = m.nb_keyframes

    # Filter if there are too many keypoints.
    # if m.current_frame.nb_keypoints > m.params.max_nb_keypoints
    #     # TODO
    # end

    for keypoint in get_keypoints(m.current_frame)
        # Get related MapPoint.
        mp = get(m.map_points, keypoint.id, MapPoint(Val{:invalid}))
        if !is_valid(mp)
            remove_obs_from_current_frame!(m, keypoint.id)
            continue
        end
        # Link new Keyframe to the MapPoint.
        add_keyframe_observation!(mp, m.current_keyframe_id)
    end
end

function extract_keypoints!(m::MapManager, image)
    keypoints = m.current_frame |> get_keypoints
    current_points = [kp.pixel for kp in keypoints]

    # describe keypoints if using brief

    nb_2_detect = m.params.max_nb_keypoints - m.current_frame.nb_occupied_cells
    nb_2_detect ≤ 0 && return
    # Detect keypoints in the provided `image` using current keypoints
    # and roi to set a mask.
    keypoints = detect(m.extractor, image, current_points)
    isempty(keypoints) && return

    descriptors, keypoints = describe(m.extractor, image, keypoints)
    add_keypoints_to_frame!(m, m.current_frame, keypoints, descriptors)
end

function add_keypoints_to_frame!(
    m::MapManager, frame::Frame, keypoints, descriptors,
)
    for (kp, dp) in zip(keypoints, descriptors)
        # m.current_mappoint_id is incremented in `add_mappoint!`.
        add_keypoint!(frame, m.current_mappoint_id)
        add_mappoint!(m, dp)
    end
end

function add_mappoint!(m::MapManager, descriptor)
    new_mappoint = MapPoint(
        m.current_mappoint_id, m.current_keyframe_id, descriptor,
    )
    m.map_points[m.current_mappoint_id] = new_mappoint
    m.current_mappoint_id += 1
    m.nb_mappoints += 1
end

"""
Copy current MapManager's Frame and add it to the KeyFrame map.
Increase current keyframe id & total number of keyframes.
"""
function add_keyframe!(m::MapManager)
    m.frames_map[m.current_keyframe_id, deepcopy(m.current_frame)]
    m.current_keyframe_id += 1
    m.nb_keyframes += 1
end

"""
Remove a MapPoint observation from current Frame by `id`.
"""
function remove_obs_from_current_frame!(m::MapManager, id::Int64)
    remove_keypoint!(m.current_frame, id)
    # TODO visualization related part. Point-cloud point removal.
    # Set MapPoint as not observable.
    # if !(id in keys(m.map_points))
    #     # TODO Reset point in point-cloud to origin-point.
    #     return
    # end
end
