mutable struct MapPoint
    id::Int64
    """
    Id of the KeyFrame, from which it was created.
    """
    kfid::Int64
    """
    Set of KeyFrame's ids that are visible from this MapPoint.
    """
    observer_keyframes_ids::OrderedSet{Int64}
    """
    Position in world coordinate system.
    """
    position::Point3f
    """
    True if the MapPoint has been initialized.
    """
    is_3d::Bool
    """
    True if the MapPoint is visible in the current Frame.
    """
    is_observed::Bool

    mappoint_lock::ReentrantLock
end

function MapPoint(::Val{:invalid})
    MapPoint(
        -1, -1, Set{Int64}(), BitVector(),
        Dict{Int64, BitVector}(), Dict{Int64, Float64}(),
        Point3f(0, 0, 0), 0, false, false, ReentrantLock(),
    )
end

function MapPoint(id, kfid, is_observed::Bool = true)
    observed_keyframes_ids = Set{Int64}(kfid)
    position = Point3f(0, 0, 0)
    is_3d = false

    MapPoint(
        id, kfid, observed_keyframes_ids,
        position, is_3d, is_observed, ReentrantLock(),
    )
end

@inline is_valid(m::MapPoint)::Bool = m.id != -1

function add_keyframe_observation!(m::MapPoint, kfid)
    lock(m.mappoint_lock) do
        push!(m.observer_keyframes_ids, kfid)
    end
end

function get_observers(m::MapPoint)
    lock(m.mappoint_lock) do
        deepcopy(m.observer_keyframes_ids)
    end
end

function get_observers_number(m::MapPoint)
    lock(m.mappoint_lock) do
        length(m.observer_keyframes_ids)
    end
end

function get_position(m::MapPoint)
    lock(m.mappoint_lock) do
        m.position
    end
end

function set_position!(m::MapPoint, position)
    lock(m.mappoint_lock) do
        m.position = position
        m.is_3d = true
    end
end

function remove_kf_observation!(m::MapPoint, kfid)
    lock(m.mappoint_lock) do
        kfid in keys(m.observer_keyframes_ids) || return
        pop!(m.observer_keyframes_ids, kfid)
        isempty(m.observer_keyframes_ids) && return
        # Set new anchor KeyFrame if removed previous anchor.
        kfid == m.kfid && (m.kfid = m.observer_keyframes_ids[1];)
    end
end

"""
Check if MapPoint is bad and update it if so.

Bad MapPoint is a 3D point that is observed by < 2 KeyFrames
and not observed by current Frame (e.g. m.observed = false).
If bad, set `is_3d` to `false`.
"""
function is_bad!(m::MapPoint)::Bool
    lock(m.mappoint_lock) do
        length(m.observer_keyframes_ids) < 2 && !m.is_observed && m.is_3d &&
            (m.is_3d = false; return true)
        isempty(m.observer_keyframes_ids) && !m.is_observed &&
            (m.is_3d = false; return true)
        false
    end
end
