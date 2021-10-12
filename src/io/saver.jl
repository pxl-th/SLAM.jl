"""
```julia
ReplaySaver()
```

ReplaySaver is used to save/load SLAM result for the visualizer.
If added to the SlamManager instead of the visualizer, it accumulates
poses needed for the visualization/replay and can be later used
to replay the results.

# Usage:

To use it, pass it to `visualizer` keyword argument, when creating `SlamManager`.
Do some work, and in the end, save it to the directory.

```julia
saver = ReplaySaver()
SlamManager(params, camera; visualizer=saver)
# ...run slam...
SLAM.save(saver, "save_dir")
```

Then you can load it from the save directory and use it to replay results.

```julia
SLAM.load!(saver, "save_dir")
```
"""
mutable struct ReplaySaver
    ids::Dict{Int64, Int64} # frame id -> position id
    positions::Vector{Point3f0}

    positions_lock::ReentrantLock
end

ReplaySaver() = ReplaySaver(Dict{Int64, Int64}(), Point3f0[], ReentrantLock())

"""
```julia
set_frame_wc!(saver::ReplaySaver, frame_id, wc)
```

Add new pose to store.
"""
function set_frame_wc!(saver::ReplaySaver, frame_id, wc)
    lock(saver.positions_lock) do
        base_position = SVector{4, Float64}(0, 0, 0, 1)
        position = (wc * base_position)[[1, 3, 2]]

        pid = get(saver.ids, frame_id, -1)
        if pid == -1
            push!(saver.positions, position)
            saver.ids[frame_id] = length(saver.positions)
        else
            saver.positions[pid] = position
        end
    end
end

"""
```julia
save(saver::ReplaySaver, save_dir)
```

Save results into the given directory.
"""
function save(saver::ReplaySaver, save_dir)
    isdir(save_dir) || mkdir(save_dir)
    positions_file = joinpath(save_dir, "positions.bson")
    ids_file = joinpath(save_dir, "ids.bson")

    positions = saver.positions
    @save positions_file positions

    ids = saver.ids
    @save ids_file ids
end

"""
```julia
load!(saver::ReplaySaver, save_dir)
```

Load results from a given directory.
"""
function load!(saver::ReplaySaver, save_dir)
    isdir(save_dir) || error("Directory `$save_dir` does not exist.")

    positions_file = joinpath(save_dir, "positions.bson")
    isfile(positions_file) || error("Positions file `$positions_file` not found.")

    ids_file = joinpath(save_dir, "ids.bson")
    isfile(ids_file) || error("Ids file `$ids_file` not found.")

    @load positions_file positions
    @load ids_file ids

    saver.ids = ids
    saver.positions = positions
end
