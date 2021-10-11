mutable struct ReplaySaver
    ids::Dict{Int64, Int64} # frame id -> position id
    positions::Vector{Point3f0}

    positions_lock::ReentrantLock
end

function ReplaySaver()
    ReplaySaver(Dict{Int64, Int64}(), Point3f0[], ReentrantLock())
end

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

function save(saver::ReplaySaver, save_dir)
    isdir(save_dir) || mkdir(save_dir)
    positions_file = joinpath(save_dir, "positions.bson")
    ids_file = joinpath(save_dir, "ids.bson")

    positions = saver.positions
    @save positions_file positions

    ids = saver.ids
    @save ids_file ids
end

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
