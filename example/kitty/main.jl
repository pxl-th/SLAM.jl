using BSON: @save, @load
using GeometryBasics
using GLMakie
using SLAM

include("kitty.jl")

function main(n_frames::Int)
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"
    dataset = KittyDataset(base_dir, sequence)
    println(dataset)

    save_dir = joinpath("/home/pxl-th/projects", "2-kitty-$sequence")
    frames_dir = joinpath(save_dir, "frames")
    isdir(save_dir) || mkdir(save_dir)
    isdir(frames_dir) || mkdir(frames_dir)
    @info "Save directory: $save_dir"

    mappoints_save_file = joinpath(save_dir, "mappoints.bson")
    positions_save_file = joinpath(save_dir, "positions.bson")

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    height, width = 376, 1241
    # height, width = 370, 1226
    camera = SLAM.Camera(fx, fy, cx, cy, 0, 0, 0, 0, height, width)
    params = Params(;
        window_size=9, max_distance=35, pyramid_levels=3,
        max_nb_keypoints=1000, max_reprojection_error=3.0)
    slam_manager = SlamManager(params, camera)
    slam_manager_thread = Threads.@spawn run!(slam_manager)

    t1 = time()

    for i in 1:n_frames
        timestamp = dataset.timestamps[i]
        frame = dataset[i] .|> Gray{Float64}
        add_image!(slam_manager, frame, timestamp)

        q_size = get_queue_size(slam_manager)
        f_size = length(slam_manager.mapper.estimator.frame_queue)
        m_size = length(slam_manager.mapper.keyframe_queue)
        while q_size > 0 || f_size > 0 || m_size > 0
            sleep(1e-2)
            q_size = get_queue_size(slam_manager)
            f_size = length(slam_manager.mapper.estimator.frame_queue)
            m_size = length(slam_manager.mapper.keyframe_queue)
        end
        sleep(1e-2)
    end

    slam_manager.exit_required = true
    wait(slam_manager_thread)

    t2 = time()
    @info "SLAM took: $(t2 - t1) seconds."

    # Visualize result.
    map_manager = slam_manager.map_manager
    kfids = sort!(collect(keys(map_manager.frames_map)))

    min_bound = Point3f0(maxintfloat())
    max_bound = Point3f0(-maxintfloat())

    base_position = SVector{4, Float64}(0, 0, 0, 1)
    slam_mappoints = Vector{Point3f0}[]
    slam_positions = Point3f0[]

    for kfid in kfids
        pose = map_manager.frames_map[kfid].wc
        position = (pose * base_position)[1:3]
        position = position[[1, 3, 2]]
        push!(slam_positions, position)

        min_bound = min.(min_bound, position)
        max_bound = max.(max_bound, position)
    end
    slam_mappoints = [
        Point3f0(m.position[[1, 3, 2]])
        for m in values(map_manager.map_points)
        if m.is_3d]

    # @save mappoints_save_file slam_mappoints
    # @save positions_save_file slam_positions

    # @load mappoints_save_file slam_mappoints
    # @load positions_save_file slam_positions

    visualizer = Visualizer((height, width))
    markersize = minimum(max_bound .- min_bound) * 1e-2

    lines!(
        visualizer.pc_axis, slam_positions;
        color=:red, quality=1, linewidth=2)
    meshscatter!(
        visualizer.pc_axis, slam_mappoints;
        color=:black, markersize, quality=8)

    @show min_bound
    @show max_bound

    xlims!(visualizer.pc_axis, min_bound[1], max_bound[1])
    ylims!(visualizer.pc_axis, min_bound[2], max_bound[2])
    zlims!(visualizer.pc_axis, min_bound[3], max_bound[3])

    slam_manager, visualizer.figure
end
