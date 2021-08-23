using GLMakie
using SLAM

include("kitty.jl")

function main()
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"
    dataset = KittyDataset(base_dir, sequence)
    println(dataset)

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    height, width = 376, 1241
    camera = SLAM.Camera(
        fx, fy, cx, cy,
        0, 0, 0, 0,
        height, width,
    )
    params = Params()
    slam_manager = SlamManager(params, camera)
    base_position = SVector{4, Float64}(0, 0, 0, 1)
    slam_positions = Point3f0[]

    for i in 1:150
        timestamp = dataset.timestamps[i]
        frame = dataset[i] .|> Gray{Float64}
        run!(slam_manager, frame, timestamp)

        position = Point3f0((slam_manager.current_frame.wc * base_position)[1:3])
        push!(slam_positions, position[[1, 3, 2]])

        # @info "Target"
        # display(dataset.poses[i]); println()
    end
    # slam_mappoints = [
    #     Point3f0(mp.position[1], mp.position[3], mp.position[2])
    #     for mp in values(slam_manager.map_manager.map_points)
    #     if mp.is_3d
    # ]

    # positions, directions = dataset |> get_camera_poses
    # target_positions = to_makie(positions)[1:n]
    # directions = to_makie(directions)

    visualizer = Visualizer((height, width))
    lines!(visualizer.pc_axis, slam_positions; color=:blue, quality=1)
    # meshscatter!(visualizer.pc_axis, slam_mappoints; color=:black, markersize=0.05, quality=8)
    # lines!(visualizer.pc_axis, target_positions; color=:green, quality=1)
    # visualizer.pc_axis |> Makie.autolimits!

    visualizer |> display
    sleep(1 / 60)
    visualizer.figure

    # for i in 1:length(slam_poses)
    #     timestamp = dataset.timestamps[i]
    #     frame = dataset[i]
    #     slam_frame = frame .|> Gray{Float64}

    #     t1 = time()
    #     update_frame!(visualizer, slam_manager.current_frame, rotr90(frame))
    #     visualizer |> update_axii!
    #     sleep(1 / 60)
    #     t2 = time()
    #     @info "Update frame time $(t2 - t1) sec"
    # end
end
main()
