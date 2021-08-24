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

    for i in 1:300
        timestamp = dataset.timestamps[i]
        frame = dataset[i] .|> Gray{Float64}
        run!(slam_manager, frame, timestamp)

        vframe = SLAM.draw_keypoints!(frame .|> RGB{Float64}, slam_manager.current_frame)
        cid = slam_manager.current_frame.id
        save("/home/pxl-th/projects/frame-$i-$cid.png", vframe)

        position = Point3f0((slam_manager.current_frame.wc * base_position)[1:3])
        push!(slam_positions, position[[1, 3, 2]])
    end

    slam_mappoints = [
        Point3f0(mp.position[1], mp.position[3], mp.position[2])
        for mp in values(slam_manager.map_manager.map_points)
        if mp.is_3d
    ]

    # positions, directions = dataset |> get_camera_poses
    # target_positions = to_makie(positions)[1:700]
    # directions = to_makie(directions)

    visualizer = Visualizer((height, width))
    # lines!(visualizer.pc_axis, target_positions; color=:green, quality=1)
    lines!(visualizer.pc_axis, slam_positions; color=:green, quality=1)
    meshscatter!(
        visualizer.pc_axis, slam_mappoints;
        color=:black, markersize=0.01, quality=8,
    )

    visualizer.pc_axis |> Makie.autolimits!
    xlims!(visualizer.pc_axis, (-2, 10))
    ylims!(visualizer.pc_axis, (2, 30))
    zlims!(visualizer.pc_axis, (-2, 2))

    visualizer |> display
    sleep(1 / 60)
    visualizer.figure
end
main()
