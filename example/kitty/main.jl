using GLMakie
using SLAM
using GeometryBasics
using BSON: @save, @load

include("kitty.jl")

function main(n_frames::Int)
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"
    dataset = KittyDataset(base_dir, sequence)
    println(dataset)

    save_dir = joinpath("/home/pxl-th/projects", "kitty-$sequence")
    frames_dir = joinpath(save_dir, "frames")
    isdir(save_dir) || mkdir(save_dir)
    isdir(frames_dir) || mkdir(frames_dir)
    println("Save directory: $save_dir")

    mappoints_save_file = joinpath(save_dir, "mappoints.bson")
    positions_save_file = joinpath(save_dir, "positions.bson")

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    height, width = 376, 1241
    camera = SLAM.Camera(
        fx, fy, cx, cy,
        0, 0, 0, 0,
        height, width,
    )
    params = Params()
    # slam_manager = SlamManager(params, camera)
    # base_position = SVector{4, Float64}(0, 0, 0, 1)

    # slam_positions = Point3f0[]
    # slam_mappoints = Vector{Point3f0}[]
    # slam_mp_ids = Set{Int64}()

    # for i in 1:n_frames
    #     timestamp = dataset.timestamps[i]
    #     frame = dataset[i] .|> Gray{Float64}
    #     run!(slam_manager, frame, timestamp)

    #     vframe = SLAM.draw_keypoints!(
    #         RGB{Float64}.(frame), slam_manager.current_frame,
    #     )
    #     save(joinpath(frames_dir, "frame-$i.png"), vframe)

    #     position = Point3f0((slam_manager.current_frame.wc * base_position)[1:3])
    #     push!(slam_positions, position[[1, 3, 2]])

    #     frame_pc = Point3f0[]
    #     for (mid, mp) in slam_manager.map_manager.map_points
    #         mp.is_3d || continue
    #         mid in slam_mp_ids && continue

    #         push!(slam_mp_ids, mid)
    #         push!(frame_pc, Point3f0(mp.position[[1, 3, 2]]...))
    #     end
    #     push!(slam_mappoints, frame_pc)
    # end

    # @save mappoints_save_file slam_mappoints
    # @save positions_save_file slam_positions

    @load mappoints_save_file slam_mappoints
    @load positions_save_file slam_positions

    visualizer = Visualizer((height, width))
    vpc = Observable(Point3f0[])
    vpositions = Observable(Point3f0[])
    image = Observable(zeros(RGB{Float64}, width, height))

    lines!(
        visualizer.pc_axis, vpositions;
        color=:red, quality=1, linewidth=2,
    )
    meshscatter!(
        visualizer.pc_axis, vpc;
        color=:black, markersize=0.02, quality=8,
    )

    image!(visualizer.image_axis, image)
    xlims!(visualizer.image_axis, (0, width))
    ylims!(visualizer.image_axis, (0, height))

    visualizer |> display
    sleep(1 / 60)

    min_bound = Point3f0(maxintfloat())
    max_bound = Point3f0(-maxintfloat())

    for i in 1:length(slam_positions)
        camera_pos = slam_positions[i]
        min_bound = min.(min_bound, camera_pos)
        max_bound = max.(max_bound, camera_pos)

        vpositions[] = push!(vpositions[], camera_pos)
        vpc[] = append!(vpc[], slam_mappoints[i])

        frame = rotr90(load(joinpath(frames_dir, "frame-$i.png")))
        image[] = copy!(image[], frame)

        radius = 12.5
        xlims!(visualizer.pc_axis,
            (-radius + camera_pos[1], radius + camera_pos[1]))
        ylims!(visualizer.pc_axis,
            (-radius + camera_pos[2], radius + camera_pos[2]))
        zlims!(visualizer.pc_axis, (-1 + camera_pos[3], 1 + camera_pos[3]))

        sleep(1 / 60)
    end

    xlims!(visualizer.pc_axis, (min_bound[1], max_bound[1]))
    ylims!(visualizer.pc_axis, (min_bound[2], max_bound[2]))
    zlims!(visualizer.pc_axis, (min_bound[3], max_bound[3]))

    visualizer.figure
end
