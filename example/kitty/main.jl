using GLMakie
using SLAM
using GeometryBasics
using BSON: @save, @load

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
    # slam_manager = SlamManager(params, camera)
    # base_position = SVector{4, Float64}(0, 0, 0, 1)

    # slam_positions = Point3f0[]
    # slam_mappoints = Vector{Point3f0}[]
    # slam_mp_ids = Set{Int64}()

    # for i in 1:1500
    #     timestamp = dataset.timestamps[i]
    #     frame = dataset[i] .|> Gray{Float64}
    #     run!(slam_manager, frame, timestamp)

    #     vframe = SLAM.draw_keypoints!(frame .|> RGB{Float64}, slam_manager.current_frame)
    #     cid = slam_manager.current_frame.id
    #     save("/home/pxl-th/projects/00/frames/frame-$i-$cid.png", vframe)

    #     position = Point3f0((slam_manager.current_frame.wc * base_position)[1:3])
    #     push!(slam_positions, position[[1, 3, 2]])

    #     frame_pc = Point3f0[]
    #     for (mid, mp) in slam_manager.map_manager.map_points
    #         mp.is_3d || continue
    #         mid in slam_mp_ids && continue

    #         push!(slam_mp_ids, mid)
    #         push!(frame_pc, Point3f0(mp.position[1], mp.position[3], mp.position[2]))
    #     end
    #     push!(slam_mappoints, frame_pc)
    # end

    # @save "/home/pxl-th/projects/00/pc.bson" slam_mappoints
    # @save "/home/pxl-th/projects/00/positions.bson" slam_positions

    @load "/home/pxl-th/projects/00/pc.bson" slam_mappoints
    @load "/home/pxl-th/projects/00/positions.bson" slam_positions

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

    for i in 1:length(slam_positions)
        camera_pos = slam_positions[i]
        vpositions[] = push!(vpositions[], camera_pos)
        vpc[] = append!(vpc[], slam_mappoints[i])

        frame = rotr90(load("/home/pxl-th/projects/00/frames/frame-$i-$i.png"))
        image[] = copy!(image[], frame)

        xlims!(visualizer.pc_axis, (-15 + camera_pos[1], 15 + camera_pos[1]))
        ylims!(visualizer.pc_axis, (-15 + camera_pos[2], 15 + camera_pos[2]))
        zlims!(visualizer.pc_axis, (-1 + camera_pos[3], 1 + camera_pos[3]))

        sleep(1 / 60)
    end

    # visualizer.pc_axis |> Makie.autolimits!
    xlims!(visualizer.pc_axis, (-150, 60))
    ylims!(visualizer.pc_axis, (2, 300))
    zlims!(visualizer.pc_axis, (-2, 2))

    visualizer.figure
end

# main()
