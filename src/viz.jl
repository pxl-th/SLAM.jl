using GLMakie
using Images

include("kitty.jl")


mutable struct Visualizer
    figure::Figure

    top_grid::GridLayout
    bottom_grid::GridLayout

    pc_axis::Axis3
    image_axis::Makie.Axis

    point_cloud::Observable{Vector{Point3f0}}
    camera_positions::Observable{Vector{Point3f0}}
    camera_directions::Observable{Vector{Point3f0}}
end

function Visualizer()
    set_theme!(theme_minimal())

    figure = Figure(resolution=(512, 512))
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    # Top grid configuration.

    pc_axis = Axis3(top_grid[1, 1]; aspect=:data, azimuth=π / 8, elevation=1)
    image_axis = Makie.Axis(
        top_grid[1, 2]; aspect=DataAspect(),
        leftspinevisible=false, rightspinevisible=false,
        bottomspinevisible=false, topspinevisible=false,
    )

    top_grid[1, 1, Top()] = Label(figure, "3D Map")
    top_grid[1, 2, Top()] = Label(figure, "Tracked Keypoints")

    # Bottom grid configuration.

    camera_dir_element = MarkerElement(;color=:red, marker="→")
    keypoint_element = MarkerElement(;color=:green, marker="⬤")
    mappoint_element = MarkerElement(;color=:blue, marker="⬤")

    Legend(
        bottom_grid[1, 1],
        [mappoint_element, camera_dir_element],
        ["Mappoint", "Camera direction"];
        orientation=:horizontal, tellheight=true,
    )
    Legend(
        bottom_grid[1, 2],
        [keypoint_element, mappoint_element],
        ["2D Keypoint", "Triangulated Keypoint (Mappoint)"];
        orientation=:horizontal, tellheight=true,
    )

    point_cloud = Node(Point3f0[])
    camera_positions = Node(Point3f0[])
    camera_directions = Node(Point3f0[])

    points_obj = meshscatter!(
        pc_axis, point_cloud; markersize=0.05, color=:blue,
        marker=Rect3D(Vec3f0(0, 0, 0), Vec3f0(1, 1, 1)),
    )
    # arrows!(pc_axis, camera_positions, camera_directions; color=:red, quality=4)
    lines!(pc_axis, camera_positions; color=:red, quality=1)

    trim!(figure.layout)
    hidedecorations!(image_axis)

    colsize!(top_grid, 2, Relative(1/3))
    colsize!(bottom_grid, 1, Relative(1/2))
    colsize!(bottom_grid, 2, Relative(1/2))

    Visualizer(
        figure, top_grid, bottom_grid, pc_axis, image_axis,
        point_cloud, camera_positions, camera_directions,
    )
end

Base.display(v::Visualizer) = Base.display(v.figure)

expand_point_cloud!(v::Visualizer, new_points::Vector{Point3f0}) =
    v.point_cloud[] = append!(v.point_cloud[], new_points)

update_image!(v::Visualizer, new_image) = image!(v.image_axis, new_image)

function add_camera_position!(v::Visualizer, position, direction)
    v.camera_positions[] = push!(v.camera_positions[], position)
    v.camera_directions[] = push!(v.camera_directions[], direction)
end

update_axii!(v::Visualizer) = v.pc_axis |> Makie.autolimits!

# NOTE: z is up

function main()
    visualizer = Visualizer()
    visualizer |> display
    sleep(1)

    reader = VideoIO.openvideo("/home/pxl-th/projects/slam.mp4")

    for (i, frame) in enumerate(reader)
        update_image!(visualizer, rotr90(imresize(frame; ratio=0.3)))
        add_camera_position!(
            visualizer,
            Point3f0(4 + rand() * 0.1, 0.5 + i, 2 + rand() * 0.01),
            Point3f0(rand() * 0.2, 0.4 + rand() * 0.2, rand() * 0.2),
        )
        expand_point_cloud!(visualizer, [Point3f0(
            rand() * 6 + 1, rand() * 2 + i, rand() * 3 + 0.5
        ) for _ in 1:10])

        visualizer |> update_axii!
        sleep(1 / 5)
    end

    reader |> close
end

function kitty_vis()
    visualizer = Visualizer()
    visualizer |> display
    sleep(1)

    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "02"
    dataset = KittyDataset(base_dir, sequence)

    positions, directions = dataset |> get_camera_poses
    positions = to_makie(positions)
    directions = to_makie(directions)

    for i in 1:length(dataset)
        frame = dataset[i]
        update_image!(visualizer, rotr90(imresize(frame; ratio=0.3)))
        visualizer.camera_positions[] = push!(
            visualizer.camera_positions[], positions[i],
        )

        visualizer |> update_axii!
        sleep(1 / 60)
    end
end
kitty_vis()
# main()
