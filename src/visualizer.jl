# NOTE: z is up
mutable struct Visualizer
    figure::Figure

    top_grid::GridLayout
    bottom_grid::GridLayout

    pc_axis::Axis3
    image_axis::Makie.Axis

    # point_cloud::Observable{Vector{Point3f0}}
    # camera_positions::Observable{Vector{Point3f0}}
    # camera_directions::Observable{Vector{Point3f0}}
end

function Visualizer(image_resolution)
    # set_theme!(theme_light())

    figure = Figure(resolution=(1600, 768))
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
    mappoint_element_image = MarkerElement(;color=:blue, marker="⬤")
    mappoint_element = MarkerElement(;color=:gray, marker="⬤")

    Legend(
        bottom_grid[1, 1],
        [mappoint_element, camera_dir_element],
        ["Mappoint", "Camera direction"];
        orientation=:horizontal, tellheight=true,
    )
    Legend(
        bottom_grid[1, 2],
        [keypoint_element, mappoint_element_image],
        ["2D Keypoint", "Triangulated Keypoint (Mappoint)"];
        orientation=:horizontal, tellheight=true,
    )

    xlims!(image_axis, (0, image_resolution[2]))
    ylims!(image_axis, (0, image_resolution[1]))

    # point_cloud = Node(Point3f0[])
    # camera_positions = Node(Point3f0[])
    # camera_directions = Node(Point3f0[])

    # points_obj = meshscatter!(
    #     pc_axis, point_cloud; markersize=0.05, color=:blue,
    #     marker=Rect3D(Vec3f0(0, 0, 0), Vec3f0(1, 1, 1)),
    # )
    # arrows!(pc_axis, camera_positions, camera_directions; color=:red, quality=4)
    # lines!(pc_axis, camera_positions; color=:red, quality=1)

    trim!(figure.layout)

    colsize!(top_grid, 1, Relative(1 / 2))
    colsize!(top_grid, 2, Relative(1 / 2))
    colsize!(bottom_grid, 1, Relative(1 / 2))
    colsize!(bottom_grid, 2, Relative(1 / 2))

    Visualizer(
        figure, top_grid, bottom_grid, pc_axis, image_axis,
        # point_cloud, camera_positions, camera_directions,
    )
end

Base.display(v::Visualizer) = Base.display(v.figure)

# expand_point_cloud!(v::Visualizer, new_points::Vector{Point3f0}) =
#     v.point_cloud[] = append!(v.point_cloud[], new_points)

# update_image!(v::Visualizer, new_image) = image!(v.image_axis, new_image)

# function add_camera_position!(v::Visualizer, position, direction)
#     v.camera_positions[] = push!(v.camera_positions[], position)
#     v.camera_directions[] = push!(v.camera_directions[], direction)
# end

# function update_axii!(v::Visualizer)
#     isempty(v.point_cloud[]) && isempty(v.camera_positions[]) && return
#     v.pc_axis |> Makie.autolimits!
# end

# function update_frame!(v::Visualizer, f::Frame, image)
#     height = size(image, 2)
#     image!(v.image_axis, image)

#     keypoints, keypoints3d  = Point2f0[], Point2f0[]
#     for kp in values(f.keypoints)
#         push!(
#             kp.is_3d ? keypoints3d : keypoints,
#             Point2f0(kp.pixel[2], height - kp.pixel[1] + 1),
#         )
#     end

#     scatter!(v.image_axis, keypoints; color=:green, markersize=2)
#     scatter!(v.image_axis, keypoints3d; color=:blue, markersize=2)
# end
