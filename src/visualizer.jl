# NOTE: z is up
mutable struct Visualizer
    figure::Figure

    top_grid::GridLayout
    bottom_grid::GridLayout

    pc_axis::Axis3
    image_axis::Makie.Axis
end

function Visualizer(image_resolution)
    figure = Figure(resolution=(1600, 768))
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    # Top grid configuration.
    pc_axis = Axis3(top_grid[1, 1]; aspect=:data, azimuth=π / 2, elevation=2)
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

    trim!(figure.layout)

    colsize!(top_grid, 1, Relative(1 / 2))
    colsize!(top_grid, 2, Relative(1 / 2))
    colsize!(bottom_grid, 1, Relative(1 / 2))
    colsize!(bottom_grid, 2, Relative(1 / 2))

    Visualizer(figure, top_grid, bottom_grid, pc_axis, image_axis)
end

Base.display(v::Visualizer) = Base.display(v.figure)
