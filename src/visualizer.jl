# NOTE: z is up
mutable struct Visualizer
    figure::Figure

    top_grid::GridLayout
    bottom_grid::GridLayout

    pc_axis::Axis3
end

function Visualizer(image_resolution)
    figure = Figure(resolution=(1600, 768))
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    pc_axis = Axis3(top_grid[1, 1:2]; aspect=:data, azimuth=-π / 2, elevation=1)
    top_grid[1, 1:2, Top()] = Label(figure, "Map")

    camera_dir_element = MarkerElement(;color=:red, marker="→")
    keypoint_element = MarkerElement(;color=:green, marker="⬤")
    mappoint_element_image = MarkerElement(;color=:blue, marker="⬤")
    mappoint_element = MarkerElement(;color=:gray, marker="⬤")

    Legend(
        bottom_grid[1, 1:2],
        [mappoint_element, camera_dir_element],
        ["Mappoint", "Camera direction"];
        orientation=:horizontal, tellheight=true,
    )
    trim!(figure.layout)

    Visualizer(figure, top_grid, bottom_grid, pc_axis)
end

Base.display(v::Visualizer) = Base.display(v.figure)
