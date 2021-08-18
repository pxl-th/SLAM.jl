using GLMakie
using Images

# NOTE: z is up

function main()
    points = [Point3f0(rand() * 6 + 1, rand() * 6 + 1, rand() * 3 + 0.5) for _ in 1:100]
    image = load("/home/pxl-th/Pictures/1.jpg")

    figure = Figure(resolution=(768, 768))
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    l = 6


    points_axis = Axis3(
        top_grid[1, 1]; aspect=:data,
        limits=(0, l, 0, l, 0, l),
        azimuth=π/8, elevation=0.5,
    )
    image_axis = Makie.Axis(
        top_grid[1, 2]; aspect=DataAspect(),
        leftspinevisible=false, rightspinevisible=false,
        bottomspinevisible=false, topspinevisible=false,
    )

    points_obj = meshscatter!(
        points_axis, points; markersize=0.05, color=:blue,
    )
    arrows!(points_axis, [Point3f0(0.5, 0.5, 0)], [Point3f0(0.5, 0.5, 0)]; color=:red)
    image_obj = image!(image_axis, rotr90(image))

    camera_dir_element = MarkerElement(;color=:red, marker="→")
    keypoint_element = MarkerElement(;color=:green, marker="⬤")
    mappoint_element = MarkerElement(;color=:blue, marker="⬤")

    top_grid[1, 1, Top()] = Label(figure, "3D Map")
    top_grid[1, 2, Top()] = Label(figure, "Tracked Keypoints")

    Legend(
        bottom_grid[1, 1],
        [mappoint_element, camera_dir_element],
        ["Mappoint", "Camera direction"];
        orientation=:horizontal, tellheight=true,
    )
    Legend(
        bottom_grid[1, 2],
        [keypoint_element, mappoint_element],
        ["2D Keypoint", "3D Mappoint"];
        orientation=:horizontal, tellheight=true,
    )


    colsize!(top_grid, 2, Relative(1/3))
    colsize!(bottom_grid, 1, Relative(1/2))
    colsize!(bottom_grid, 2, Relative(1/2))

    trim!(figure.layout)
    hidedecorations!(image_axis)

    figure
end

with_theme(main, theme_black())
