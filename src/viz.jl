using GLMakie
using Images

# NOTE: z is up

function main()
    set_theme!(theme_black())

    figure = Figure(resolution=(512, 512))
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    points_axis = Axis3(
        top_grid[1, 1]; aspect=:data,
        azimuth=π/8, elevation=1,
    )
    image_axis = Makie.Axis(
        top_grid[1, 2]; aspect=DataAspect(),
        leftspinevisible=false, rightspinevisible=false,
        bottomspinevisible=false, topspinevisible=false,
    )

    points = Node(Point3f0[])
    points_obj = meshscatter!(
        points_axis, points; markersize=0.05, color=:blue,
        marker=Rect3D(Vec3f0(0, 0, 0), Vec3f0(1, 1, 1)),
    )

    camera_positions = Node(Point3f0[])
    camera_directions = Node(Point3f0[])
    arrows!(points_axis, camera_positions, camera_directions; color=:red)

    image = Node(zeros(RGB{Float64}, 100, 100))
    image!(image_axis, rotr90(image[]))

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
        ["2D Keypoint", "Triangulated Keypoint (Mappoint)"];
        orientation=:horizontal, tellheight=true,
    )

    colsize!(top_grid, 2, Relative(1/3))
    colsize!(bottom_grid, 1, Relative(1/2))
    colsize!(bottom_grid, 2, Relative(1/2))

    trim!(figure.layout)
    hidedecorations!(image_axis)

    display(figure)
    sleep(1)

    reader = VideoIO.openvideo("/home/pxl-th/projects/slam.mp4")

    for (i, frame) in enumerate(reader)
        image!(image_axis, rotr90(imresize(frame; ratio=0.3)))

        camera_positions[] = push!(camera_positions[],
            Point3f0(4 + rand() * 0.1, 0.5 + i, 2 + rand() * 0.01),
        )
        camera_directions[] = push!(camera_directions[],
            Point3f0(rand() * 0.2, 0.4 + rand() * 0.2, rand() * 0.2),
        )
        new_points = [Point3f0(
            rand() * 6 + 1, rand() * 2 + i, rand() * 3 + 0.5
        ) for _ in 1:10]
        points[] = append!(points[], new_points)
        points_axis |> autolimits!

        sleep(1 / 5)
    end

    reader |> close

    nothing
end
main()
