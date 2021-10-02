using LinearAlgebra
using StaticArrays
using Images
using ImageDraw
using ImageFeatures
using ImageTracking
using Rotations
using Manifolds
using DataStructures
using GLMakie

using LeastSquaresOptim
using SparseArrays
using SparseDiffTools

image = rand(RGB{Float64}, 128, 128)
save("./tmp.png", image)
load("./tmp.png")

algorithm = LucasKanade(10; window_size=9, pyramid_levels=3)
ImageTracking.LKPyramid(image, 3)

RotXYZ(0, 0, 0)

const SE3 = SpecialEuclidean(3)
x = SMatrix{4, 4, Float64}(I)
exp_lie(SE3, log_lie(SE3, x))

function makie_main()
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
    arrows!(pc_axis, camera_positions, camera_directions; color=:red, quality=4)
    lines!(pc_axis, camera_positions; color=:red, quality=1)

    trim!(figure.layout)
    hidedecorations!(image_axis)

    colsize!(top_grid, 2, Relative(1 / 3))
    colsize!(bottom_grid, 1, Relative(1 / 2))
    colsize!(bottom_grid, 2, Relative(1 / 2))

    display(figure)
end
makie_main()
