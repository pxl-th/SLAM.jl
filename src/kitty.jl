using StaticArrays
using GLMakie
using LinearAlgebra
using Printf
using Images

@inline function parse_matrix(line)
    m = parse.(Float64, split(line, " "))
    SMatrix{4, 4, Float64}(m..., 0, 0, 0, 1)'
end

function read_poses(poses_file)
    poses = SMatrix{4, 4, Float64}[]
    open(poses_file, "r") do reader
        while !eof(reader)
            push!(poses, parse_matrix(readline(reader)))
        end
    end
    poses
end

# Convert XYZ to XZY.
@inbounds to_makie(positions) = [Point3f0(p[1], p[3], p[2]) for p in positions]

struct KittyDataset
    """
    Left camera (aka P0) intrinsic matrix.
    Dropped last column, which contains baselines in meters.
    """
    K::SMatrix{3, 3, Float64}
    """
    Ground truth poses. Each pose transforms from the origin.
    """
    poses::Vector{SMatrix{4, 4, Float64}}
    """
    Frames from the left camera.
    """
    frames_dir::String
end

function KittyDataset(base_dir::String, sequence::String)
    frames_dir = joinpath(base_dir, "sequences", sequence)
    K = parse_matrix(
        readline(joinpath(frames_dir, "calib.txt"))[5:end],
    )[1:3, 1:3]
    frames_dir = joinpath(frames_dir, "image_0")

    poses_file = joinpath(base_dir, "poses", sequence * ".txt")
    poses = read_poses(poses_file)

    KittyDataset(K, poses, frames_dir)
end

function get_camera_poses(dataset::KittyDataset)
    n_poses = length(dataset.poses)
    base_dir = SVector{3, Float64}(0, 0, 1)

    positions = Vector{SVector{3, Float64}}(undef, n_poses)
    directions = Vector{SVector{3, Float64}}(undef, n_poses)
    for (i, pose) in enumerate(dataset.poses)
        @inbounds positions[i] = -(pose[1:3, 1:3]' * pose[1:3, 4])
        @inbounds directions[i] = normalize(pose[1:3, 1:3]' * base_dir)
    end
    positions, directions
end

Base.length(dataset::KittyDataset) = length(dataset.poses)
function Base.getindex(dataset::KittyDataset, i)
    joinpath(dataset.frames_dir, @sprintf("%.06d.png", i - 1)) |> load
end

function main()
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"
    dataset = KittyDataset(base_dir, sequence)

    positions, directions = dataset |> get_camera_poses
    positions = to_makie(positions)
    directions = to_makie(directions)

    set_theme!(theme_black())
    figure = Figure(resolution=(512, 512))
    axis = Axis3(figure[1, 1]; aspect=:data, elevation=1)
    im_axis = Makie.Axis(figure[1, 2]; aspect=DataAspect())

    lines!(axis, positions; color=1:length(positions), colormap=:autumn1)
    image!(im_axis, rotr90(dataset[1]))

    figure
end

main()
