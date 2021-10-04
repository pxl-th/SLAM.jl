using StaticArrays
using LinearAlgebra
using Printf
using Images
using Manifolds

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

function read_timestamps(timestamps_file)
    timestamps = Float64[]
    open(timestamps_file, "r") do reader
        while !eof(reader)
            push!(timestamps, parse(Float64, readline(reader)))
        end
    end
    timestamps
end

# Convert XYZ to XZY.
@inbounds to_makie(positions) = [Point3f0(p[1], p[3], p[2]) for p in positions]

struct KittyDataset
    """
    Left camera (aka P0) intrinsic matrix.
    Dropped last column, which contains baselines in meters.
    """
    K::SMatrix{4, 4, Float64, 16}
    """
    Transformation from 0-th camera to 1-st camera.
    """
    T::SMatrix{4, 4, Float64, 16}
    """
    Ground truth poses. Each pose transforms from the origin.
    """
    poses::Vector{SMatrix{4, 4, Float64, 16}}
    """
    Vector of timestamps for each frame.
    """
    timestamps::Vector{Float64}
    left_frames_dir::String
    right_frames_dir::String
    stereo::Bool
end

function KittyDataset(base_dir::String, sequence::String; stereo::Bool)
    frames_dir = joinpath(base_dir, "sequences", sequence)

    Ks = readlines(joinpath(frames_dir, "calib.txt"))
    K1 = parse_matrix(Ks[1][5:end])
    K1_inv = inv(K1)

    KT2 = parse_matrix(Ks[2][5:end])
    T12 = inv(SpecialEuclidean(3), K1_inv * KT2)

    timestamps = read_timestamps(joinpath(frames_dir, "times.txt"))

    left_frames_dir = joinpath(frames_dir, "image_0")
    right_frames_dir = joinpath(frames_dir, "image_1")

    poses_file = joinpath(base_dir, "poses", sequence * ".txt")
    poses = read_poses(poses_file)

    KittyDataset(
        K1, T12, poses, timestamps, left_frames_dir, right_frames_dir, stereo)
end

function get_camera_poses(dataset::KittyDataset)
    n_poses = length(dataset.poses)
    base_dir = SVector{3, Float64}(0, 0, 1)
    base_point = SVector{4, Float64}(0, 0, 0, 1)

    positions = Vector{SVector{3, Float64}}(undef, n_poses)
    directions = Vector{SVector{3, Float64}}(undef, n_poses)
    for (i, pose) in enumerate(dataset.poses)
        @inbounds positions[i] = (pose * base_point)[1:3]
        @inbounds directions[i] = normalize(pose[1:3, 1:3] * base_dir)
    end
    positions, directions
end

Base.length(dataset::KittyDataset) = length(dataset.poses)
function Base.getindex(dataset::KittyDataset, i)
    left_image = load(joinpath(
        dataset.left_frames_dir, @sprintf("%.06d.png", i - 1)))

    right_image = left_image
    if dataset.stereo
        right_image = load(joinpath(
            dataset.right_frames_dir, @sprintf("%.06d.png", i - 1)))
    end

    left_image, right_image
end

function Base.show(io::IO, d::KittyDataset)
    println(io, "Kitty Dataset:")
    println(io, "- Number of frames: $(length(d))")
    println(io, "- Intrinsics:")
    println(repr(MIME("text/plain"), d.K; context=io))
end
