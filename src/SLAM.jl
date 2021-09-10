module SLAM
export SlamManager, Params, Camera, run!, to_cartesian
export Visualizer

using DataStructures: OrderedSet, OrderedDict
using GLMakie
using Images
using ImageDraw
using ImageFeatures
using ImageTracking
using LeastSquaresOptim
using LinearAlgebra
using Manifolds
using Random
using RecoverPose
using Rotations
using StaticArrays
using SparseArrays
using SparseDiffTools

const Point2 = SVector{2}
const Point2i = SVector{2, Int64}
const Point2f = SVector{2, Float64}
const Point3f = SVector{3, Float64}
const Point3f0 = SVector{3, Float32}

const SE3 = SpecialEuclidean(3)

@inline convert(x::Point2f)::Point2i = x .|> round .|> Int64
@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

@inline to_homogeneous(p::SVector{3, T}) where T = SVector{4, T}(p..., one(T))
@inline to_homogeneous(p::SVector{4}) = p

"""
Params:
    x::Point2 Point to convert to CartesianIndex in (row, col) format.
"""
@inline to_cartesian(x) = CartesianIndex(convert(x)...)
@inline function to_cartesian(x::Point2, cell_size::Int64)
    x = convert(x) .÷ cell_size .+ 1
    CartesianIndex(x...)
end

function to_4x4(m::StaticMatrix{3, 3, T})::SMatrix{4, 4, T} where T
    m = vcat(m, SMatrix{1, 3, T}(0, 0, 0))
    hcat(m, SVector{4, T}(0, 0, 0, 1))
end
function to_4x4(m::SMatrix{3, 4, T})::SMatrix{4, 4, T} where T
    vcat(m, SMatrix{1, 4, T}(0, 0, 0, 1))
end
function to_4x4(m, t)
    vcat(SMatrix{3, 4}(m..., t...), SMatrix{1, 4}(0, 0, 0, 1.0))
end

include("camera.jl")
include("extractor.jl")
include("tracker.jl")
include("params.jl")
include("frame.jl")
include("motion_model.jl")
include("map_point.jl")
include("map_manager.jl")
include("front_end.jl")
include("mapper.jl")
include("visualizer.jl")
include("bundle_adjustment.jl")

mutable struct SlamManager
    params::Params

    image_queue::Vector{Matrix{Gray{Float64}}}
    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    map_manager::MapManager
    mapper::Mapper
    extractor::Extractor

    camera::Camera
    exit_required::Bool
end

function SlamManager(params::Params, camera::Camera)
    avoidance_radius = max(5, params.max_distance ÷ 2)
    image_resolution = (camera.height, camera.width)
    grid_resolution = ceil.(Int64, image_resolution ./ params.max_distance)

    frame = Frame(;camera, cell_size=params.max_distance)
    extractor = Extractor(
        params.max_nb_keypoints, avoidance_radius,
        grid_resolution, params.max_distance,
    )

    map_manager = MapManager(params, frame, extractor)
    front_end = FrontEnd(params, frame, map_manager)
    mapper = Mapper(params, map_manager, frame)

    SlamManager(
        params,
        [], frame, frame.id,
        front_end, map_manager, mapper, extractor,
        camera, false,
    )
end

function run!(sm::SlamManager, image, time)
    sm.exit_required && return

    sm.frame_id += 1
    sm.current_frame.id = sm.frame_id
    sm.current_frame.time = time
    @debug "[SM] Frame $(sm.frame_id) @ $time"
    @debug "[SM] Fid $(sm.current_frame.id), KFid $(sm.current_frame.kfid)"

    # Send image to the front end.
    is_kf_required = track!(sm.front_end, image, time)
    sm.params.reset_required && (reset!(sm); return)
    # Create new KeyFrame if needed.
    # Send it to the mapper queue for traingulation.
    is_kf_required || return

    @debug "[SM] Adding new KF $(sm.current_frame.kfid)."
    add_new_kf!(sm.mapper, KeyFrame(sm.current_frame.kfid, image))
    sm.mapper |> run!
end

function draw_keypoints!(image::Matrix{T}, frame::Frame) where T <: RGB
    radius = 2
    n_outside = 0
    for kp in values(frame.keypoints)
        in_image(frame.camera, kp.pixel) || (n_outside += 1; continue;)
        color = kp.is_3d ? T(0, 0, 1) : T(0, 1, 0)
        draw!(image, CirclePointRadius(to_cartesian(kp.pixel), radius), color)
    end
    image
end

function reset!(sm::SlamManager)
    @warn "[Slam Manager] Reset required."
    sm.params |> reset!

    sm.current_frame |> reset!
    sm.front_end |> reset!
    sm.map_manager |> reset!
    @warn "[Slam Manager] Reset applied."
end

end
