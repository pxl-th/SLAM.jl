module SLAM

using LinearAlgebra
using StaticArrays
using Images
using ImageFeatures
using ImageTracking
using VideoIO
using Rotations
using Manifolds
using Parameters: @with_kw
using DataStructures: OrderedSet

using RecoverPose

const Point2 = SVector{2}
const Point2i = SVector{2, Int64}
const Point2f = SVector{2, Float64}

const Point3f = SVector{3, Float64}
const Point3f0 = SVector{3, Float32}

const SE3 = SpecialEuclidean(3)

@inline convert(x::Point2f)::Point2i = x .|> round .|> Int64
@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

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

function extract_fast(image, n_keypoints::Int64, threshold::Float64 = 0.4)
    fastcorners(image, n_keypoints, threshold) |> Keypoints
end

include("params.jl")
include("frame.jl")

include("motion_model.jl")
include("map_point.jl")
include("map_manager.jl")
include("front_end.jl")
include("mapper.jl")

mutable struct SlamManager
    image_queue::Vector{Matrix{Gray}}
    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    map_manager::MapManager
    mapper::Mapper
    extractor::Extractor

    camera::Camera
    exit_required::Bool
end

function SlamManager(
    params::Params, camera::Camera,
)
    frame = Frame(;camera=camera, cell_size=params.max_distance)
    extractor = Extractor(params.max_nb_keypoints, BRIEF())
    map_manager = MapManager(params, frame, extractor)
    front_end = FrontEnd(params, frame, map_manager)
    mapper = Mapper(params, map_manager, frame)

    SlamManager(
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
    @debug "[Slam Manager] Frame $(sm.frame_id) @ $time"
    @debug "[Slam Manager] Fid $(sm.current_frame.id), KFid $(sm.current_frame.kfid)"

    # Send image to the front end.
    is_kf_required = track(sm.front_end, image, time)

    # TODO check for reset `params.reset_required`

    # Create new KeyFrame if needed.
    # Send it to the mapper queue for traingulation.
    is_kf_required || return

    @debug "[Slam Manager] Adding new KeyFrame to Mapper @ $(sm.current_frame.kfid) id"
    add_new_kf!(sm.mapper, KeyFrame(sm.current_frame.kfid, image))
    @debug "[Slam Manager] Running Mapper routine."
    sm.mapper |> run!
end

function draw_keypoints!(image, frame::Frame)
    for kp in values(frame.keypoints)
        image[kp.pixel |> to_cartesian] = RGB(1, 0, 0)
    end
end

function main()
    params = Params()
    camera = Camera(
        910, 910, 582, 437,
        0, 0, 0, 0,
        874, 1164,
    )
    slam_manager = SlamManager(params, camera)

    reader = VideoIO.openvideo("./data/5.hevc")
    for (i, frame) in enumerate(reader)
        frame = frame .|> Gray{Float64}

        run!(slam_manager, frame, i)

        i == 3 && break
    end
    reader |> close
end
main()

end
