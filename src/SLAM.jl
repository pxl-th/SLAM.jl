module SLAM
export SlamManager, add_image!, add_stereo_image!, get_queue_size
export Params, Camera, run!, to_cartesian, Visualizer, reset!

using OrderedCollections: OrderedSet, OrderedDict
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
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        0,       0,       0,       1)
end
function to_4x4(m::SMatrix{3, 4, T})::SMatrix{4, 4, T} where T
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        m[1, 4], m[2, 4], m[3, 4], 1)
end
function to_4x4(m, t)
    SMatrix{4, 4, Float64}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        t[1],    t[2],    t[3],    1)
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
include("estimator.jl")
include("mapper.jl")
include("visualizer.jl")
include("bundle_adjustment.jl")

mutable struct SlamManager
    params::Params

    image_queue::Vector{Matrix{Gray{Float64}}}
    right_image_queue::Vector{Matrix{Gray{Float64}}}
    time_queue::Vector{Float64}

    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    map_manager::MapManager
    mapper::Mapper
    extractor::Extractor

    camera::Camera
    exit_required::Bool

    mapper_thread
    image_lock::ReentrantLock
end

function SlamManager(
    params::Params, camera::Camera;
    right_camera::Union{Nothing, Camera} = nothing,
)
    params.stereo && right_camera ≡ nothing &&
        error("[SM] Provide `right_camera` when in stereo mode.")

    avoidance_radius = max(5, params.max_distance ÷ 2)
    image_resolution = (camera.height, camera.width)
    grid_resolution = ceil.(Int64, image_resolution ./ params.max_distance)

    image_queue = Matrix{Gray{Float64}}[]
    right_image_queue = Matrix{Gray{Float64}}[]
    time_queue = Float64[]

    frame = Frame(;camera, right_camera, cell_size=params.max_distance)
    extractor = Extractor(
        params.max_nb_keypoints, avoidance_radius,
        grid_resolution, params.max_distance)

    map_manager = MapManager(params, frame, extractor)
    front_end = FrontEnd(params, frame, map_manager)

    mapper = Mapper(params, map_manager, frame)
    mapper_thread = Threads.@spawn run!(mapper)
    @info "[SM] Launched mapper thread."

    SlamManager(
        params, image_queue, right_image_queue,
        time_queue, frame, frame.id,
        front_end, map_manager, mapper, extractor,
        camera, false, mapper_thread, ReentrantLock())
end

function add_image!(sm::SlamManager, image, time)
    lock(sm.image_lock) do
        push!(sm.image_queue, image)
        push!(sm.time_queue, time)
        @debug "[SM] Image queue size $(length(sm.image_queue))."
    end
end

function add_stereo_image!(sm::SlamManager, image, right_image, time)
    lock(sm.image_lock) do
        push!(sm.image_queue, image)
        push!(sm.right_image_queue, right_image)
        push!(sm.time_queue, time)
        @debug "[SM] Stereo image queue size $(length(sm.image_queue))."
    end
end

function get_image!(sm::SlamManager)
    lock(sm.image_lock) do
        isempty(sm.image_queue) && return nothing, nothing
        image = popfirst!(sm.image_queue)
        time = popfirst!(sm.time_queue)
        image, time
    end
end

function get_stereo_image!(sm::SlamManager)
    lock(sm.image_lock) do
        (isempty(sm.image_queue) || isempty(sm.right_image_queue)) &&
            return nothing, nothing, 0.0

        image = popfirst!(sm.image_queue)
        right_image = popfirst!(sm.right_image_queue)
        time = popfirst!(sm.time_queue)
        image, right_image, time
    end
end

function get_queue_size(sm::SlamManager)
    lock(sm.image_lock) do
        length(sm.image_queue)
    end
end

function run!(sm::SlamManager)
    image::Union{Nothing, Matrix{Gray{Float64}}} = nothing
    right_image::Union{Nothing, Matrix{Gray{Float64}}} = nothing
    time = 0.0

    while !(sm.exit_required)
        if sm.params.stereo
            image, right_image, time = get_stereo_image!(sm)
        else
            image, time = get_image!(sm)
        end
        if image ≡ nothing
            sleep(1e-2)
            continue
        end

        sm.frame_id += 1
        sm.current_frame.id = sm.frame_id
        sm.current_frame.time = time
        @debug "[SM] Frame $(sm.frame_id) @ $time."

        # Send image to the front end.
        is_kf_required = track!(sm.front_end, image, time)
        if sm.params.reset_required
            reset!(sm)
            continue
        end

        # Create new KeyFrame if needed.
        # Send it to the mapper queue for traingulation.
        is_kf_required || continue

        add_new_kf!(sm.mapper, KeyFrame(
            sm.current_frame.kfid,
            sm.params.stereo ? sm.front_end.current_pyramid : nothing,
            sm.params.stereo ? right_image : nothing))
        sleep(1e-2)
    end

    sm.mapper.exit_required = true
    wait(sm.mapper_thread)
    @info "[SM] Exit required."
end

function reset!(sm::SlamManager)
    @warn "[Slam Manager] Reset required."
    sm.params |> reset!

    sm.current_frame |> reset!
    sm.front_end |> reset!
    sm.map_manager |> reset!
    @warn "[Slam Manager] Reset applied."
end

function draw_keypoints!(
    image::Matrix{T}, frame::Frame; right::Bool = false,
) where T <: RGB
    radius = 2
    for kp in values(frame.keypoints)
        right && !kp.is_stereo && continue

        pixel = (right && kp.is_stereo) ? kp.right_pixel : kp.pixel
        in_image(frame.camera, pixel) || continue

        color = kp.is_3d ? T(0, 0, 1) : T(0, 1, 0)
        kp.is_retracked && (color = T(1, 0, 0);)
        draw!(image, CirclePointRadius(to_cartesian(pixel), radius), color)
    end
    image
end

precompile(track!, (FrontEnd, Matrix{Float64}, Float64))
precompile(track_mono!, (FrontEnd, Matrix{Float64}, Float64))
precompile(compute_pose!, (FrontEnd,))
precompile(compute_pose_5pt!, (FrontEnd,))

precompile(run!, (Mapper,))
precompile(triangulate_temporal!, (Mapper, Frame,))

precompile(run!, (Estimator,))
precompile(local_bundle_adjustment!, (Estimator, Frame,))

end
