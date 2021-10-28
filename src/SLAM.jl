module SLAM
export SlamManager, add_image!, add_stereo_image!, get_queue_size
export Params, Camera, run!, to_cartesian, reset!
export ReplaySaver
export set_frame_wc!, set_image!, set_position!

using BSON: @save, @load
using OrderedCollections: OrderedSet, OrderedDict
using Interpolations
using Images
using ImageDraw
using ImageFeatures
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

function to_4x4(m::SMatrix{3, 3, T, 9}) where T
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        0,       0,       0,       1)
end
function to_4x4(m::SMatrix{3, 4, T, 12}) where T
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

include("optical_flow/utils.jl")
include("optical_flow/pyramid.jl")
include("optical_flow/lucas_kanade.jl")

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
include("io/saver.jl")
include("bundle_adjustment.jl")

"""
```julia
SlamManager(
    params::Params, camera::Camera;
    right_camera::Union{Nothing, Camera} = nothing,
)
```

Slam Manager that is the highest level component in the system.
It is responsible for sending new frames to the other components
and for processing their outputs.

**Note**, that upon creating, SlamManager launches Mapper
in the separate thread.

# Arguments

- `params::Params`: Parameters of the system.
- `image_queue::Vector{Matrix{Gray{Float64}}}`: Queue of the images to be
    processed.
- `right_image_queue::Vector{Matrix{Gray{Float64}}}`: In case of stereo mode,
    queue of images for the right camera.
    It should be in sync with `image_queue` which in this case is for the left
    camera.
- `time_queue::Vector{Float64}`: Queue of timestamps for each of the frame.
    The timestamps are used in the motion model to predict next pose for the
    frame, before it is actually computed.
- `current_frame::Frame`: Current frame that is processed. It is shared among
    all other components in the system.
- `frame_id::Int64`: Id of the current frame.
- `front_end::FrontEnd`: Front-End component that is used for tracking.
- `map_manager::MapManager`: Map manager for the managment of
    keyframes & mappoints.
- `mapper::Mapper`: Mapper that is used for triangulation of keypoints.
    It is launched in the constructor as a separate thread.
- `extractor::Extractor`: Used in extraction of keypoints from the frames.
- `exit_required::Bool`: Set it to `true` to stop SlamManager
    once it is launched.
"""
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

    visualizer::Union{Nothing, ReplaySaver}

    exit_required::Bool

    mapper_thread
    image_lock::ReentrantLock
end

function SlamManager(
    params, camera; right_camera = nothing, visualizer = nothing,
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
    @debug "[SM] Launched mapper thread."

    SlamManager(
        params, image_queue, right_image_queue,
        time_queue, frame, frame.id,
        front_end, map_manager, mapper, extractor,
        visualizer,
        false, mapper_thread, ReentrantLock())
end

"""
```julia
run!(sm::SlamManager)
```

Main routine for the SlamManager. It runs until `exit_required` variable
is set to `true`. After that, it will end its work, wait for other threads
and finish.

Once there is a frame in the queue, it will get it and first send it to the
FrontEnd for tracking. If FrontEnd requires a new Keyframe, then it will also
send it to the mapper thread for Keyframe creation and triangulation.
"""
function run!(sm::SlamManager)
    image::Union{Nothing, Matrix{Gray{Float64}}} = nothing
    right_image::Union{Nothing, Matrix{Gray{Float64}}} = nothing
    time = 0.0

    while !sm.exit_required
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

        is_kf_required = track!(sm.front_end, image, time, sm.visualizer)
        if sm.params.reset_required
            reset!(sm)
            continue
        end

        is_kf_required || continue
        try
            add_new_kf!(sm.mapper, KeyFrame(
                sm.current_frame.kfid,
                sm.params.stereo ? deepcopy(sm.front_end.current_pyramid) : nothing,
                sm.params.stereo ? right_image : nothing))
        catch e
            showerror(stdout, e)
            display(stacktrace(catch_backtrace()))
        end
        sleep(1e-2)
    end

    sm.mapper.exit_required = true
    wait(sm.mapper_thread)
    @debug "[SM] Exit required."
end

"""
```julia
add_image!(sm::SlamManager, image, time)
```

Add monocular image and its timestamp to the queue.
"""
function add_image!(sm::SlamManager, image, time)
    lock(sm.image_lock) do
        push!(sm.image_queue, image)
        push!(sm.time_queue, time)
        @debug "[SM] Image queue size $(length(sm.image_queue))."
    end
end

"""
```julia
add_stereo_image!(sm::SlamManager, image, right_image, time)
```

Add stereo image and its timestamp to the queue.
"""
function add_stereo_image!(sm::SlamManager, image, right_image, time)
    lock(sm.image_lock) do
        push!(sm.image_queue, image)
        push!(sm.right_image_queue, right_image)
        push!(sm.time_queue, time)
        @debug "[SM] Stereo image queue size $(length(sm.image_queue))."
    end
end

"""
```julia
get_image!(sm::SlamManager)
```

Get monocular image and its timestamp from the queue.

# Returns:

`(image, timestamp)` if there is an image in the queue.
Otherwise `(nothing, nothing)`.
"""
function get_image!(sm::SlamManager)
    lock(sm.image_lock) do
        isempty(sm.image_queue) && return nothing, nothing
        image = popfirst!(sm.image_queue)
        time = popfirst!(sm.time_queue)
        image, time
    end
end

"""
```julia
get_stereo_image!(sm::SlamManager)
```

Get stereo image and its timestamp from the queue.

# Returns:

`(image, right_image, timestamp)` if there is an image in the queue.
Otherwise `(nothing, nothing, nothing)`.
"""
function get_stereo_image!(sm::SlamManager)
    lock(sm.image_lock) do
        (isempty(sm.image_queue) || isempty(sm.right_image_queue)) &&
            return nothing, nothing, nothing

        image = popfirst!(sm.image_queue)
        right_image = popfirst!(sm.right_image_queue)
        time = popfirst!(sm.time_queue)
        image, right_image, time
    end
end

"""
```julia
get_queue_size(sm::SlamManager)
```

Get size of the queue of images to be processed.
"""
function get_queue_size(sm::SlamManager)
    lock(sm.image_lock) do
        length(sm.image_queue)
    end
end

"""
```julia
reset!(sm::SlamManager)
```

Reset slam manager, front-end and map_manager.
"""
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

# let
#     pyr = LKPyramid(rand(Gray{Float64}, 28, 28), 2; reusable=true)
#     update!(pyr, rand(Gray{Float64}, 28, 28))
# end

end
