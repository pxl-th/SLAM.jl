module SLAM

using LinearAlgebra
using StaticArrays
using Images
using ImageFeatures
using ImageTracking
using VideoIO
using Rotations
using Manifolds

const Point2 = SVector{2}
const Point2i = SVector{2, Int64}
const Point2f = SVector{2, Float64}

const Point3f = SVector{3, Float64}
const Point3f0 = SVector{3, Float32}

const SE3 = SpecialEuclidean(3)

@inline convert(x::Point2f)::Point2i = x .|> round .|> Int64
# @inline convert(x::Point2)::Point2i = x .|> round .|> Int64
@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

"""
Params:
    x::Point2 Point to convert to CartesianIndex in (row, col) format.
"""
@inline function to_cartesian(x)
    # x = x |> convert
    # CartesianIndex(x[1], x[2])
    CartesianIndex(convert(x)...)
end

@inline function to_cartesian(x::Point2, cell_size::Int64)
    x = convert(x) .÷ cell_size .+ 1
    CartesianIndex(x[1], x[2])
end

function expand(m::StaticMatrix{3, 3, T})::SMatrix{4, 4, T} where T
    m = vcat(m, SMatrix{1, 3, T}(0, 0, 0))
    hcat(m, SVector{4, T}(0, 0, 0, 1))
end

include("camera.jl")
include("extractor.jl")
include("tracker.jl")

# - Visual front end processes each image (and creates keyframes via mapmanager)
# - Puts Keyframes into Mapper

function extract_fast(image, n_keypoints::Int64, threshold::Float64 = 0.4)
    fastcorners(image, n_keypoints, threshold) |> Keypoints
end

include("params.jl")
include("frame.jl")

struct KeyFrame
    id::Int64
    image::Matrix{Gray}
    # optical flow pyramid levels
    # raw image
end

include("motion_model.jl")
include("map_manager.jl")
include("front_end.jl")

mutable struct SlamManager
    image_queue::Vector{Matrix{Gray}}
    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    map_manager::MapManager
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

    SlamManager(
        [], frame, frame.id,
        front_end, map_manager, extractor,
        camera, false,
    )
end

function run!(sm::SlamManager, image, time)
    sm.exit_required && return

    sm.frame_id += 1
    sm.current_frame.id = sm.frame_id
    sm.current_frame.time = time
    @debug "[Slam Manager] Frame $(sm.frame_id) @ $time"

    # Send image to front end.
    is_kf_required = track(sm.front_end, image, time)
    # check for reset
    # create new kf if needed (which is a copy from frontend)
    # send new kf to mapper to add to its queue
end

function main()
    params = Params(
        1000, 50, 0.5,
        3, 11, false, false,
    )
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

        i == 2 && break
    end
    reader |> close
end

function test_tracking_simple()
    ex = Extractor(1000, BRIEF())

    frame1 = load("frame-1.jpg") .|> Gray{Float64}
    frame2 = load("frame-2.jpg") .|> Gray{Float64}
    @show size(frame1)

    raw_keypoints1 = [
        SVector{2, Float64}(k[1], k[2])
        for k in detect(ex, frame1, [])
    ]
    # new_keypoints, status = fb_tracking(
    #     frame1, frame2, raw_keypoints1;
    #     nb_iterations=30, window_size=21, pyramid_levels=3,
    # )
    # new_keypoints = new_keypoints[status]

    pyramid1 = ImageTracking.LKPyramid(frame1, 3)
    pyramid2 = ImageTracking.LKPyramid(frame2, 3; compute_gradients=false)
    algo = LucasKanade(30; window_size=21, pyramid_levels=3)

    displacement = fill(SVector{2, Float64}(0.0, 0.0), length(raw_keypoints1))
    displacement, status = ImageTracking.optflow!(
        pyramid1, pyramid2, raw_keypoints1, displacement, algo,
    )
    new_keypoints = [
        rk + d
        for (rk, d, s) in zip(raw_keypoints1, displacement, status)
        if s
    ]
    @info length(raw_keypoints1)
    @info length(new_keypoints)

    displacement = fill(SVector{2, Float64}(0.0, 0.0), length(raw_keypoints1))
    # @btime ImageTracking.optflow!(
    #     $pyramid1, $pyramid2, $raw_keypoints1,
    #     $displacement, $algo,
    # )

    frame1 = frame1 .|> RGB
    for kp in raw_keypoints1
        kp = kp |> to_cartesian
        frame1[kp] = RGB(0, 1, 0)
    end
    for nkp in new_keypoints
        nkp = nkp |> to_cartesian
        frame1[nkp] = RGB(0, 0, 1)
    end

    frame2 = frame2 .|> RGB
    for kp in raw_keypoints1
        kp = kp |> to_cartesian
        frame2[kp] = RGB(0, 1, 0)
    end
    for nkp in new_keypoints
        nkp = nkp |> to_cartesian
        frame2[nkp] = RGB(0, 0, 1)
    end

    save("jl-t1.jpg", frame1)
    save("jl-t2.jpg", frame2)
end

function test_motion()
    model = MotionModel()

    x = RotX(0.025) |> expand
    @show x
    update!(model, x, 1)

    x = RotX(0.05) |> expand
    @show x
    x = model(x, 2)
    @show x

    x = RotX(0.075) |> expand
    @show x
    update!(model, x, 2)
end

main()
# test_motion()
# test_tracking_simple()

end
