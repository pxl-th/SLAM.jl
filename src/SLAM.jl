module SLAM

using LinearAlgebra

using StaticArrays
using GeometryBasics
using Images
using ImageFeatures
using ImageTracking
using VideoIO
using Rotations
using Manifolds
using BenchmarkTools

"""
2D Point in (x, y) format.
"""
const Point2i = Point2{Int64}
const Point2f = Point2{Float64}

const SE3 = SpecialEuclidean(3)

@inline convert(x::SVector{2, Float64})::Point2i = x .|> round .|> Int64
@inline convert(x::Point2)::Point2i = x .|> round .|> Int64
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
    x = convert(x) .÷ cell_size
    CartesianIndex(x[1], x[2])
end

function expand(m::StaticMatrix{3, 3, T})::SMatrix{4, 4, T} where T
    m = vcat(m, SMatrix{1, 3, T}(0, 0, 0))
    hcat(m, SVector{4, T}(0, 0, 0, 1))
end

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

struct SlamManager
    image_queue::Vector{Matrix{Gray}}
    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    # map manager
    # mapper
    # feature extractor
    # feature tracker
    # exit required
end

function SlamManager()
    frame = Frame()
    front_end = FrontEnd(frame)
    SlamManager([], frame, frame.id, front_end)
end

function run(sm::SlamManager, image, time)
    sm.frame_id += 1
    sm.current_frame.id = sm.frame_id
    sm.current_frame.time = time
    # Send image to front end.
    is_kf_required = track(sm.front_end, image, time)
    # check for reset
    # create new kf if needed (which is a copy from frontend)
    # send new kf to mapper to add to its queue
end

"""
Immediate TODO:
+ image preprocessing
+ map manager: create kf (prepare, extract, add)
+ motion model
+ Keypoint struct
+ feature extractor
+ slam state (params)

- feature tracker
- compute pose 2d-3d

- camera calibration
"""

function main()
    reader = VideoIO.openvideo("./data/5.hevc")

    ex = Extractor(2000, BRIEF())

    for (i, frame) in enumerate(reader)
        # frame = frame .|> Gray

        # kps = detect(ex, frame, [])
        # descriptors, kps = describe(ex, frame, kps)

        # frame = frame .|> RGB
        # for (kp, dp) in zip(kps, descriptors)
        #     frame[kp] = RGB(1, 0, 0)
        #     Keypoint(1, kp, dp)
        # end

        save("frame-$i.jpg", frame)
        i == 2 && break
    end

    reader |> close
end

function test_tracking_simple()
    ex = Extractor(1000, BRIEF())

    frame1 = load("frame-1.jpg") .|> Gray
    frame2 = load("frame-2.jpg") .|> Gray

    # frame1 = load("t1.png") .|> Gray
    # frame2 = load("t3.png") .|> Gray

    # demo = joinpath(dirname(pathof(ImageTracking)), "..", "demo")
    # frame1 = load(joinpath(demo, "table1.jpg")) .|> Gray
    # frame2 = load(joinpath(demo, "table2.jpg")) .|> Gray

    @show size(frame1)

    raw_keypoints1 = [
        SVector{2, Float64}(k[1], k[2])
        for k in detect(ex, frame1, [])
    ]
    new_keypoints, status = fb_tracking(
        frame1, frame2, raw_keypoints1;
        nb_iterations=30, window_size=21, pyramid_levels=3,
    )
    new_keypoints = new_keypoints[status]

    @info "Full track:"
    @btime fb_tracking(
        $frame1, $frame2, $raw_keypoints1;
        nb_iterations=30, window_size=21, pyramid_levels=3,
    )

    # first_pyramid = ImageTracking.LKPyramid(frame1, 3)
    # second_pyramid = ImageTracking.LKPyramid(frame2, 3; compute_gradients=false)
    # displacement = fill(SVector{2, Float64}(0.0, 0.0), length(raw_keypoints1))
    # algorithm = LucasKanade(30; window_size=21, pyramid_levels=3)

    # @info "In-place:"
    # @btime ImageTracking.optflow!(
    #     $first_pyramid, $second_pyramid,
    #     $raw_keypoints1, $displacement,
    #     $algorithm,
    # )

    @info length(raw_keypoints1)
    @info length(new_keypoints)

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

# main()
# test_motion()
test_tracking_simple()

end
