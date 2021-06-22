module SLAM

using LinearAlgebra

using StaticArrays
using GeometryBasics
using Images
using ImageFeatures
using VideoIO
using Rotations
using Manifolds

"""
2D Point in (x, y) format.
"""
const Point2i = Point2{Int64}
const Point2f = Point2{Float64}

@inline convert(x::Point2)::Point2i = x .|> round .|> Int64
@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

@inline function to_cartesian(x::Point2)
    x = x |> convert
    CartesianIndex(x[2], x[1])
end

@inline function to_cartesian(x::Point2, cell_size::Int64)
    x = convert(x) .÷ cell_size
    CartesianIndex(x[2], x[1])
end

include("extractor.jl")

function expand(m::StaticMatrix{3, 3, T})::SMatrix{4, 4, T} where T
    m = vcat(m, SMatrix{1, 3, T}(0, 0, 0))
    hcat(m, SVector{4, T}(0, 0, 0, 1))
end

# - Visual front end processes each image (and creates keyframes via mapmanager)
# - Puts Keyframes into Mapper

function extract_fast(image, n_keypoints::Int64, threshold::Float64 = 0.4)
    fastcorners(image, n_keypoints, threshold) |> Keypoints
end

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
- map manager: create kf (prepare, extract, add)
+ motion model
- feature tracker
- compute pose 2d-3d
- slam state
+ Keypoint struct
+ extractor
- camera calibration
"""

function main()
    reader = VideoIO.openvideo("./data/5.hevc")

    ex = Extractor(2000, BRIEF())

    for (i, frame) in enumerate(reader)
        @show i
        frame = frame .|> Gray
        @show size(frame), typeof(frame)

        kps = detect(ex, frame, [])
        @show length(kps)
        descriptors, kps = describe(ex, frame, kps)
        @show length(kps)
        @show typeof(descriptors)

        frame = frame .|> RGB
        for (kp, dp) in zip(kps, descriptors)
            frame[kp] = RGB(1, 0, 0)
            Keypoint(1, kp, dp)
        end

        save("frame-$i.jpg", frame)
        i == 20 && break
    end

    reader |> close
end
main()

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
# test_motion()

end
