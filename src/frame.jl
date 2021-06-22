struct Keypoint
    id::Int64
    """
    Coordinates of a keypoint in (X, Y) format.
    """
    pixel::Point2f
    undistorted_pixel::Point2f

    descriptor::BitVector
    # is_retracked::Bool
    # is_3d::Bool
end

function Keypoint(id, kp::ImageFeatures.Keypoint, descriptor::BitVector)
    kp = Point2f(kp[1], kp[2])
    Keypoint(id, kp, kp, descriptor)
end

mutable struct Frame
    id::Int64
    kfid::Int64
    time::Float64
    # pose cam -> world
    cw::SMatrix{4, 4, Float64}
    # pose world -> cam
    wc::SMatrix{4, 4, Float64}
    # calibration model (camera)
    """
    Map of observed keypoints.
    """
    keypoints::Dict{Int64, Keypoint}
    keypoints_grid::Matrix{Set{Int64}}

    nb_keypoints::Int64
    nb_occupied_cells::Int64
    cell_size::Int64
end

function Frame(;
    image_resolution::Tuple{Int64, Int64}, # height, width format.
    cell_size::Int64,
    id::Int64 = 0, kfid::Int64 = 0, time::Float64 = 0.0,
    cw::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
    wc::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
)
    nb_keypoints = 0
    nb_occupied_cells = 0
    keypoints = Dict{Int64, Keypoint}()

    cells = image_resolution .÷ cell_size
    @debug "Frame grid resolution $cells"
    grid = [Set{Int64}() for _=1:cells[1], _=1:cells[2]]

    Frame(
        id, kfid, time, cw, wc,
        keypoints, grid,
        nb_keypoints, nb_occupied_cells, cell_size,
    )
end

function add_keypoint!(
    f::Frame, kid::Int64,
    keypoint::ImageFeatures.Keypoint, descriptor::BitVector,
)
    # TODO undistort keypoint using calibration model
    add_keypoint!(f, Keypoint(kid, keypoint, descriptor))
end

function add_keypoint!(f::Frame, keypoint::Keypoint)
    keypoint.id in keys(f.keypoints) && return

    f.keypoints[keypoint.id] = keypoint
    add_keypoint_to_grid!(f, keypoint)

    f.nb_keypoints += 1
    # TODO update either 2d nb kpt or 3d nb kpt
end

function add_keypoint_to_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells += 1)
    push!(f.keypoints_grid[kpi], keypoint.id)
end
