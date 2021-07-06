struct Keypoint
    id::Int64
    """
    Coordinates of a keypoint in (X, Y) format.
    """
    pixel::Point2f
    undistorted_pixel::Point2f

    descriptor::BitVector
    # is_retracked::Bool
    is_3d::Bool
end

Keypoint(::Val{:invalid}) = Keypoint(-1, Point2f(), BitVector(), false)

function Keypoint(id, kp::ImageFeatures.Keypoint, descriptor::BitVector)
    kp = Point2f(kp[1], kp[2])
    Keypoint(id, kp, kp, descriptor, false)
end

@inline is_valid(k::Keypoint)::Bool = k.id != -1

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

    nb_occupied_cells::Int64
    cell_size::Int64

    nb_keypoints::Int64
    nb_2d_kpts::Int64
    nb_3d_kpts::Int64
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
        nb_occupied_cells, cell_size,
        nb_keypoints, 0, 0,
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
    if keypoint.is_3d
        f.nb_3d_kpts += 1
    else
        f.nb_2d_kpts += 1
    end
end

function add_keypoint_to_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells += 1)
    push!(f.keypoints_grid[kpi], keypoint.id)
end

function remove_keypoint!(f::Frame, id::Int64)
    # TODO is invalid keypoint constructed in any case?
    kp = get(f.keypoints, id, Keypoint(Val{:invalid}))
    !is_valid(kp) && return

    remove_keypoint_from_grid!(f, kp)

    f.nb_keypoints -= 1
    if kp.is_3d
        f.nb_3d_kpts -= 1
    else
        f.nb_2d_kpts -= 1
    end
end

function remove_keypoint_from_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    if keypoint.id in f.keypoints_grid[kpi]
        pop!(f.keypoints_grid[kpi], keypoint.id)
        isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells -= 1)
    end
end

function set_transformation!(f::Frame, wc::SMatrix{4, 4, Float32})
    f.wc = wc
    f.cw = inv(SE3, wc)
end
