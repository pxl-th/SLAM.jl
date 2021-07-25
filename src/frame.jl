mutable struct Keypoint
    """
    Unique Keypoint id.
    """
    id::Int64
    """
    Coordinates of a keypoint in `(y, x)` format.
    """
    pixel::Point2f
    undistorted_pixel::Point2f
    """
    Position of a keypoint in 3D space in `(x, y, z)` format. Normalized.
    """
    position::Point3f

    descriptor::BitVector
    # is_retracked::Bool
    is_3d::Bool
end

function Keypoint(::Val{:invalid})
    Keypoint(
        -1, Point2f(0, 0), Point2f(0, 0), Point3f(0, 0, 0),
        BitVector(), false,
    )
end

function Keypoint(id, kp::ImageFeatures.Keypoint, descriptor::BitVector)
    kp = Point2f(kp[1], kp[2])
    Keypoint(id, kp, kp, descriptor, false)
end

@inline is_valid(k::Keypoint)::Bool = k.id != -1

mutable struct Frame
    id::Int64
    kfid::Int64
    time::Float64
    # world -> camera transformation.
    cw::SMatrix{4, 4, Float64}
    # camera -> world transformation.
    wc::SMatrix{4, 4, Float64}
    # Calibration camera model.
    camera::Camera
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
    camera::Camera, cell_size::Int64,
    id::Int64 = 0, kfid::Int64 = 0, time::Float64 = 0.0,
    cw::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
    wc::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
)
    nb_keypoints = 0
    nb_occupied_cells = 0
    keypoints = Dict{Int64, Keypoint}()

    image_resolution = (camera.height, camera.width)
    cells = image_resolution .÷ cell_size
    @debug "[Frame] Grid resolution: $cells"
    grid = [Set{Int64}() for _=1:cells[1], _=1:cells[2]]

    Frame(
        id, kfid, time, cw, wc,
        camera,
        keypoints, grid,
        nb_occupied_cells, cell_size,
        nb_keypoints, 0, 0,
    )
end

get_keypoints(f::Frame) = f.keypoints |> values

function add_keypoint!(
    f::Frame, point::Point2f, id::Int64;
    descriptor::BitVector = BitVector(), is_3d::Bool = false,
)
    # TODO undistort `point` using calibration model
    undistorted_point = point
    # Compute normalized 3d position of a point in (x, y, z) format.
    # Note that `point` is in `(y, x)` format.
    position = normalize(f.camera.iK * Point3f(point[2], point[1], 1.0))
    kp = Keypoint(id, point, undistorted_point, position, descriptor, is_3d)
    add_keypoint!(f, kp)
end

function add_keypoint!(f::Frame, keypoint::Keypoint)
    if keypoint.id in keys(f.keypoints)
        @warn "[Frame] $(f.id) already has keypoint $(keypoint.id). Skipping."
        return
    end

    f.keypoints[keypoint.id] = keypoint
    add_keypoint_to_grid!(f, keypoint)

    f.nb_keypoints += 1
    if keypoint.is_3d
        f.nb_3d_kpts += 1
    else
        f.nb_2d_kpts += 1
    end
end

function update_keypoint!(f::Frame, id::Int64, point)
    ckp = get(f.keypoints, id, Keypoint(Val(:invalid)))
    is_valid(ckp) || return

    kp = ckp |> deepcopy
    kp.pixel = point
    # TODO undistort
    kp.undistorted_pixel = point
    kp.position = normalize(f.camera.iK * Point3f(point[2], point[1], 1.0))

    update_keypoint_in_grid!(f, ckp, kp)
    f.keypoints[id] = kp
end

function update_keypoint_in_grid!(
    f::Frame, previous_keypoint::Keypoint, new_keypoint::Keypoint,
)
    prev_kpi = to_cartesian(previous_keypoint.pixel, f.cell_size)
    new_kpi = to_cartesian(new_keypoint.pixel, f.cell_size)
    prev_kpi == new_kpi && return
    # Update grid, when new keypoint changes its position
    # so much as to move to the other grid cell.
    remove_keypoint_from_grid!(f, previous_keypoint)
    add_keypoint_to_grid!(f, new_keypoint)
end

function add_keypoint_to_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells += 1)
    push!(f.keypoints_grid[kpi], keypoint.id)
end

function remove_keypoint!(f::Frame, id::Int64)
    # TODO is invalid keypoint constructed in any case?
    kp = get(f.keypoints, id, Keypoint(Val(:invalid)))
    is_valid(kp) || return

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

function set_wc!(f::Frame, wc::SMatrix{4, 4, Float64})
    f.wc = wc
    f.cw = inv(SE3, wc)
end

function set_cw!(f::Frame, cw::SMatrix{4, 4, Float64})
    f.cw = cw
    f.wc = inv(SE3, cw)
end

function project_camera_to_world(f::Frame, point)
    f.wc * point
end

function project_world_to_camera(f::Frame, point)
    f.cw * point
end

function project_world_to_image(f::Frame, point)
    project(f.camera, project_world_to_camera(f, point))
end

function project_world_to_image_distort(f::Frame, point)
    project_undistort(f.camera, project_world_to_camera(f, point))
end
