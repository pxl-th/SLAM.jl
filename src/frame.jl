"""
Keypoint is a feature in the image used for tracking.

# Parameters:

- `id::Int64`: Id of the Keypoint.
- `pixel::Point2f`: Pixel coordinate in the image plane in `(y, x)` format.
- `undistorted_pixel::Point2f`: In presence of distortion in camera,
    this is the undistorted `pixel` coordinates.
- `position::Point3f`: Pre-divided (backprojected) keypoint in camera space.
    This is used in algorithms like 5Pt for Essential matrix calculation.
- `descriptor::BitVector`: Descriptor of a keypoint.
- `is_3d::Bool`: If `true`, then this keypoint was triangulated.
- `is_retracked::Bool`: If `true`, then this keypoint was lost
    and re-tracked back by `match_local_map!` method.
- `is_stereo::Bool`: If `true`, then this keypoint has left↔right
    correspondence for the stereo image.
- `right_pixel::Point2f`: Pixel coordinate in the right image plane
    in `(y, x)` format.
- `right_undistorted_pixel::Point2f`: In presence of distortion in camera,
    this is the undistorted `right_pixel` coordinates.
- `right_position::Point3f`: Pre-divided (backprojected) right keypoint
    in camera space. This is used in algorithms like 5Pt
    for Essential matrix calculation.
"""
mutable struct Keypoint
    id::Int64
    pixel::Point2f
    undistorted_pixel::Point2f

    position::Point3f
    descriptor::BitVector

    is_3d::Bool
    is_retracked::Bool
    is_stereo::Bool

    right_pixel::Point2f
    right_undistorted_pixel::Point2f
    right_position::Point3f
end

function Keypoint(id, pixel, undistorted_pixel, position, descriptor, is_3d)
    Keypoint(
        id, pixel, undistorted_pixel, position, descriptor,
        is_3d, false, false,
        pixel, undistorted_pixel, position)
end

"""
Frame that encapsulates information of a camera in space at a certain moment.

# Parameters:

- `id::Int64`: Id of the Frame.
- `kfid::Int64`: Id of the Frame in the global map, contained in MapManager.
    Frames that are in this map are called "Key-Frames".
- `time::Float64`: Time of the frame at which it was taken on the camera.
- `cw::SMatrix{4, 4, Float64, 16}`: Transformation matrix `[R|t]`
    that transforms from world to camera space.
- `wc::SMatrix{4, 4, Float64, 16}`: Transformation matrix `[R|t]`
    that transforms from camera to world space.
- `camera::Camera`: Camera associated with this frame.
- `right_camera::Camera`: In case of stereo, this is the camera
    associated with the right frame.
- `keypoints::Dict{Int64, Keypoint}`: Map of that this frames observes.
- `ketpoints_grid::Matrix{Set{Int64}}`: Grid, where each cell contains
    several keypoints. This is useful when want to retrieve neighbours
    for a certain Keypoint.
- `nb_occupied_cells::Int64`: Number of cells in `keypoints_grid` that have
    at least one Keypoint.
- `cell_size::Int64`: Cell size in pixels.
- `nb_keypoints::Int64`: Total number of keypoints in the Frame.
- `nb_2d_keypoints::Int64`: Total number of 2D keypoints in the Frame.
- `nb_3d_keypoints::Int64`: Total number of 3D keypoints in the Frame.
- `nb_3d_keypoints::Int64`: Total number of stereo keypoints in the Frame.
- `covisible_kf::OrderedDict{Int64, Int64}`: Dictionary with `kfid` => `score`
    of ids of Frames that observe the sub-set of size `score` of keypoints
    in Frame.
- `local_map_ids::Set{Int64}`: Set of ids of MapPoints that are not visible
    in this Frame, but are a potential candidates for remapping
    back into this Frame.
"""
mutable struct Frame
    id::Int64
    kfid::Int64

    time::Float64
    cw::SMatrix{4, 4, Float64, 16}
    wc::SMatrix{4, 4, Float64, 16}

    camera::Camera
    right_camera::Camera

    keypoints::Dict{Int64, Keypoint}
    keypoints_grid::Matrix{Set{Int64}}

    nb_occupied_cells::Int64
    cell_size::Int64

    nb_keypoints::Int64
    nb_2d_kpts::Int64
    nb_3d_kpts::Int64
    nb_stereo_kpts::Int64

    covisible_kf::OrderedDict{Int64, Int64}
    local_map_ids::Set{Int64}

    keypoints_lock::ReentrantLock
    pose_lock::ReentrantLock
    grid_lock::ReentrantLock
    covisibility_lock::ReentrantLock
end

function Frame(;
    camera, right_camera = nothing, cell_size,
    id = 0, kfid = 0, time = 0.0,
    cw = SMatrix{4, 4, Float64}(I),
    wc = SMatrix{4, 4, Float64}(I),
)
    if right_camera ≡ nothing
        @warn "[F] No right camera."
        right_camera = camera
    end

    nb_keypoints = 0
    nb_occupied_cells = 0
    keypoints = Dict{Int64, Keypoint}()

    image_resolution = (camera.height, camera.width)
    cells = ceil.(Int64, image_resolution ./ cell_size)
    grid = [Set{Int64}() for _=1:cells[1], _=1:cells[2]]

    keypoints_lock = ReentrantLock()
    pose_lock = ReentrantLock()
    grid_lock = ReentrantLock()
    covisibility_lock = ReentrantLock()

    Frame(
        id, kfid, time, cw, wc,
        camera, right_camera,
        keypoints, grid,
        nb_occupied_cells, cell_size,
        nb_keypoints, 0, 0, 0,
        OrderedDict{Int64, Int64}(), Set{Int64}(),
        keypoints_lock, pose_lock, grid_lock, covisibility_lock,
    )
end

function get_keypoints(f::Frame)
    lock(f.keypoints_lock) do
        [deepcopy(kp) for kp in values(f.keypoints)]
    end
end

function get_2d_keypoints(f::Frame)
    lock(f.keypoints_lock) do
        kps = Vector{Keypoint}(undef, f.nb_2d_kpts)
        i = 1
        for k in values(f.keypoints)
            k.is_3d || (kps[i] = deepcopy(k); i += 1;)
        end
        @assert (i - 1) == f.nb_2d_kpts
        kps
    end
end

function get_3d_keypoints(f::Frame)
    lock(f.keypoints_lock) do
        kps = Vector{Keypoint}(undef, f.nb_3d_kpts)
        i = 1
        for k in values(f.keypoints)
            k.is_3d && (kps[i] = deepcopy(k); i += 1;)
        end
        @assert (i - 1) == f.nb_3d_kpts
        kps
    end
end

function get_stereo_keypoints(f::Frame)
    lock(f.keypoints_lock) do
        kps = Vector{Keypoint}(undef, f.nb_stereo_kpts)
        i = 1
        for k in values(f.keypoints)
            k.is_stereo && (kps[i] = deepcopy(k); i += 1;)
        end
        @assert (i - 1) == f.nb_stereo_kpts
        kps
    end
end

function get_3d_keypoints_nb(f::Frame)
    lock(f.keypoints_lock) do
        return f.nb_3d_kpts
    end
end

function get_3d_keypoints_ids(f::Frame)
    lock(f.keypoints_lock) do
        ids = Vector{Int64}(undef, f.nb_3d_kpts)
        i = 1
        for k in values(f.keypoints)
            k.is_3d && (ids[i] = k.id; i += 1;)
        end
        @assert (i - 1) == f.nb_3d_kpts
        ids
    end
end

function get_keypoint(f::Frame, kpid)
    lock(f.keypoints_lock) do
        deepcopy(get(f.keypoints, kpid, nothing))
    end
end

function get_keypoint_unpx(f::Frame, kpid)
    lock(f.keypoints_lock) do
        kpid in keys(f.keypoints) ?
            f.keypoints[kpid].undistorted_pixel : nothing
    end
end

function add_keypoint!(
    f::Frame, point, id; descriptor::BitVector = BitVector(),
    is_3d::Bool = false,
)
    undistorted_point = undistort_point(f.camera, point)
    position = backproject(f.camera, undistorted_point)
    add_keypoint!(f, Keypoint(
        id, point, undistorted_point, position, descriptor, is_3d))
end

function add_keypoint!(f::Frame, keypoint::Keypoint)
    lock(f.keypoints_lock) do
        if keypoint.id in keys(f.keypoints)
            @warn "[Frame] $(f.id) already has keypoint $(keypoint.id)."
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
end

function update_keypoint!(f::Frame, kpid, point)
    lock(f.keypoints_lock) do
        ckp = get(f.keypoints, kpid, nothing)
        ckp ≡ nothing && return

        kp = ckp |> deepcopy
        kp.pixel = point
        kp.undistorted_pixel = undistort_point(f.camera, kp.pixel)
        kp.position = backproject(f.camera, kp.undistorted_pixel)

        if kp.is_stereo
            kp.is_stereo = false
            f.nb_stereo_kpts -= 1
        end

        update_keypoint_in_grid!(f, ckp, kp)
        f.keypoints[kpid] = kp
    end
end

function update_stereo_keypoint!(f::Frame, kpid, right_pixel)
    lock(f.keypoints_lock) do
        kp = get(f.keypoints, kpid, nothing)
        kp ≡ nothing && return

        kp.right_pixel = right_pixel
        kp.right_undistorted_pixel =
            undistort_point(f.right_camera, right_pixel)
        kp.right_position =
            backproject(f.right_camera, kp.right_undistorted_pixel)

        if !kp.is_stereo
            kp.is_stereo = true
            f.nb_stereo_kpts += 1
        end
    end
end

function update_keypoint!(f::Frame, prev_id, new_id, is_3d)
    has_new = false
    lock(f.keypoints_lock)
    has_new = new_id in keys(f.keypoints)
    unlock(f.keypoints_lock)
    has_new && return false

    prev_kp = get_keypoint(f, prev_id)
    prev_kp ≡ nothing && return false

    prev_kp.id = new_id
    prev_kp.is_retracked = true
    prev_kp.is_3d = is_3d

    remove_keypoint!(f, prev_id)
    add_keypoint!(f, prev_kp)
    true
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
    lock(f.grid_lock) do
        isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells += 1;)
        push!(f.keypoints_grid[kpi], keypoint.id)
    end
end

function remove_keypoint_from_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    lock(f.grid_lock) do
        if keypoint.id in f.keypoints_grid[kpi]
            pop!(f.keypoints_grid[kpi], keypoint.id)
            isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells -= 1)
        end
    end
end

function remove_keypoint!(f::Frame, kpid)
    lock(f.keypoints_lock) do
        kp = get(f.keypoints, kpid, nothing)
        kp ≡ nothing && return

        pop!(f.keypoints, kpid)
        remove_keypoint_from_grid!(f, kp)

        f.nb_keypoints -= 1
        kp.is_stereo && (f.nb_stereo_kpts -= 1;)
        if kp.is_3d
            f.nb_3d_kpts -= 1
        else
            f.nb_2d_kpts -= 1
        end
    end
end

function remove_stereo_keypoint!(f::Frame, kpid)
    lock(f.keypoints_lock) do
        kp = get(f.keypoints, kpid, nothing)
        kp ≡ nothing && return
        kp.is_stereo || return

        kp.is_stereo = false
        f.nb_stereo_kpts -= 1
    end
end

function set_wc!(f::Frame, wc, visualizer = nothing)
    lock(f.pose_lock) do
        f.wc = wc
        f.cw = inv(SE3, wc)
        visualizer ≢ nothing && set_frame_wc!(visualizer, f.id, f.wc)
    end
end

function set_cw!(f::Frame, cw, visualizer = nothing)
    lock(f.pose_lock) do
        f.cw = cw
        f.wc = inv(SE3, cw)
        visualizer ≢ nothing && set_frame_wc!(visualizer, f.id, f.wc)
    end
end

function get_cw(f::Frame)
    lock(f.pose_lock) do
        f.cw
    end
end

function get_wc(f::Frame)
    lock(f.pose_lock) do
        f.wc
    end
end

function get_Rwc(f::Frame)
    lock(f.pose_lock) do
        f.wc[1:3, 1:3]
    end
end

function get_Rcw(f::Frame)
    lock(f.pose_lock) do
        f.cw[1:3, 1:3]
    end
end

function get_twc(f::Frame)
    lock(f.pose_lock) do
        f.wc[1:3, 4]
    end
end

function get_tcw(f::Frame)
    lock(f.pose_lock) do
        f.cw[1:3, 4]
    end
end

function get_cw_ba(f::Frame)::NTuple{6, Float64}
    lock(f.pose_lock) do
        r = RotZYX(f.cw[1:3, 1:3])
        (r.theta1, r.theta2, r.theta3, f.cw[1:3, 4]...)
    end
end

function get_wc_ba(f::Frame)::NTuple{6, Float64}
    lock(f.pose_lock) do
        r = RotXYZ(f.wc[1:3, 1:3])
        (r.theta1, r.theta2, r.theta3, f.wc[1:3, 4]...)
    end
end

function set_cw_ba!(f::Frame, θ)
    lock(f.pose_lock) do
        set_cw!(f, to_4x4(RotZYX(θ[1:3]...), θ[4:6]))
    end
end

function project_camera_to_world(f::Frame, point)
    lock(f.pose_lock) do
        f.wc * to_homogeneous(point)
    end
end

function project_world_to_camera(f::Frame, point)
    lock(f.pose_lock) do
        f.cw * to_homogeneous(point)
    end
end

function project_world_to_right_camera(f::Frame, point)
    lock(f.pose_lock) do
        f.right_camera.Ti0 * f.cw * to_homogeneous(point)
    end
end

function project_world_to_image(f::Frame, point)
    project(f.camera, project_world_to_camera(f, point))
end

function project_world_to_right_image(f::Frame, point)
    project(f.camera, f.right_camera.Ti0 * project_world_to_camera(f, point))
end

function project_world_to_image_distort(f::Frame, point)
    project_undistort(f.camera, project_world_to_camera(f, point))
end

function project_world_to_right_image_distort(f::Frame, point)
    project_undistort(f.camera, project_world_to_right_camera(f, point))
end

function turn_keypoint_3d!(f::Frame, id)
    lock(f.keypoints_lock) do
        kp = get(f.keypoints, id, nothing)
        kp ≡ nothing && return
        kp.is_3d && return

        kp.is_3d = true
        f.nb_2d_kpts -= 1
        f.nb_3d_kpts += 1
    end
end

function get_covisible_map(f::Frame)
    lock(f.covisibility_lock) do
        deepcopy(f.covisible_kf)
    end
end

function set_covisible_map!(f::Frame, covisible_map)
    lock(f.covisibility_lock) do
        f.covisible_kf = covisible_map
    end
end

function add_covisibility!(f::Frame, kfid, cov_score)
    kfid == f.kfid && return
    lock(f.covisibility_lock) do
        f.covisible_kf[kfid] = cov_score
    end
end

function add_covisibility!(f::Frame, kfid)
    kfid == f.kfid && return
    lock(f.covisibility_lock) do
        score = get(f.covisible_kf, kfid, 0)
        f.covisible_kf[kfid] = score + 1
    end
end

function decrease_covisible_kf!(f::Frame, kfid)
    lock(f.covisibility_lock) do
        kfid == f.kfid && return
        cov_score = get(f.covisible_kf, kfid, nothing)
        (cov_score ≡ nothing || cov_score == 0) && return

        cov_score -= 1
        f.covisible_kf[kfid] = cov_score
        cov_score == 0 && pop!(f.covisible_kf, kfid)
    end
end

function remove_covisible_kf!(f::Frame, kfid)
    kfid == f.kfid && return
    lock(f.covisibility_lock) do
        kfid in keys(f.covisible_kf) && pop!(f.covisible_kf, kfid)
    end
end

function is_observing_kp(f::Frame, kpid)
    lock(f.keypoints_lock) do
        kpid in keys(f.keypoints)
    end
end

function get_surrounding_keypoints(f::Frame, kp::Keypoint)
    keypoints = Vector{Keypoint}(undef, 0)
    sizehint!(keypoints, 20)
    kpi = to_cartesian(kp.pixel, f.cell_size)

    lock(f.keypoints_lock)
    lock(f.grid_lock)
    try
        for r in (kpi[1] - 1):(kpi[1] + 1), c in (kpi[2] - 1):(kpi[2] + 1)
            (r < 1 || c < 1 || r > size(f.keypoints_grid, 1)
                || c > size(f.keypoints_grid, 2)) && continue

            for cell_kpid in f.keypoints_grid[r, c]
                cell_kpid == kp.id && continue
                cell_kp = get(f, cell_kpid, nothing)
                cell_kp ≡ nothing || push!(keypoints, cell_kp)
            end
        end
    finally
        unlock(f.grid_lock)
        unlock(f.keypoints_lock)
    end

    keypoints
end

function get_surrounding_keypoints(f::Frame, pixel)
    keypoints = Vector{Keypoint}(undef, 0)
    sizehint!(keypoints, 20)
    kpi = to_cartesian(pixel, f.cell_size)

    lock(f.keypoints_lock)
    lock(f.grid_lock)
    try
        for r in (kpi[1] - 1):(kpi[1] + 1), c in (kpi[2] - 1):(kpi[2] + 1)
            (r < 1 || c < 1 || r > size(f.keypoints_grid, 1)
                || c > size(f.keypoints_grid, 2)) && continue

            for cell_kpid in f.keypoints_grid[r, c]
                cell_kp = get(f.keypoints, cell_kpid, nothing)
                cell_kp ≡ nothing || push!(keypoints, cell_kp)
            end
        end
    finally
        unlock(f.grid_lock)
        unlock(f.keypoints_lock)
    end

    keypoints
end

in_image(f::Frame, point) = in_image(f.camera, point)
in_right_image(f::Frame, point) = in_image(f.right_camera, point)

function reset!(f::Frame)
    lock(f.covisibility_lock)
    lock(f.keypoints_lock)
    lock(f.pose_lock)
    lock(f.grid_lock)

    f.nb_2d_kpts = 0
    f.nb_3d_kpts = 0
    f.nb_stereo_kpts = 0
    f.nb_keypoints = 0
    f.nb_occupied_cells = 0
    f.time = 0

    f.keypoints |> empty!
    f.keypoints_grid .|> empty!
    f.covisible_kf |> empty!

    f.wc = SMatrix{4, 4, Float64}(I)
    f.cw = SMatrix{4, 4, Float64}(I)

    unlock(f.grid_lock)
    unlock(f.pose_lock)
    unlock(f.keypoints_lock)
    unlock(f.covisibility_lock)
end
