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
    Position of a keypoint in 3D space in `(x, y, z = 1)` format.
    """
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

mutable struct Frame
    """
    Id of this Frame.
    """
    id::Int64
    """
    Id of the corresponding KeyFrame, which is created by Mapper.
    KeyFrame id in the MapManager.
    """
    kfid::Int64

    time::Float64
    # world → camera transformation.
    cw::SMatrix{4, 4, Float64, 16}
    # camera → world transformation.
    wc::SMatrix{4, 4, Float64, 16}

    camera::Camera
    right_camera::Camera

    """
    Map of observed keypoints.
    Keypoint id → Keypoint.
    """
    keypoints::Dict{Int64, Keypoint}
    keypoints_grid::Matrix{Set{Int64}}

    nb_occupied_cells::Int64
    cell_size::Int64

    nb_keypoints::Int64
    nb_2d_kpts::Int64
    nb_3d_kpts::Int64
    nb_stereo_kpts::Int64

    """
    Map of covisible KeyFrames.
    KFid → Number of MapPoints that this Frame shared with `KFid` frame.
    """
    covisible_kf::OrderedDict{Int64, Int64}
    local_map_ids::Set{Int64}

    keypoints_lock::ReentrantLock
    pose_lock::ReentrantLock
    grid_lock::ReentrantLock
    covisibility_lock::ReentrantLock
end

function Frame(;
    camera::Camera, right_camera::Union{Nothing, Camera} = nothing,
    cell_size::Int64,
    id::Int64 = 0, kfid::Int64 = 0, time::Float64 = 0.0,
    cw::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
    wc::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
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

function update_keypoint!(f::Frame, prev_id, new_id, is_3d::Bool)
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

function set_wc!(f::Frame, wc)
    lock(f.pose_lock) do
        f.wc = wc
        f.cw = inv(SE3, wc)
    end
end

function set_cw!(f::Frame, cw)
    lock(f.pose_lock) do
        f.cw = cw
        f.wc = inv(SE3, cw)
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

function get_surrounding_keypoints(f::Frame, pixel::Point2f)
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
