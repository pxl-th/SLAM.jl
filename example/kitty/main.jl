using BSON: @save, @load
using GeometryBasics
using GLMakie
using SLAM

include("kitty.jl")

function main(n_frames)
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"
    stereo = true

    dataset = KittyDataset(base_dir, sequence; stereo)
    println(dataset)

    save_dir = joinpath("/home/pxl-th/projects", "kitty-$sequence")
    isdir(save_dir) || mkdir(save_dir)
    @info "Save directory: $save_dir"

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    height, width = 376, 1241
    # height, width = 370, 1226

    camera = SLAM.Camera(fx, fy, cx, cy, 0, 0, 0, 0, height, width)
    right_camera = SLAM.Camera(
        fx, fy, cx, cy, 0, 0, 0, 0, height, width; Ti0=dataset.Ti0)

    params = Params(;
        stereo,
        window_size=9, max_distance=35, pyramid_levels=3,
        max_nb_keypoints=1000, max_reprojection_error=3.0,
        do_local_bundle_adjustment=false, map_filtering=true)

    saver = ReplaySaver()
    visualizer = nothing
    # visualizer = Visualizer((900, 600))
    # display(visualizer)

    slam_manager = SlamManager(params, camera; right_camera, visualizer=saver)
    slam_manager_thread = Threads.@spawn run!(slam_manager)

    t1 = time()
    for i in 1:n_frames
        timestamp = dataset.timestamps[i]
        left_frame, right_frame = dataset[i]
        left_frame = Gray{Float64}.(left_frame)

        if params.stereo
            right_frame = Gray{Float64}.(right_frame)
            add_stereo_image!(slam_manager, left_frame, right_frame, timestamp)
        else
            add_image!(slam_manager, left_frame, timestamp)
        end

        if visualizer ≢ nothing
            SLAM.set_image!(visualizer, rotr90(left_frame))
            process_frame_wc!(visualizer)
        end

        q_size = get_queue_size(slam_manager)
        f_size = length(slam_manager.mapper.estimator.frame_queue)
        m_size = length(slam_manager.mapper.keyframe_queue)
        while q_size > 0 || f_size > 0 || m_size > 0
            sleep(1e-2)
            q_size = get_queue_size(slam_manager)
            f_size = length(slam_manager.mapper.estimator.frame_queue)
            m_size = length(slam_manager.mapper.keyframe_queue)
        end

        sleep(1e-2)
    end

    slam_manager.exit_required = true
    wait(slam_manager_thread)

    t2 = time()
    @info "SLAM took: $(t2 - t1) seconds."

    SLAM.save(saver, save_dir)
    slam_manager, visualizer
end

function replay(n_frames)
    base_dir = "/home/pxl-th/Downloads/kitty-dataset/"
    sequence = "00"

    dataset = KittyDataset(base_dir, sequence; stereo=false)
    println(dataset)

    save_dir = joinpath("/home/pxl-th/projects", "kitty-$sequence")
    isdir(save_dir) || mkdir(save_dir)
    @info "Save directory: $save_dir"

    saver = ReplaySaver()
    SLAM.load!(saver, save_dir)
    @assert length(saver.positions) == n_frames - 1

    resolution = (900, 600)
    image_resolution = (1241, 376)
    # image_resolution = (1226, 370)
    visualizer = Visualizer(;resolution, image_resolution)
    display(visualizer)

    t1 = time()
    for i in 2:n_frames
        left_frame, right_frame = dataset[i]
        left_frame = Gray{Float64}.(left_frame)

        position = saver.positions[i - 1]
        set_image!(visualizer, rotr90(left_frame))
        set_position!(visualizer, position)

        sleep(1 / 60)
    end
    t2 = time()
    @info "Visualization took: $(t2 - t1) seconds."

    visualizer
end
