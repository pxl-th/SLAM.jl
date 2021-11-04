using SLAM

include("kitty.jl")

function kitty_slam(kitty_dir, save_dir, sequence, n_frames, stereo = true)
    dataset = KittyDataset(kitty_dir, sequence; stereo)
    println(dataset)

    save_dir = joinpath(save_dir, "kitty-$sequence")
    isdir(save_dir) || mkdir(save_dir)

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    # TODO KITTY dataset should do that
    height, width = 376, 1241
    # height, width = 370, 1226

    camera = SLAM.Camera(;fx, fy, cx, cy, height, width)
    right_camera = SLAM.Camera(;fx, fy, cx, cy, height, width, Ti0=dataset.Ti0)

    params = Params(;
        stereo, do_local_bundle_adjustment=false, map_filtering=true)

    saver = ReplaySaver()
    slam_manager = SlamManager(params, camera; right_camera, slam_io=saver)
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
    @info "Saved ReplaySaver."
    slam_manager
end
