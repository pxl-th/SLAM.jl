using Images
using StaticArrays
using VideoIO
using SLAM

function main()
    vpath = "/home/pxl-th/Videos/MOV_0014.MOV"

    focal = 910.0
    width, height = 1920.0, 1080.0
    cx, cy = width / 2.0, height / 2.0
    δt = 1.0 / 30.0
    timestamp = 0.0

    camera = Camera(;fx=focal, fy=focal, cx, cy, height, width)
    params = Params(;stereo=false, do_local_bundle_adjustment=true, map_filtering=false)
    saver = ReplaySaver()
    slam_manager = SlamManager(params, camera; slam_io=saver)
    slam_manager_thread = Threads.@spawn run!(slam_manager)

    for frame in VideoIO.openvideo(vpath)
        frame = Gray{Float64}.(frame)
        add_image!(slam_manager, frame, timestamp)
        timestamp += δt

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
end
main()
