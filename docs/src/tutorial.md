# Tutorial

## Launch SlamManager routine

!!! warning

    There are 3 components in the SLAM.jl that need to run on a separate
    thread. So, remember to launch Julia with at least `-t4` flag
    (+ 1 for the main thread).

To launch SLAM algorithm, first we need to set up the
[Camera](@ref) and [Parameters](@ref).

One of the main things you need to specify in parameters is mode (stereo or mono).
For other parameters, refer to the documentation, but they all have sane defaults.
To improve performance, you can disable local map matching and bundle adjustment
using `do_local_bundle_adjustment` and `do_local_matching` parameters.

```julia
using SLAM

focal = (910, 910)
resolution = 1024
principal_point = (resolution ÷ 2, resolution ÷ 2)
distortions = (0, 0, 0, 0)

camera = Camera(
    focal..., principal_point..., distortions...,
    resolution, resolution,
)
params = Params(;stereo=false)
```

After setting up parameters and camera we can create [Slam Manager](@ref)
and launch its main [`run!(::SlamManager)`](@ref) routine as a separate thread.
Under the hood, [Slam Manager](@ref) also launches two `run!` routines
for the [Estimator](@ref) and [Mapper](@ref).

```julia
manager = SlamManager(camera, params)
manager_thread = Threads.@spawn run!(manager)
```

Now you can feed `manager` with images and its timestamps to track.

```julia
add_image!(manager, image, timestamp)
```

## Exit SLAM

To tell `manager` that you want to finish its job, set `manager.exit_required`
to `true` and wait for the thread to finish `wait(manager_thread)`.

## Stereo mode

In stereo mode you also need to set right camera.
For it, you need to specify transformation matrix that transforms
points from the 0-th camera to this (referred as i-th camera) `Ti0`.

```julia
Ti0 = SMatrix{4, 4, Float64}(...)
right_camera = SLAM.Camera(
    focal..., principal_point..., distortions...,
    resolution, resolution; Ti0)
params = Params(;stereo=true)
```

Launch manager, providing right camera:

```julia
manager = SlamManager(camera, params; right_camera)
manager_thread = Threads.@spawn run!(manager)
```

In stereo mode, feed images:

```julia
add_stereo_image!(manager, left_image, right_image, timestamp)
```

## Synchronizing threads

Because different components take different time to run,
you need to periodically synchronize them, so that they are roughly processing
the same frames and not lagging behind.

Also because they start their work each in different time and due
to JIT compilation, they take different time to compile.

Dirty way to do this right now is to check length of each of the
queue and wait untill they become empty:

```julia
q_size = get_queue_size(slam_manager)
f_size = length(slam_manager.mapper.estimator.frame_queue)
m_size = length(slam_manager.mapper.keyframe_queue)
while q_size > 0 || f_size > 0 || m_size > 0
    sleep(1e-2)
    q_size = get_queue_size(slam_manager)
    f_size = length(slam_manager.mapper.estimator.frame_queue)
    m_size = length(slam_manager.mapper.keyframe_queue)
end
```

!!! note

    In future this ideally would not be needed.

## Visualization and saving results

If you want to save all results, you can save [Slam Manager](@ref) directly
as it contains all the processed information.

If you want to visualize the results as the algorithm runs, use
[`SLAM.Visualizer`](@ref) and pass it as keyword to [Slam Manager](@ref).

```julia
visualizer = Visualizer((900, 600))
display(visualizer)
manager = SlamManager(params, camera; right_camera, visualizer)
```

Otherwise, you can use [`ReplaySaver`](@ref) instead of `visualizer`
which will save all the necessary results for later visualization and replay.

You can see [example/kitty/main.jl](https://github.com/pxl-th/SLAM.jl/blob/master/example/kitty/main.jl#L84)
for how to use it for visualization.
