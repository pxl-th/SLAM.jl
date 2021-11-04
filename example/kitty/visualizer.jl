using OrderedCollections: OrderedDict
using GLMakie
using SLAM

include("kitty.jl")

"""
Visualizer for the SLAM.
Can be used to play in "live" mode as the SLAM is working
or to replay from ReplaySaver.
# Usage:
To use it in "live" preview mode, pass it as a keyword argument,
when creating SlamManager:
```julia
visualizer = Visualizer(;resolution=(900, 600), image_resolution=(370, 1226))
SlamManager(params, camera; visualizer)
```
"""
mutable struct Visualizer
    figure::Figure

    top_grid::GridLayout
    bottom_grid::GridLayout

    map_axis::Axis3
    image_axis::Makie.Axis

    positions::Observable{Vector{Point3f0}}
    position_ids::Dict{Int64, Int64}
    positions_queue::OrderedDict{Int64, Point3f0}

    image::Observable{Matrix{Gray{Float64}}}

    positions_lock::ReentrantLock
end

"""
```julia
Visualizer(;resolution, image_resolution)
```
# Arguments:
- `resolution`: Initial resolution of the figure in `(width, height)` format.
- `image_resolution`: Initial resolution of the figure in `(height, width)` format.
"""
function Visualizer(;resolution, image_resolution)
    figure = Figure(;resolution)
    top_grid = figure[1, 1:2] = GridLayout()
    bottom_grid = figure[2, 1:2] = GridLayout()

    map_axis = Axis3(
        top_grid[1, 1]; aspect=:data, azimuth=-π / 2, elevation=1)
    image_axis = Makie.Axis(
        top_grid[1, 2]; aspect=DataAspect(),
        leftspinevisible=false, rightspinevisible=false,
        bottomspinevisible=false, topspinevisible=false)

    top_grid[1, 1, Top()] = Label(figure, "Map")
    top_grid[1, 2, Top()] = Label(figure, "Current Image")

    camera_dir_element = MarkerElement(;color=:red, marker="-")
    mappoint_element = MarkerElement(;color=:gray, marker="•")

    Legend(
        bottom_grid[1, 1],
        [mappoint_element, camera_dir_element],
        ["Mappoint", "Camera position"];
        orientation=:horizontal, tellheight=true)
    trim!(figure.layout)

    positions = Observable(Point3f0[])
    position_ids = Dict{Int64, Int64}()
    positions_queue = OrderedDict{Int64, Point3f0}()

    lines!(map_axis, positions; color=:red, quality=1, linewidth=2)

    image = Observable(zeros(Gray{Float64}, image_resolution))
    image!(image_axis, image)

    Visualizer(
        figure, top_grid, bottom_grid,
        map_axis, image_axis,
        positions, position_ids, positions_queue,
        image,
        ReentrantLock())
end

Base.display(v::Visualizer) = Base.display(v.figure)

"""
```julia
set_image!(v::Visualizer, image)
```
Update image in the visualizer.
It should have the same dimensions as the original.
"""
function set_image!(v::Visualizer, image)
    v.image[] = copy!(v.image[], image)
end

"""
```julia
set_position!(v::Visualizer, position)
```
Add new position to the camera positions. This updates the plot immediately.
"""
function set_position!(v::Visualizer, position)
    v.positions[] = push!(v.positions[], position)
    autolimits!(v.map_axis)
end

"""
```julia
set_frame_wc!(v::Visualizer, frame_id, wc)
```
Add new frame `wc` to the visualizer queue.
This is used when other threads are updating the visualizer.
The queue is processed in `process_frame_wc!` method.
"""
function set_frame_wc!(v::Visualizer, frame_id, wc)
    lock(v.positions_lock) do
        base_position = SVector{4, Float64}(0, 0, 0, 1)
        position = (wc * base_position)[[1, 3, 2]]
        v.positions_queue[frame_id] = position
    end
end

"""
```julia
process_frame_wc!(v::Visualizer)
```
Process pose queue. This updates the plot immediately.
"""
function process_frame_wc!(v::Visualizer)
    lock(v.positions_lock) do
        is_done = isempty(v.positions_queue)
        do_update = !is_done

        while !is_done
            frame_id, position = popfirst!(v.positions_queue)
            is_done = isempty(v.positions_queue)

            pid = get(v.position_ids, frame_id, -1)
            if pid == -1
                v.positions[] = push!(v.positions[], position)
                v.position_ids[frame_id] = length(v.positions[])
            else
                v.positions[][pid] = position
            end
        end

        if do_update
            autolimits!(v.map_axis)
            sleep(1 / 60)
        end
    end
end

function replay_kitty(kitty_dir, save_dir, sequence, n_frames)
    dataset = KittyDataset(kitty_dir, sequence; stereo=false)
    println(dataset)

    save_dir = joinpath(save_dir, "kitty-$sequence")
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
