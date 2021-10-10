# SLAM.jl

Simultaneous Localization and Mapping.

## Features

- Monocular / Stereo modes.
- Bundle-Adjustment over a subset of Keyframes.
- Local Map Matching for re-tracking lost Mappoints back into Frame.

## Usage

The system runs in 3 threads:

- Slam Manager
- Mapper
- Estimator

So you need to start Julia with at least `-t4` flag (+ 1 for the main thread).


## Install

```julia
]add https://github.com/pxl-th/RecoverPose.jl.git
]add https://github.com/pxl-th/ImageTracking.jl.git
]add https://github.com/pxl-th/SLAM.jl.git
```

[RecoverPose.jl](https://github.com/pxl-th/RecoverPose.jl) contains functions
for computing poses and triangulation methods.
Fork of [ImageTracking.jl](https://github.com/pxl-th/ImageTracking.jl)
contains certain memory improvements and
has no restriction on the magnitude of the optical flow.

## Usage

Minimal abstract example of how to use.

```julia
using SLAM

camera = Camera(...)
params = Params(; stereo=false, ...)
manager = SlamManager(camera, params)
manager_thread = Threads.@spawn run!(manager)

images = Matrix{Gray{Float64}}[...]
timestamps = Float64[...]

for (time, image) in zip(timestamps, images)
    add_image!(manager, image, timestamp)
    sleep(1e-2)
end

manager.exit_required = true
wait(manager_thread)
```

## Examples

For a complete example, see [KITTY Dataset Example](https://github.com/pxl-th/SLAM.jl/tree/master/example/kitty).

## Results

Final map on the `00` sequence taken from KITTY dataset in stereo mode.

![KITTY 00 sequence](./assets/kitty-00-stereo.jpg)
