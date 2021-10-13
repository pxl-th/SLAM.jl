# SLAM.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://pxl-th.github.io/SLAM.jl/dev/)

Simultaneous Localization and Mapping.

![KITTY 05 sequence demo](res/kitty-05.gif)

## Install

```bash
]add https://github.com/pxl-th/RecoverPose.jl.git
]add https://github.com/pxl-th/ImageTracking.jl.git
]add https://github.com/pxl-th/SLAM.jl.git
```

[RecoverPose.jl](https://github.com/pxl-th/RecoverPose.jl) contains functions for computing poses and triangulation methods.
Fork of [ImageTracking.jl](https://github.com/pxl-th/ImageTracking.jl) contains certain memory improvements
and has no restriction on the magnitude of the optical flow.

## Usage

See [docs](https://pxl-th.github.io/SLAM.jl/dev/) for instructions
or see [example/kitty/main.jl](https://github.com/pxl-th/SLAM.jl/tree/master/example/kitty)
for a full example on how to use.
