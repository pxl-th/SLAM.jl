struct Camera
    # Focal length.
    fx::Float64
    fy::Float64
    # Principal point.
    cx::Float64
    cy::Float64
    # Radial lens distortion.
    k1::Float64
    k2::Float64
    # Tangential lens distortion.
    p1::Float64
    p2::Float64
    # Intrinsic matrix.
    K::SMatrix{3, 3, Float64}
    iK::SMatrix{3, 3, Float64}
    # Image resolution.
    height::Int64
    width::Int64
end

function Camera(
    fx, fy, cx, cy, # Intrinsics.
    k1, k2, p1, p2, # Distortion coefficients.
    height, width,
)
    K = SMatrix{3, 3, Float64}(
        fx, 0.0, 0.0,
        0.0, fy, 0.0,
        cx, cy, 1.0,
    )
    @debug "[Camera] K $K"
    iK = K |> inv
    @debug "[Camera] iK $iK"

    Camera(
        fx, fy, cx, cy,
        k1, k2, p1, p2,
        K, iK, height, width,
    )
end

"""
Project point from 3D space onto the image plane.

# Arguments:
- `point`: Point in 3D space in `(x, y, z)` format.

# Returns:
    Projected point in `(y, x)` format.
"""
function project(c::Camera, point)
    inv_z = 1.0 / point[3]
    Point2f(
        c.fy * point[2] * inv_z + c.cy,
        c.fx * point[1] * inv_z + c.cx,
    )
end

"""
Project `point` onto image plane of the `Camera`,
accounting for the distortion parameters of the camera.

# Arguments:

- `point`: 3D point to project in `(x, y, z)` format.

# Returns:

2D floating point coordinates in `(y, x)` format.
"""
function project_undistort(c::Camera, point)
    normalized = point[1:2] ./ point[3]
    sqrd_normalized = normalized .^ 2
    # Square radius from center.
    sqrd_radius = sqrd_normalized |> sum
    # Radial distortion factor.
    rd = 1.0 + c.k1 * sqrd_radius + c.k2 * sqrd_radius ^ 2
    # Tangential distortion component.
    p = prod(normalized)
    dtx = 2 * c.p1 * p + c.p2 * (sqrd_radius + 2 * sqrd_normalized[1])
    dty = c.p1 * (sqrd_radius + 2 * sqrd_normalized[2]) + 2 * c.p2 * p
    # Lens distortion coordinates.
    distorted = rd .* normalized .+ (dtx, dty)
    # Final projection (assume skew is always `0`).
    distorted[2, 1] .* (c.fy, c.fx) .+ (c.cy, c.cx)
end

"""
Check if `point` is in the image bounds of the `Camera`.

# Arguments:

- `point::Point2`: Point to check. In (y, x) format.
"""
in_image(c::Camera, point::Point2) = all(1 .≤ point .≤ (c.height, c.width))
