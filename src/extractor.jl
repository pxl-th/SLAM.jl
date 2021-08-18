"""
Feature extractor.

Performs Shi-Tomasi feature extraction algorithm with support
for masking-out regions in which to avoid feature detection.
"""
struct Extractor
    """
    Maximum number of points to detect.
    """
    max_points::Int
    descriptor::BRIEF
    """
    Radius of the circle that is used to create a mask with regions
    in which to avoid feature detection.
    """
    radius::Int
end
Extractor(max_points, radius) = Extractor(max_points, BRIEF(), radius)

function _shi_tomasi(image, n_keypoints::Int; min_response::Real = 1e-4)
    corners = fill(false, size(image)...)
    responses = shi_tomasi(image)
    maxima = findlocalmaxima(responses)

    max_responses = [responses[mx] for mx in maxima]
    P = sortperm(max_responses; lt=(x, y) -> x > y)
    length(P) > n_keypoints && (P = P[1:n_keypoints];)

    nb_detected = 0
    for p in P
        maxima_corner = maxima[p]
        min_response < responses[maxima_corner] || continue
        corners[maxima_corner] = true
        nb_detected += 1
    end

    corners, responses, nb_detected
end

"""
Detect keypoints in the `image`.

# Arguments:
- `current_points`:
    Vector of points, which define circular regions, where to avoid
    detecting new features. This can be used to decrease the amount of
    similar keypoints. Pass empty vector to skip this step.
- `σ::Real`:
    Standard deviation for the gaussian blur.
    This is used to smooth out circles in the mask to reduce the chance
    of detecting aliased circle corners. Default value is `3`.
    Pass `0` to skip this step.

# Returns:
    Keypoints in the `(y, x)` format.
"""
function detect(
    e::Extractor, image, current_points; σ::Real = 3,
)::Vector{CartesianIndex{2}}
    length(current_points) ≥ e.max_points && return Point2f[]

    if !isempty(current_points)
        mask = get_mask(image, current_points, e.radius)
        # Smooth out image, to avoid detecting features on the circle edges.
        σ ≉ 0 && (mask = imfilter(mask, Kernel.gaussian(σ));)
        image = image .* mask
    end

    δ = e.max_points - length(current_points)
    corners, responses, nb_detected = _shi_tomasi(image, δ)
    nb_detected == 0 && return CartesianIndex{2}[]

    Keypoints(corners)
end

function describe(
    e::Extractor, image, keypoints,
)::Tuple{Vector{BitVector}, Vector{CartesianIndex{2}}}
    create_descriptor(image, keypoints, e.descriptor)
end

"""
Create a mask from a set of points.

Each point defines a circular region that is used to avoid
detecting features in that region.

Circular regions are filled with `zero(T)` values,
while the rest of the mask is has `ones(T)` values.
"""
function get_mask(image::Matrix{T}, points, radius) where T
    mask = ones(T, size(image))
    for point in points
        draw!(mask, CirclePointRadius(to_cartesian(point), radius), zero(T))
    end
    mask
end
