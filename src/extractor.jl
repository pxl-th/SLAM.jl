struct Extractor
    max_points::Int64
end

@inline convert(x::Vector{HomogeneousPoint{Float64, 3}}) =
    map(p -> Point2f(p.coords[1], p.coords[2]), x)

@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

function _shi_tomasi(image, n_keypoints::Int64)
    corners = fill(false, size(image)...)
    responses = shi_tomasi(image)
    maxima = findlocalmaxima(responses)

    max_responses = [responses[mx] for mx in maxima]
    P = sortperm(max_responses; lt=(x, y) -> x > y)[1:n_keypoints]

    nb_detected = 0
    for m in maxima[P]
        corners[m] = true
        nb_detected += 1
    end

    corners, responses, nb_detected
end

function detect(e::Extractor, image, current_points)
    length(current_points) ≥ e.max_points && return Point2f[]

    δ = e.max_points - length(current_points)
    corners, responses, nb_detected = _shi_tomasi(image, δ)
    nb_detected == 0 && return Point2f[]

    corner2subpixel(responses, corners) |> convert
end
