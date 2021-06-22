mutable struct Extractor
    max_points::Int64
end

@inline convert(x::Vector{HomogeneousPoint{Float64, 3}}) =
    map(p -> Point2f(p.coords[1], p.coords[2]), x)

@inline convert(x::Vector{Point2f}) =
    Point2i[xi .|> round .|> Int64 for xi in x]

function _detect_shi_tomasi(image, n_keypoints::Int64; args...)
    corners = fill(false, size(image)...)
    responses = shi_tomasi(image; args...)
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

function detect(
    e::Extractor, image, current_points, nb_max::Int64 = -1
)::Vector{Point2f}
    length(current_points) ≥ e.max_points && return Point2f[]

    δ = e.max_points - length(current_points)
    nb_max != -1 && (δ = nb_max;)

    corners, responses, nb_detected = _detect_shi_tomasi(image, δ)
    nb_detected == 0 && return Point2f[]

    corner2subpixel(responses, corners) |> convert
end

# TODO use shi tomasi instead of fast?
function detect_grid_fast(
    e::Extractor, image, cell_size::Int64,
    current_points::Vector{Point2i}, roi,
)
    height, width = image |> size

    vertical_cells = height ÷ cell_size
    horizontal_cells = width ÷ cell_size
    nb_cells = vertical_cells * horizontal_cells

    occupied_cells = fill(false, vertical_cells, horizontal_cells) # + 1 to size?
    for px in current_points
        occupied_cells[px[2] ÷ cell_size, px[1] ÷ cell_size] = true
    end

    detected_pixels = Matrix{Point2i}(undef, vertical_cells, horizontal_cells)
    nb_empty, nb_occupied = 0, 0

    for i in CartesianIndices(detected_pixels)
        if occupied_cells[i]
            nb_occupied += 1
            continue
        end

        nb_empty += 1
        # YX format.
        ti = i |> Tuple
        cell_roi_start = max.(1, (ti .- 1) .* cell_size)
        cell_roi_end = cell_roi_start .+ cell_size .+ 1

        if cell_roi_end[1] > height || cell_roi_end[2] > width
            continue
        end

        cell_view = @view(image[
            cell_roi_start[1]:cell_roi_end[1],
            cell_roi_start[2]:cell_roi_end[2],
        ])
        cell_keypoints = fastcorners(cell_view, 12, e.threshold)
        isempty(cell_keypoints) && continue
    end
end
