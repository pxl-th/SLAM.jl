using ImageTracking
using Images, ImageView
using StaticArrays

# demo = joinpath(dirname(pathof(ImageTracking)), "..", "demo")
# img1 = load(joinpath(demo, "table1.jpg"))
# img2 = load(joinpath(demo, "table2.jpg"))
# bw1 = img1 .|> Gray
# bw2 = img2 .|> Gray

img1 = load("basketball1.png")
img2 = load("basketball2.png")
bw1 = img1 .|> Gray
bw2 = img2 .|> Gray

corners = imcorner(bw1, method=shi_tomasi)
I = findall(!iszero, corners)
r, c = (getindex.(I, 1), getindex.(I, 2))
points = map((ri, ci) -> SVector{2}(Float64(ri), Float64(ci)), r, c)

algorithm = LucasKanade(;window_size=31)
flow, indicator = optical_flow(bw1, bw2, points, algorithm)

valid_points = points[indicator]
valid_flow = flow[indicator]
valid_correspondence = map((x, Δx)-> x + Δx, valid_points, valid_flow)

@info length(valid_correspondence)

pts0 = map(x-> round.(Int, (last(x), first(x))), points)
pts1 = map(x-> round.(Int, (last(x), first(x))), valid_points)
pts2 = map(x-> round.(Int, (last(x), first(x))), valid_correspondence)
lines = map((p1, p2) -> (p1, p2), pts1, pts2)

guidict = imshow(img1)
idx2 = annotate!(guidict, AnnotationLines(lines, linewidth=1, color=RGB(1, 0, 0), coord_order="xyxy"))
