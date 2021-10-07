using Documenter
using SLAM

makedocs(
    sitename="SLAM.jl",
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/SLAM.jl/blob/{commit}{path}#L{line}",
    modules=[SLAM],
    pages=[
        "Home" => "index.md",
        "API Reference" => ["api.md"]
    ],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
)
