using Documenter
using SLAM

makedocs(
    sitename="SLAM.jl",
    authors="Anton Smirnov",
    repo="https://github.com/pxl-th/SLAM.jl/blob/{commit}{path}#L{line}",
    modules=[SLAM],
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Slam Manager" => "api/slam-manager.md",
            "Front-End" => "api/front-end.md",
            "Estimator" => "api/estimator.md",
            "Map Manager" => "api/map-manager.md",
            "Parameters" => "api/params.md",
            "IO" => "api/io.md",
        ]
    ],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
)
