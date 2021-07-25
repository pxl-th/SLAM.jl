using PackageCompiler

create_sysimage(
    [
        :LinearAlgebra,
        :StaticArrays,
        :GeometryBasics,
        :Images,
        :ImageFeatures,
        :ImageTracking,
        :VideoIO,
        :Rotations,
        :Manifolds,
    ]; sysimage_path="sys-slam.so", precompile_execution_file="precompile.jl",
)
