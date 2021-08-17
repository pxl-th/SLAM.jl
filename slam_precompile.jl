using PackageCompiler

create_sysimage(
    [
        :LinearAlgebra,
        :StaticArrays,
        :Images,
        :ImageDraw,
        :ImageFeatures,
        :ImageTracking,
        :VideoIO,
        :Rotations,
        :Manifolds,
        :Parameters,
        :DataStructures,
    ]; sysimage_path="slam.so", precompile_execution_file="precompile.jl",
)