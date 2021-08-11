using PackageCompiler

create_sysimage(
    [
        :LinearAlgebra,
        :StaticArrays,
        :Images,
        :ImageFeatures,
        :ImageTracking,
        :VideoIO,
        :Rotations,
        :Manifolds,
        :Parameters,
    ]; sysimage_path="slam.so", precompile_execution_file="precompile.jl",
)
