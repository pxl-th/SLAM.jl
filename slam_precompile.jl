using PackageCompiler

create_sysimage(
    [
        :LinearAlgebra,
        :StaticArrays,
        :Images,
        :ImageDraw,
        :ImageFeatures,
        :ImageTracking,
        :Rotations,
        :Manifolds,
        :DataStructures,
        :GLMakie,

        :LeastSquaresOptim,
        :SparseArrays,
        :SparseDiffTools,
    ]; sysimage_path="slam.so", precompile_execution_file="precompile.jl",
)
