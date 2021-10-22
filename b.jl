using Images
using ImageFiltering
using BenchmarkTools

function main()
    image = load("/home/pxl-th/Downloads/FastGaussianBlur-main/data/demo.png") .|> RGB{Float64}
    out = similar(image)
    outIy = similar(image)
    border = Fill(zero(RGB{Float64}))

    kerng = KernelFactors.IIRGaussian(2.0)
    kernel = ntuple(_ -> kerng, Val(2))

    imfilter!(out, image, kernel)
    save("/home/pxl-th/b.png", out)
    @btime imfilter!($out, $image, $kernel)

    imfilter!(outIy, image, KernelFactors.scharr((true, true), 1), border)
    save("/home/pxl-th/g.png", map(clamp01nan, outIy))
    @btime imfilter!($outIy, $image, KernelFactors.scharr((true, true), 1), $border)
    @btime imfilter!($outIy, $image, KernelFactors.scharr((true, true), 1))
end
main()
