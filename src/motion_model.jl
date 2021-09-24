"""
Constant Velocity Motion Model
"""
mutable struct MotionModel
    prev_time::Float64
    """
    Element of SE(3) group.
    """
    prev_wc::SMatrix{4, 4, Float64}
    """
    Element of se(3) algebra.
    """
    log_rel_t::SMatrix{4, 4, Float64}
end

function MotionModel(;
    prev_time::Float64 = -1.0,
    prev_wc::SMatrix{4, 4, Float64} = SMatrix{4, 4, Float64}(I),
    log_rel_t::SMatrix{4, 4, Float64} = zeros(SMatrix{4, 4, Float64}),
)
    MotionModel(prev_time, prev_wc, log_rel_t)
end

function reset!(m::MotionModel)
    m.prev_time = -1.0
    m.log_rel_t = zeros(SMatrix{4, 4, Float64})
end

"""
Apply Motion Model to a given `wc` transformation.
"""
function (m::MotionModel)(wc::SMatrix{4, 4, Float64}, time)
    m.prev_time < 0 && return wc
    # `wc` and `m.prev_wc` should be equal here,
    # since prev_wc is updated right after pose computation.
    # If not - update to stay consistent.
    δ = log_lie(SE3, wc * inv(SE3, m.prev_wc))
    all(isapprox.(δ, 0; atol=1e-5)) || (m.prev_wc = wc;)

    δt = time - m.prev_time
    wc * exp_lie(SE3, m.log_rel_t .* δt)
end

function update!(m::MotionModel, wc::SMatrix{4, 4, Float64}, time)
    if m.prev_time < 0
        m.prev_time = time
        m.prev_wc = wc
        return
    end

    δt = time - m.prev_time
    m.prev_time = time
    δt < 0 && error(
        "Got older than previous image! " *
        "Previous time $(m.prev_time) vs time $time."
    )

    new_rel = inv(SE3, m.prev_wc) * wc
    m.log_rel_t = log_lie(SE3, new_rel) ./ δt
    m.prev_wc = wc
end
