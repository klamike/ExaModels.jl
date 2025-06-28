module ExaModelsMetal

import ExaModels: ExaModels
import Metal: Metal, MetalBackend, MtlArray


ExaModels.convert_array(v, ::MetalBackend) = MtlArray(v)
ExaModels.convert_array(v::AbstractArray{Float64}, ::MetalBackend) = MtlArray{Float32}(v)
ExaModels.convert_array(v::AbstractVector{<:NamedTuple}, ::MetalBackend) = MtlArray(_namedtuple_float64_32.(v))
ExaModels.convert_array(v::AbstractVector{<:Tuple}, ::MetalBackend) = MtlArray(_tuple_float64_32.(v))
ExaModels.default_float_type(::MetalBackend) = Float32
ExaModels.sort!(array::A; lt = isless) where {A<:MtlArray} = copyto!(array, sort!(Array(array); lt = lt))

_float64_32(x::Float64) = convert(Float32, x)
_float64_32(x) = x
_tuple_float64_32(t::Tuple) = Tuple(_float64_32.(t))
_namedtuple_float64_32(nt::NamedTuple) = NamedTuple{keys(nt)}(_float64_32.(values(nt)))
end
