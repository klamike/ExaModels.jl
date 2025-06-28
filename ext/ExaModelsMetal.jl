module ExaModelsMetal

import ExaModels: ExaModels
import Metal: Metal, MetalBackend, MtlArray


ExaModels.convert_array(v, ::MetalBackend) = MtlArray(v)

end
