package com.qxotic.jota.runtime.xla;

import com.qxotic.jota.DataType;
import com.qxotic.jota.runtime.xla.pjrt.PjrtBufferType;
import java.util.Map;
import java.util.Optional;

/** jota {@link DataType}s that XLA buffers can hold, and their {@link PjrtBufferType}. */
public final class XlaDataTypes {

    private static final Map<DataType, PjrtBufferType> TO_PJRT =
            Map.of(
                    DataType.BOOL, PjrtBufferType.PRED,
                    DataType.I8, PjrtBufferType.S8,
                    DataType.I16, PjrtBufferType.S16,
                    DataType.I32, PjrtBufferType.S32,
                    DataType.I64, PjrtBufferType.S64,
                    DataType.FP16, PjrtBufferType.F16,
                    DataType.BF16, PjrtBufferType.BF16,
                    DataType.FP32, PjrtBufferType.F32,
                    DataType.FP64, PjrtBufferType.F64);

    private XlaDataTypes() {}

    /** True when XLA can hold {@code dataType} (block-quantized types cannot). */
    public static boolean isSupported(DataType dataType) {
        return TO_PJRT.containsKey(dataType);
    }

    /** The buffer type of {@code dataType}, or empty. */
    public static Optional<PjrtBufferType> find(DataType dataType) {
        return Optional.ofNullable(TO_PJRT.get(dataType));
    }

    /** The buffer type of {@code dataType}; throws for types XLA cannot hold. */
    public static PjrtBufferType of(DataType dataType) {
        PjrtBufferType t = TO_PJRT.get(dataType);
        if (t == null) {
            throw new UnsupportedOperationException("XLA backend does not support " + dataType);
        }
        return t;
    }
}
