package com.qxotic.jota.runtime.isolated;

import com.qxotic.jota.runtime.isolated.spi.CallContext;
import com.qxotic.jota.runtime.isolated.spi.IsolatedService;
import com.qxotic.jota.runtime.isolated.wire.Wire;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/** Child-side service used by the tests: a stateful counter and a shared-memory kernel. */
public final class TestService implements IsolatedService {

    private final AtomicLong counter = new AtomicLong();

    @Override
    public String name() {
        return "test";
    }

    @Override
    public byte[] invoke(String method, byte[] payload, CallContext context) {
        return switch (method) {
            case "increment" -> Wire.writer().i64(counter.incrementAndGet()).toByteArray();
            case "scaleShm" -> {
                // (path, count, factor): x[i] *= factor over `count` floats, in place; returns sum
                Wire.Reader r = Wire.reader(payload);
                Path path = Path.of(r.string());
                int count = r.i32();
                float factor = (float) r.f64();
                try (SharedMemoryFile shm = SharedMemoryFile.open(path)) {
                    MemorySegment s = shm.segment();
                    double sum = 0;
                    for (int i = 0; i < count; i++) {
                        float v = s.getAtIndex(ValueLayout.JAVA_FLOAT, i) * factor;
                        s.setAtIndex(ValueLayout.JAVA_FLOAT, i, v);
                        sum += v;
                    }
                    yield Wire.writer().f64(sum).toByteArray();
                }
            }
            default -> throw new UnsupportedOperationException("test." + method);
        };
    }
}
