package com.qxotic.jota.runtime.xla.pjrt;

import java.io.ByteArrayOutputStream;

/**
 * Serialized {@code CompileOptionsProto} (xla/pjrt/proto/compile_options.proto), written by hand to
 * avoid a protobuf dependency. Only the fields jota sets are supported.
 */
public final class CompileOptions {

    // CompileOptionsProto
    private static final int EXECUTABLE_BUILD_OPTIONS = 3;
    // ExecutableBuildOptionsProto
    private static final int NUM_REPLICAS = 4;
    private static final int NUM_PARTITIONS = 5;
    private static final int DEVICE_ASSIGNMENT = 9;
    // DeviceAssignmentProto (xla/xla_data.proto)
    private static final int REPLICA_COUNT = 1;
    private static final int COMPUTATION_COUNT = 2;
    private static final int COMPUTATION_DEVICES = 3;
    // DeviceAssignmentProto.ComputationDevice
    private static final int REPLICA_DEVICE_IDS = 1;

    private CompileOptions() {}

    /** One replica, one partition: a single-device program, on the client's default device. */
    public static byte[] singleDevice() {
        return singleDevice(-1);
    }

    /**
     * One replica, one partition, assigned to the device whose {@code PJRT_DeviceDescription_Id} is
     * {@code deviceId} (negative: the client's default device).
     */
    public static byte[] singleDevice(int deviceId) {
        ProtoWriter build = new ProtoWriter();
        build.varintField(NUM_REPLICAS, 1);
        build.varintField(NUM_PARTITIONS, 1);
        if (deviceId >= 0) {
            ProtoWriter device = new ProtoWriter();
            device.varintField(REPLICA_DEVICE_IDS, deviceId);
            ProtoWriter assignment = new ProtoWriter();
            assignment.varintField(REPLICA_COUNT, 1);
            assignment.varintField(COMPUTATION_COUNT, 1);
            assignment.bytesField(COMPUTATION_DEVICES, device.toByteArray());
            build.bytesField(DEVICE_ASSIGNMENT, assignment.toByteArray());
        }
        ProtoWriter options = new ProtoWriter();
        options.bytesField(EXECUTABLE_BUILD_OPTIONS, build.toByteArray());
        return options.toByteArray();
    }

    /** The protobuf wire format, varint and length-delimited fields only. */
    static final class ProtoWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void varintField(int field, long value) {
            varint((long) field << 3);
            varint(value);
        }

        void bytesField(int field, byte[] bytes) {
            varint(((long) field << 3) | 2);
            varint(bytes.length);
            out.writeBytes(bytes);
        }

        private void varint(long v) {
            while ((v & ~0x7FL) != 0) {
                out.write((int) ((v & 0x7F) | 0x80));
                v >>>= 7;
            }
            out.write((int) v);
        }

        byte[] toByteArray() {
            return out.toByteArray();
        }
    }
}
