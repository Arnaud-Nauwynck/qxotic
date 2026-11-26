package com.qxotic.jota.runtime.xla.pjrt;

import java.lang.foreign.MemorySegment;

/**
 * An addressable {@code PJRT_Device} of a client: its handle (owned by the client), its position in
 * {@code PJRT_Client_AddressableDevices}, its {@code PJRT_DeviceDescription_Id} (what device
 * assignments name) and its kind (e.g. {@code "cpu"}, {@code "NVIDIA H100"}).
 */
public record PjrtDevice(MemorySegment handle, int index, int id, String kind) {}
