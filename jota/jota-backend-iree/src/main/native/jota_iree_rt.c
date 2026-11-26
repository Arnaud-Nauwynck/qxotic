// libjota_iree_rt: re-exports the IREE runtime C API (iree/runtime/api.h and the VM / HAL APIs it
// pulls in) from one shared object, linked whole from iree_runtime_unified.
//
// The few helpers below give the Java binding by-value-free entry points for IREE functions that
// take or return small structs by value (FFM can call those too, but pointer forms keep the
// binding simple and independent of struct-passing ABI details).

#include "iree/async/util/proactor_pool.h"
#include "iree/runtime/api.h"

// Version of this shim's helper ABI, bumped on any change below.
int jota_iree_rt_abi_version(void) { return 1; }

// iree_status_t is a pointer; formats it into a heap string the caller frees with
// jota_iree_rt_free. Consumes (frees) the status. Returns NULL for OK.
char* jota_iree_rt_status_consume(iree_status_t status) {
  if (iree_status_is_ok(status)) return NULL;
  char* buffer = NULL;
  iree_host_size_t length = 0;
  iree_allocator_t allocator = iree_allocator_system();
  if (!iree_status_to_string(status, &allocator, &buffer, &length)) {
    buffer = NULL;
  }
  iree_status_free(status);
  return buffer;
}

int jota_iree_rt_status_code(iree_status_t status) {
  return (int)iree_status_code(status);
}

void jota_iree_rt_free(void* ptr) { iree_allocator_free(iree_allocator_system(), ptr); }

// Instance with all compiled-in drivers registered.
iree_status_t jota_iree_rt_instance_create(iree_runtime_instance_t** out_instance) {
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  return iree_runtime_instance_create(&options, iree_allocator_system(), out_instance);
}

// Device from a driver name or URI such as "local-task", "local-sync", "cuda://0".
// Devices require a proactor pool (async I/O threads) in their create params, as
// iree_runtime_instance_try_create_default_device does; the device retains it.
iree_status_t jota_iree_rt_device_create(iree_runtime_instance_t* instance, const char* uri,
                                         iree_hal_device_t** out_device) {
  *out_device = NULL;
  iree_allocator_t host_allocator = iree_runtime_instance_host_allocator(instance);
  iree_async_proactor_pool_t* proactor_pool = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL, iree_async_proactor_pool_options_default(),
      host_allocator, &proactor_pool));
  iree_hal_device_create_params_t params = iree_hal_device_create_params_default();
  params.proactor_pool = proactor_pool;
  iree_status_t status =
      iree_hal_create_device(iree_runtime_instance_driver_registry(instance),
                             iree_make_cstring_view(uri), &params, host_allocator, out_device);
  iree_async_proactor_pool_release(proactor_pool);
  return status;
}

// Session (VM context) bound to one device.
iree_status_t jota_iree_rt_session_create(iree_runtime_instance_t* instance,
                                          iree_hal_device_t* device,
                                          iree_runtime_session_t** out_session) {
  iree_runtime_session_options_t options;
  iree_runtime_session_options_initialize(&options);
  return iree_runtime_session_create_with_device(
      instance, &options, device, iree_runtime_instance_host_allocator(instance), out_session);
}

// Appends a .vmfb held in caller memory, which must outlive the session.
iree_status_t jota_iree_rt_session_append_module(iree_runtime_session_t* session,
                                                 const uint8_t* data, iree_host_size_t length) {
  return iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(data, length), iree_allocator_null());
}

// A call to `function_name` (e.g. "module.main"), heap allocated; release with
// jota_iree_rt_call_release.
iree_status_t jota_iree_rt_call_create(iree_runtime_session_t* session, const char* function_name,
                                       iree_runtime_call_t** out_call) {
  *out_call = NULL;
  iree_runtime_call_t* call = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(iree_allocator_system(), sizeof(*call), (void**)&call));
  iree_status_t status = iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(function_name), call);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), call);
    return status;
  }
  *out_call = call;
  return iree_ok_status();
}

void jota_iree_rt_call_release(iree_runtime_call_t* call) {
  if (!call) return;
  iree_runtime_call_deinitialize(call);
  iree_allocator_free(iree_allocator_system(), call);
}

// Copies a dense row-major host buffer to a new device buffer view (retained by the call).
iree_status_t jota_iree_rt_call_push_input(iree_runtime_call_t* call, const int64_t* shape,
                                           iree_host_size_t rank, uint32_t element_type,
                                           const void* data, iree_host_size_t byte_length) {
  iree_runtime_session_t* session = call->session;
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_dim_t dims[16];
  if (rank > IREE_ARRAYSIZE(dims)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "rank %zu > 16", (size_t)rank);
  }
  for (iree_host_size_t i = 0; i < rank; ++i) dims[i] = (iree_hal_dim_t)shape[i];
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_hal_buffer_view_t* view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_runtime_session_device_allocator(session), rank, dims,
      (iree_hal_element_type_t)element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, byte_length), &view));
  iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(call, view);
  iree_hal_buffer_view_release(view);
  return status;
}

iree_status_t jota_iree_rt_call_invoke(iree_runtime_call_t* call) {
  return iree_runtime_call_invoke(call, /*flags=*/0);
}

// Number of outputs left to pop after an invoke.
iree_host_size_t jota_iree_rt_call_output_count(iree_runtime_call_t* call) {
  return iree_vm_list_size(iree_runtime_call_outputs(call));
}

// Pops the next output buffer view; the caller owns the returned reference.
iree_status_t jota_iree_rt_call_pop_output(iree_runtime_call_t* call,
                                           iree_hal_buffer_view_t** out_view) {
  return iree_runtime_call_outputs_pop_front_buffer_view(call, out_view);
}

iree_host_size_t jota_iree_rt_buffer_view_rank(iree_hal_buffer_view_t* view) {
  return iree_hal_buffer_view_shape_rank(view);
}

int64_t jota_iree_rt_buffer_view_dim(iree_hal_buffer_view_t* view, iree_host_size_t i) {
  return (int64_t)iree_hal_buffer_view_shape_dim(view, i);
}

uint32_t jota_iree_rt_buffer_view_element_type(iree_hal_buffer_view_t* view) {
  return (uint32_t)iree_hal_buffer_view_element_type(view);
}

iree_device_size_t jota_iree_rt_buffer_view_byte_length(iree_hal_buffer_view_t* view) {
  return iree_hal_buffer_view_byte_length(view);
}

// Copies the whole buffer view to host memory (blocking device-to-host transfer).
iree_status_t jota_iree_rt_buffer_view_read(iree_runtime_session_t* session,
                                            iree_hal_buffer_view_t* view, void* target,
                                            iree_device_size_t length) {
  return iree_hal_device_transfer_d2h(
      iree_runtime_session_device(session), iree_hal_buffer_view_buffer(view), 0, target,
      length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}
