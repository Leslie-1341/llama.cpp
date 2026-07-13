#include "llama-memory.h"

const char * llama_paged_swap_error_reason_name(llama_paged_swap_error_reason reason) {
    switch (reason) {
        case llama_paged_swap_error_reason::NONE:                           return "NONE";
        case llama_paged_swap_error_reason::SWAP_IN_IO_FAILURE:             return "SWAP_IN_IO_FAILURE";
        case llama_paged_swap_error_reason::ACTIVE_VISIBLE_RESTORE_FAILURE: return "ACTIVE_VISIBLE_RESTORE_FAILURE";
        case llama_paged_swap_error_reason::NO_DUMMY_RESTORE_FAILURE:       return "NO_DUMMY_RESTORE_FAILURE";
        case llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK:      return "ACTIVE_READ_RELEASED_BLOCK";
        case llama_paged_swap_error_reason::PAGED_ROW_MAPPING_INVALID:       return "PAGED_ROW_MAPPING_INVALID";
        case llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID:     return "PAGED_WRITE_MAPPING_INVALID";
        case llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT:         return "ACTIVE_ROW_NOT_RESIDENT";
        case llama_paged_swap_error_reason::INPUT_SETUP_FAILURE:             return "INPUT_SETUP_FAILURE";
    }

    return "UNKNOWN";
}

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
