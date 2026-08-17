#pragma once

// Production `/slots.kv_resident` physical-resident observation helpers.
//
// These helpers are intentionally independent of the transaction-local OFFLOAD
// marker (`kv_g0_s1_resident_observation`, gated on mode "1"/"both") and of the
// periodic `LLAMA_KV_RESIDENT_PREFLIGHT=1` whole-KV+claimant telemetry
// (`kv_g0_s1_resident_preflight`).  They restore the `/slots` mincore authority
// that was removed in commit 9aefabb1e without re-merging the three contracts
// into a shared boolean.
//
// See the MODE ROUTING MATRIX below: `LLAMA_KV_G0_S1_RESIDENT_OBSERVATION`
// selects which of the two G0-S1 observation contracts is active, while
// `LLAMA_KV_RESIDENT_PREFLIGHT=1` remains a separate, orthogonal switch that
// must not perturb this matrix.

#include "llama-kv-cache-action.h"

#include "common.h"
#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cstring>

// Use the fully-qualified ordered_json type so this header is safe to include
// from translation units that already define (or reuse) their own `json` alias.
using kv_slots_observation_json = nlohmann::ordered_json;

// Mode routing for the `/slots.kv_resident` physical authority.
//
//   mode        transaction marker   /slots kv_resident
//   1            yes                 no
//   preflight    no                  yes
//   both         yes                 yes
//   empty/other  no                  no
struct server_kv_slots_resident_routing {
    bool transaction_marker = false;   // OFFLOAD before/after physical observation
    bool slots_observation  = false;   // whole-KV snapshot copied to every slot
};

// Parse the `LLAMA_KV_G0_S1_RESIDENT_OBSERVATION` mode string into the routing
// pair.  `nullptr` / empty / unrecognized values yield both flags false
// (fail-closed).  This function is pure — no environment reads, no globals — so
// it can be unit-tested directly.
inline server_kv_slots_resident_routing parse_kv_slots_resident_routing(const char * mode) {
    server_kv_slots_resident_routing r;
    if (mode == nullptr) {
        return r;
    }
    if (std::strcmp(mode, "1") == 0) {
        r.transaction_marker = true;
    } else if (std::strcmp(mode, "preflight") == 0) {
        r.slots_observation = true;
    } else if (std::strcmp(mode, "both") == 0) {
        r.transaction_marker = true;
        r.slots_observation  = true;
    }
    return r;
}

// Build the `/slots.kv_resident` JSON object from a whole-KV mincore sample.
//
// Available sample → the 9-field canonical schema required by the Final F16
// parser contract.  Unavailable probe → `{"status":"unavailable"}` fail-closed
// (never a zero, never a proxy by budget / logical / requested / advised /
// backing bytes, and never reinterpreted measurement semantics).
//
// Returns an empty object when the slots-observation mode is disabled, so the
// caller can decide whether to attach the field at all.
inline kv_slots_observation_json format_kv_slots_resident_observation(const llama_kv_resident_sample & s) {
    kv_slots_observation_json out;
    if (s.available) {
        out = kv_slots_observation_json {
            {"status", "available"},
            {"source", "paged_sample_mincore"},
            {"object_id", s.object_id},
            {"generation", s.generation},
            {"page_size", s.page_size},
            {"total_bytes", s.total_bytes},
            {"resident_bytes", s.resident_bytes},
            {"total_pages", s.total_pages},
            {"resident_pages", s.resident_pages},
        };
    } else {
        out = kv_slots_observation_json {
            {"status", "unavailable"},
        };
    }
    return out;
}
