/* residency_iface.h -- THE SEAM between placement and the run phase.
 *
 * core/runtime/ owns the declaration and this file only forwards to it. Placement does not keep a
 * copy: two declarations of a struct that crosses a component boundary is an ODR violation waiting
 * for the first translation unit that includes both, and the symptom would be a WeightSlot whose
 * fields the two components disagree about.
 *
 * The contract itself is documented where it is declared. Read core/runtime/residency.h -- in
 * particular the rule that begin_transfer and publish are the ONLY writers and that BOTH must bump
 * the generation, because the run phase memoises the stream wait per generation and installing a
 * new event at the old generation makes it skip the wait entirely.
 */
#pragma once
#include "runtime/residency.h"
