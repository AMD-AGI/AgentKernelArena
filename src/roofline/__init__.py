"""Evaluator-side roofline floors for schema-v3 kernel tasks.

A floor is a time no correct implementation of a task case can beat on the
target GPU: ``max(F_matrix / P_matrix, F_vector / P_vector, B_min / BW)``.
Every counting rule errs low (fewer bytes, fewer FLOPs, higher peaks), so a
floor can understate the remaining headroom but never overstate it.

This package reads task packages and never changes them. Task code does not
import it and it is not materialized into workspaces. Generated floors,
joined measurements and measured peaks belong in an output directory outside
the repository.
"""
