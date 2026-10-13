# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The protected C++ host program owns this task's independent numerical reference.

Unlike the Python extension tasks there are no unverified extra performance
variants. The original correctness action checks every declared shape against
its host reference; the native timing driver additionally validates graph replay.
These structural checks make no claim of CPU or GPU numerical validation.
"""


def self_test(harness):
    if not harness.TEST_SHAPES or any(any(type(x) is not int or x <= 0 for x in shape)
                                      for shape in harness.TEST_SHAPES):
        raise ValueError("Native host workload must contain positive integer dimensions")


def check_additional_paths(harness):
    # All numerical checks are in the original protected host and replay driver.
    # Called only after the native correctness executable successfully ran.
    return None
