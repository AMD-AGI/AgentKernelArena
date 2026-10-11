"""Select a native SG520 block-size policy for each original captured case."""


def select_routes():
    return {
        "decode_m64_live": "captured",
        "prefill_m8192_live": "captured",
    }
