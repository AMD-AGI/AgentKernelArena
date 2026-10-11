"""Choose only a validated image-native backend for each recorded geometry."""

def select_routes():
    return {
        'prefill_m16384_n4608_k8192': "recorded",
        'decode_m64_n32_k8192': "recorded",
        'decode_m64_n512_k8192': "recorded",
        'decode_m64_n4608_k8192': "recorded",
        'decode_m64_n8192_k256': "recorded",
    }
