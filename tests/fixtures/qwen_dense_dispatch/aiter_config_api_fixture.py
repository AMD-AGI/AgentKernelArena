# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Verbatim AITER_CONFIG class excerpt for CPU import/cache regression tests.
# Source: ROCm/aiter d9e5ef7ce08ee7045d583aed768cff41aa9210fe, aiter/jit/core.py.
import functools
import logging
import os
import re

logger = logging.getLogger("aiter-config-api-test")
AITER_ROOT_DIR = ""
AITER_CONFIG_GEMM_BF16 = ""

class AITER_CONFIG:
    @property
    def AITER_CONFIG_GEMM_A4W4_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A4W4",
            AITER_CONFIG_GEMM_A4W4,
            "a4w4_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm"
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
            "a8w8_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
            "a8w8_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_FMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_FMOE", AITER_CONFIG_FMOE, "tuned_fmoe"
        )

    @property
    def AITER_CONFIG_GROUPED_FMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GROUPED_FMOE",
            AITER_CONFIG_GROUPED_FMOE,
            "tuned_grouped_fmoe",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_A8W8_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_A8W8_BATCHED_GEMM",
            AITER_CONFIG_A8W8_BATCHED_GEMM,
            "a8w8_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_BF16_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_BF16_BATCHED_GEMM",
            AITER_CONFIG_BF16_BATCHED_GEMM,
            "bf16_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_BF16_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_BF16", AITER_CONFIG_GEMM_BF16, "bf16_tuned_gemm"
        )

    def update_config_files(self, file_path: str, merge_name: str):
        path_list = file_path.split(os.pathsep) if file_path else []
        if len(path_list) <= 1:
            return file_path
        source_pairs = []
        ## merge config files
        ##example: AITER_CONFIG_GEMM_A4W4="/path1:/path2"
        import pandas as pd

        for i, path in enumerate(path_list):
            if not os.path.exists(path):
                logger.info(f"path {i + 1}: {path} (not exist)")
                continue

            df = pd.read_csv(path)
            source_pairs.append((path, df))

        if not source_pairs:
            raise FileNotFoundError(
                f"No existing config files found in '{file_path}' "
                f"when merging '{merge_name}'."
            )

        _FILL_DEFAULTS = {"xbf16": 0, "run_1stage": 0, "ksplit": 0}
        all_cols = list(source_pairs[0][1].columns)
        for _, df in source_pairs[1:]:
            for c in df.columns:
                if c not in all_cols:
                    insert_before = "tflops" if "tflops" in all_cols else all_cols[-1]
                    all_cols.insert(all_cols.index(insert_before), c)
        for i, (path, df) in enumerate(source_pairs):
            for c in all_cols:
                if c not in df.columns:
                    if c == "gfx" and "cu_num" in df.columns:
                        # Legacy config without a gfx column: infer the arch from
                        # cu_num (256->gfx950, 80/304->gfx942) so archs that share
                        # a cu_num stay distinguishable after the merge.
                        from aiter.jit.utils.chip_info import gfx_from_cu_num

                        df[c] = df["cu_num"].map(gfx_from_cu_num)
                    else:
                        df[c] = _FILL_DEFAULTS.get(c, 0)
            source_pairs[i] = (path, df[all_cols])

        non_empty = [df for _, df in source_pairs if not df.empty]
        merge_df = (
            pd.concat(non_empty, ignore_index=True)
            if non_empty
            else source_pairs[0][1].iloc[0:0].copy()
        )
        has_tag = "_tag" in merge_df.columns
        if has_tag:
            merge_df["_tag"] = merge_df["_tag"].fillna("")

        ## get keys from untuned file to drop_duplicates
        untuned_name = (
            re.sub(r"(?:_)?tuned$", r"\1untuned", merge_name)
            if re.search(r"(?:_)?tuned$", merge_name)
            else merge_name.replace("tuned", "untuned")
        )
        untuned_path = f"{AITER_ROOT_DIR}/aiter/configs/{untuned_name}.csv"
        if os.path.exists(untuned_path):
            untunedf = pd.read_csv(untuned_path)
            keys = untunedf.columns.to_list()
            if "cu_num" not in keys:
                keys.append("cu_num")
            if "gfx" in merge_df.columns and "gfx" not in keys:
                keys.append("gfx")
            dedup_keys = keys + ["_tag"] if has_tag else keys
            duplicated_mask = merge_df.duplicated(subset=dedup_keys, keep=False)
            if duplicated_mask.any():
                dup_count = int(duplicated_mask.sum())
                dup_rows = merge_df[duplicated_mask].sort_values(dedup_keys)
                if "us" not in merge_df.columns:
                    raise RuntimeError(
                        f"Found {dup_count} duplicate shape entries during merge of '{merge_name}'. "
                        f"No 'us' column to determine best performing entry. "
                        f"Please remove duplicates manually.\n"
                        f"Duplicate rows:\n{dup_rows.to_string(index=False)}"
                    )

                # Auto-dedup: globally determine best row (lowest 'us') per shape
                best_row_index = set(
                    merge_df.sort_values("us", kind="stable")
                    .drop_duplicates(subset=dedup_keys, keep="first")
                    .index
                )

                saved_files = []
                offset = 0
                for src_path, src_df in source_pairs:
                    start, end = offset, offset + len(src_df)
                    offset = end
                    file_rows = merge_df.iloc[start:end]
                    new_src_df = file_rows[
                        file_rows.index.isin(best_row_index)
                    ].reset_index(drop=True)
                    if len(new_src_df) < len(src_df):
                        new_src_df.to_csv(src_path, index=False)
                        saved_files.append(
                            f"  {src_path}: {len(src_df)} -> {len(new_src_df)} rows"
                        )
                saved_info = (
                    "\n".join(saved_files) if saved_files else "  (no files updated)"
                )
                raise RuntimeError(
                    f"Found {dup_count} duplicate shape entries during merge of '{merge_name}'. "
                    f"Auto-resolved by keeping best performing (lowest 'us') for each shape "
                    f"and saved back to source config files. Please re-run.\n"
                    f"Duplicate rows:\n{dup_rows.to_string(index=False)}\n"
                    f"Updated files:\n{saved_info}"
                )
        else:
            logger.warning(
                f"Untuned config file not found: {untuned_path}. Using all columns for deduplication."
            )
        from pathlib import Path

        config_path = Path("/tmp/aiter_configs/")
        if not config_path.exists():
            config_path.mkdir(parents=True, exist_ok=True)
        new_file_path = f"{config_path}/{merge_name}.csv"
        lock_path = f"{new_file_path}.lock"
        tmp_file_path = f"{new_file_path}.tmp"

        def write_config():
            merge_df.to_csv(tmp_file_path, index=False)
            os.replace(tmp_file_path, new_file_path)

        mp_lock(lock_path, write_config)
        return new_file_path

    # Cache is keyed on (self, env_name, ...); this object is a
    # process-lifetime singleton, so the retained reference is not a leak.
    @functools.lru_cache(maxsize=20)  # noqa: B019
    def get_config_file(self, env_name, default_file, tuned_file_name):
        config_env_file = os.getenv(env_name)
        # default_file = f"{AITER_ROOT_DIR}/aiter/configs/{tuned_file_name}.csv"
        from pathlib import Path

        if not config_env_file:
            model_config_dir = Path(f"{AITER_ROOT_DIR}/aiter/configs/model_configs/")
            op_tuned_file_list = [
                p
                for p in model_config_dir.glob(f"*{tuned_file_name}*.csv")
                if (p.is_file() and "untuned" not in p.name)
            ]

            if not op_tuned_file_list:
                config_file = default_file
            else:
                tuned_files = ":".join(str(p) for p in op_tuned_file_list)
                tuned_files = default_file + ":" + tuned_files
                logger.info(
                    f"merge tuned file under model_configs/ and configs/ {tuned_files}"
                )
                config_file = self.update_config_files(tuned_files, tuned_file_name)
        else:
            config_file = self.update_config_files(config_env_file, tuned_file_name)
            # print(f"get config file from environment ", config_file)
        return config_file
