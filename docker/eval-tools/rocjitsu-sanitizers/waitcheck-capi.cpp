// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <rocjitsu/analysis/rj_waitcheck.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Diagnostic {
  std::string code;
  std::string kernel_name;
  uint64_t kernel_entry = 0;
  int counter = 0;
  int access = 0;
  int register_class = 0;
  unsigned register_index = 0;
  unsigned register_width = 0;
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t file_offset = 0;
  std::string instruction;
  uint64_t producer_section_offset = 0;
  uint64_t producer_file_offset = 0;
  std::string producer_instruction;
  uint32_t required_count = 0;
  std::string message;
};

std::string copy_string(const char *value) { return value == nullptr ? "" : value; }

void write_json_string(std::ostream &output, std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
    case '"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (character < 0x20) {
        output << "\\u00" << hex[character >> 4] << hex[character & 0xf];
      } else {
        output << character;
      }
    }
  }
  output << '"';
}

void diagnostic_callback(const rj_waitcheck_diagnostic_t *input, void *opaque) {
  if (input == nullptr || opaque == nullptr)
    return;
  auto &diagnostics = *static_cast<std::vector<Diagnostic> *>(opaque);
  Diagnostic value;
  value.code = copy_string(rj_waitcheck_diagnostic_code_name(input->code));
  value.kernel_name = copy_string(input->kernel_name);
  value.kernel_entry = input->kernel_entry_offset;
  value.counter = static_cast<int>(input->counter);
  value.access = static_cast<int>(input->access);
  value.register_class = static_cast<int>(input->reg.register_class);
  value.register_index = input->reg.index;
  value.register_width = input->reg.width;
  value.section_name = copy_string(input->section_name);
  value.section_offset = input->section_offset;
  value.file_offset = input->file_offset;
  value.instruction = copy_string(input->instruction);
  value.producer_section_offset = input->producer_section_offset;
  value.producer_file_offset = input->producer_file_offset;
  value.producer_instruction = copy_string(input->producer_instruction);
  value.required_count = input->required_count;
  value.message = copy_string(input->message);
  diagnostics.push_back(std::move(value));
}

void error_callback(const char *message, void *opaque) {
  if (opaque != nullptr)
    *static_cast<std::string *>(opaque) = copy_string(message);
}

bool parse_uint64(std::string_view text, uint64_t &value) {
  try {
    size_t consumed = 0;
    value = std::stoull(std::string(text), &consumed, 0);
    return consumed == text.size();
  } catch (...) {
    return false;
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string path;
  uint64_t kernel_entry = 0;
  bool has_entry = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--code-object" && index + 1 < argc) {
      path = argv[++index];
    } else if (argument == "--kernel-entry" && index + 1 < argc) {
      has_entry = parse_uint64(argv[++index], kernel_entry);
    } else if (argument == "--help") {
      std::cout << "usage: aka-waitcheck-capi --code-object PATH --kernel-entry OFFSET\n";
      return 0;
    } else {
      std::cerr << "invalid argument: " << argument << "\n";
      return 2;
    }
  }
  if (path.empty() || !has_entry) {
    std::cerr << "code object and kernel entry are required\n";
    return 2;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "could not open code object\n";
    return 2;
  }
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  if (bytes.empty()) {
    std::cerr << "code object is empty\n";
    return 2;
  }

  std::vector<Diagnostic> diagnostics;
  std::string analysis_error;
  rj_waitcheck_options_t options{};
  rj_waitcheck_result_t result{};
  if (rj_waitcheck_options_init(&options, sizeof(options)) != ROCJITSU_STATUS_SUCCESS ||
      rj_waitcheck_result_init(&result, sizeof(result)) != ROCJITSU_STATUS_SUCCESS) {
    std::cerr << "could not initialize waitcheck ABI structures\n";
    return 2;
  }
  options.max_diagnostics = 0;
  options.stop_after_first_diagnostic = 0;
  options.diagnostic_callback = diagnostic_callback;
  options.error_callback = error_callback;
  // Both callbacks need distinct storage, so switch user_data immediately
  // around the API call through a small aggregate.
  struct CallbackState {
    std::vector<Diagnostic> *diagnostics;
    std::string *error;
  } state{&diagnostics, &analysis_error};
  options.user_data = &state;
  options.diagnostic_callback = [](const rj_waitcheck_diagnostic_t *value, void *opaque) {
    auto *callback_state = static_cast<CallbackState *>(opaque);
    diagnostic_callback(value, callback_state->diagnostics);
  };
  options.error_callback = [](const char *value, void *opaque) {
    auto *callback_state = static_cast<CallbackState *>(opaque);
    error_callback(value, callback_state->error);
  };

  const rj_status_t status = rj_waitcheck_analyze_kernel(
      bytes.data(), bytes.size(), kernel_entry, &options, &result);
  std::cout << "AKA_WAITCHECK_CAPI_RESULT {\"schema_version\":1,\"api_status\":"
            << static_cast<int>(status) << ",\"analysis_complete\":"
            << (status == ROCJITSU_STATUS_SUCCESS ? "true" : "false") << ",\"target\":";
  write_json_string(std::cout, rj_waitcheck_target_name(result.target));
  std::cout << ",\"instructions_analyzed\":" << result.instructions_analyzed
            << ",\"memory_events_tracked\":" << result.memory_events_tracked
            << ",\"kernels_discovered\":" << result.kernels_discovered
            << ",\"kernels_analyzed\":" << result.kernels_analyzed
            << ",\"diagnostics_observed\":" << result.diagnostics_observed
            << ",\"diagnostics_reported\":" << result.diagnostics_reported
            << ",\"passed\":" << (result.passed ? "true" : "false")
            << ",\"diagnostics_truncated\":"
            << (result.diagnostics_truncated ? "true" : "false")
            << ",\"stopped_early\":" << (result.stopped_early ? "true" : "false")
            << ",\"analysis_error\":";
  write_json_string(std::cout, analysis_error);
  std::cout << ",\"diagnostics\":[";
  for (size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    const Diagnostic &value = diagnostics[index];
    std::cout << "{\"code\":";
    write_json_string(std::cout, value.code);
    std::cout << ",\"kernel_name\":";
    write_json_string(std::cout, value.kernel_name);
    std::cout << ",\"kernel_entry\":" << value.kernel_entry
              << ",\"counter\":" << value.counter << ",\"access\":" << value.access
              << ",\"register_class\":" << value.register_class
              << ",\"register_index\":" << value.register_index
              << ",\"register_width\":" << value.register_width
              << ",\"section_name\":";
    write_json_string(std::cout, value.section_name);
    std::cout << ",\"section_offset\":" << value.section_offset
              << ",\"file_offset\":" << value.file_offset << ",\"instruction\":";
    write_json_string(std::cout, value.instruction);
    std::cout << ",\"producer_section_offset\":" << value.producer_section_offset
              << ",\"producer_file_offset\":" << value.producer_file_offset
              << ",\"producer_instruction\":";
    write_json_string(std::cout, value.producer_instruction);
    std::cout << ",\"required_count\":" << value.required_count << ",\"message\":";
    write_json_string(std::cout, value.message);
    std::cout << '}';
  }
  std::cout << "]}\n";
  return status == ROCJITSU_STATUS_SUCCESS ? 0 : 2;
}
