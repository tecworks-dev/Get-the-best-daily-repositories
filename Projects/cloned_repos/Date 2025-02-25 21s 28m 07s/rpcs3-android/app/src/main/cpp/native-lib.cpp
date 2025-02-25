#include "Emu/CPU/CPUDisAsm.h"
#include "Emu/CPU/CPUThread.h"
#include "Emu/Cell/PPUDisAsm.h"
#include "Emu/Cell/PPUThread.h"
#include "Emu/Cell/SPUDisAsm.h"
#include "Emu/Cell/SPUThread.h"
#include "Emu/IdManager.h"
#include "Emu/RSX/RSXDisAsm.h"
#include "Emu/system_config.h"
#include "Utilities/Thread.h"
#include "Utilities/cheat_info.h"
#include "Utilities/date_time.h"
#include "rpcs3_version.h"
#include "util/console.h"
#include <Emu/Cell/Modules/cellSaveData.h>
#include <Emu/Cell/Modules/sceNpTrophy.h>
#include <Emu/Io/pad_config.h>
#include <Emu/RSX/GSFrameBase.h>
#include <Emu/System.h>

#include <jni.h>
#include <string>

std::string g_input_config_override;
cfg_input_configurations g_cfg_input_configs;

void check_microphone_permissions() {}

bool is_using_interpreter(thread_class t_class) {
  switch (t_class) {
  case thread_class::ppu:
    return g_cfg.core.ppu_decoder != ppu_decoder_type::llvm;
  case thread_class::spu:
    return g_cfg.core.spu_decoder != spu_decoder_type::asmjit &&
           g_cfg.core.spu_decoder != spu_decoder_type::llvm;
  default:
    return true;
  }
}

std::shared_ptr<CPUDisAsm> make_disasm(const cpu_thread *cpu,
                                       shared_ptr<cpu_thread> handle) {
  if (!handle) {
    switch (cpu->get_class()) {
    case thread_class::ppu:
      handle = idm::get_unlocked<named_thread<ppu_thread>>(cpu->id);
      break;
    case thread_class::spu:
      handle = idm::get_unlocked<named_thread<spu_thread>>(cpu->id);
      break;
    default:
      break;
    }
  }

  std::shared_ptr<CPUDisAsm> result;

  switch (cpu->get_class()) {
  case thread_class::ppu:
    result = std::make_shared<PPUDisAsm>(cpu_disasm_mode::interpreter,
                                         vm::g_sudo_addr);
    break;
  case thread_class::spu:
    result = std::make_shared<SPUDisAsm>(
        cpu_disasm_mode::interpreter, static_cast<const spu_thread *>(cpu)->ls);
    break;
  case thread_class::rsx:
    result = std::make_shared<RSXDisAsm>(cpu_disasm_mode::interpreter,
                                         vm::g_sudo_addr, 0, cpu);
    break;
  default:
    return result;
  }

  result->set_cpu_handle(std::move(handle));
  return result;
}

bool is_input_allowed() { return true; }
void jit_announce(uptr, usz, std::string_view);

[[noreturn]] void report_fatal_error(std::string_view _text,
                                     bool is_html = false,
                                     bool include_help_text = true) {
  std::string buf;

  buf = std::string(_text);

  // Check if thread id is in string
  if (_text.find("\nThread id = "sv) == umax && !thread_ctrl::is_main()) {
    // Append thread id if it isn't already, except on main thread
    fmt::append(buf, "\n\nThread id = %u.", thread_ctrl::get_tid());
  }

  if (!g_tls_serialize_name.empty()) {
    fmt::append(buf, "\nSerialized Object: %s", g_tls_serialize_name);
  }

  const system_state state = Emu.GetStatus(false);

  if (state == system_state::stopped) {
    fmt::append(buf, "\nEmulation is stopped");
  } else {
    const std::string &name = Emu.GetTitleAndTitleID();
    fmt::append(buf, "\nTitle: \"%s\" (emulation is %s)",
                name.empty() ? "N/A" : name.data(),
                state == system_state::stopping ? "stopping" : "running");
  }

  fmt::append(buf, "\nBuild: \"%s\"", rpcs3::get_verbose_version());
  fmt::append(buf, "\nDate: \"%s\"", std::chrono::system_clock::now());

  std::string_view text = _text;

  utils::attach_console(utils::console_stream::std_err, true);

  utils::output_stderr(fmt::format("RPCS3: %s\n", text));

  jit_announce(0, 0, "");
  std::abort();
}

void disable_display_sleep()
{
}

void enable_display_sleep()
{
}

void qt_events_aware_op(int repeat_duration_ms, std::function<bool()> wrapped_op)
{
  /// ?????
}

template <>
void fmt_class_string<std::chrono::sys_time<typename std::chrono::system_clock::duration>>::format(std::string& out, u64 arg)
{
	const std::time_t dateTime = std::chrono::system_clock::to_time_t(get_object(arg));
 	out += date_time::fmt_time("%Y-%m-%dT%H:%M:%S", dateTime);
}

template <>
void fmt_class_string<cheat_type>::format(std::string &out, u64 arg) {
  format_enum(out, arg, [](cheat_type value) {
    switch (value) {
    case cheat_type::unsigned_8_cheat:
      return "Unsigned 8 bits";
    case cheat_type::unsigned_16_cheat:
      return "Unsigned 16 bits";
    case cheat_type::unsigned_32_cheat:
      return "Unsigned 32 bits";
    case cheat_type::unsigned_64_cheat:
      return "Unsigned 64 bits";
    case cheat_type::signed_8_cheat:
      return "Signed 8 bits";
    case cheat_type::signed_16_cheat:
      return "Signed 16 bits";
    case cheat_type::signed_32_cheat:
      return "Signed 32 bits";
    case cheat_type::signed_64_cheat:
      return "Signed 64 bits";
    case cheat_type::float_32_cheat:
      return "Float 32 bits";
    case cheat_type::max:
      break;
    }

    return unknown;
  });
}

extern "C" JNIEXPORT jstring JNICALL
Java_net_rpcs3_MainActivity_stringFromJNI(JNIEnv *env, jobject /* this */) {
  Emu.AddGame("test");

  EmuCallbacks cb{
      .call_from_main_thread = [](auto...) {},
      .on_run = [](auto...) {},
      .on_pause = [](auto...) {},
      .on_resume = [](auto...) {},
      .on_stop = [](auto...) {},
      .on_ready = [](auto...) {},
      .on_missing_fw = [](auto...) {},
      .on_emulation_stop_no_response = [](auto...) {},
      .on_save_state_progress = [](auto...) {},
      .enable_disc_eject = [](auto...) {},
      .enable_disc_insert = [](auto...) {},
      .try_to_quit = [](auto...) { return true; },
      .handle_taskbar_progress = [](auto...) {},
      .init_kb_handler = [](auto...) {},
      .init_mouse_handler = [](auto...) {},
      .init_pad_handler = [](auto...) {},
      .update_emu_settings = [](auto...) {},
      .save_emu_settings = [](auto...) {},
      .close_gs_frame = [](auto...) {},
      .get_gs_frame = [](auto...) { return nullptr; },
      .get_camera_handler = [](auto...) { return nullptr; },
      .get_music_handler = [](auto...) { return nullptr; },
      .init_gs_render = [](auto...) {},
      .get_audio = [](auto...) { return nullptr; },
      .get_audio_enumerator = [](auto...) { return nullptr; },
      .get_msg_dialog = [](auto...) { return nullptr; },
      .get_osk_dialog = [](auto...) { return nullptr; },
      .get_save_dialog = [](auto...) { return nullptr; },
      .get_sendmessage_dialog = [](auto...) { return nullptr; },
      .get_recvmessage_dialog = [](auto...) { return nullptr; },
      .get_trophy_notification_dialog = [](auto...) { return nullptr; },
      .get_localized_string = [](auto...) { return nullptr; },
      .get_localized_u32string = [](auto...) { return nullptr; },
      .get_localized_setting = [](auto...) { return nullptr; },
      .play_sound = [](auto...) {},
      .get_image_info = [](auto...) { return false; },
      .get_scaled_image = [](auto...) { return false; },
      .get_font_dirs = [](auto...) { return std::vector<std::string>(); },
      .on_install_pkgs = [](auto...) { return false; },
      .add_breakpoint = [](auto...) {},
  };

  std::string hello = "Hello from C++";
  return env->NewStringUTF(hello.c_str());
}