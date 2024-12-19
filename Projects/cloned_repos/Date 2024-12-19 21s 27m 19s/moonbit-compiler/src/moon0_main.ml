(*
   Copyright (C) 2024 International Digital Economy Academy.
   This program is licensed under the MoonBit Public Source
   License as published by the International Digital Economy Academy,
   either version 1 of the License, or (at your option) any later
   version. This program is distributed in the hope that it will be
   useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MoonBit
   Public Source License for more details. You should have received a
   copy of the MoonBit Public Source License along with this program. If
   not, see
   <https://www.moonbitlang.com/licenses/moonbit-public-source-license-v1>.
*)


module Io = Basic_io
module Config = Basic_config
module Parse = Parsing_parse
module Lst = Basic_lst

let ( |-> ) obj callback =
  callback obj;
  obj

let rec make_directory dir =
  if Sys.file_exists dir then ()
  else (
    make_directory (Filename.dirname dir);
    Sys.mkdir dir 0o777)

let diagnostics = Diagnostics.make ()

let elim_unused_let = Driver_config.Common_Opt.elim_unused_let
and shrink_wasm = Driver_config.Common_Opt.shrink_wasm
and source_map = Driver_config.Common_Opt.source_map
and source_map_url = Driver_config.Common_Opt.source_map_url
and debug_tokens = Driver_config.Common_Opt.debug_tokens
and contification = Driver_config.Common_Opt.contification
and quiet = Driver_config.Common_Opt.quiet
and no_intermediate_file = Driver_config.Common_Opt.no_intermediate_file
and is_main = Driver_config.Common_Opt.is_main

let moonc_executable_name = "moonc"
let bundle_core_usage = "moonc bundle-core [options] <input files>"
let link_core_usage = "moonc link-core [options] <input files>"
let build_packae_usage = "moonc build-package [options] <input files>"
let check_usage = "moonc check [options] <input files>"
let compile_usage = "moonc compile [options] <input file>"
let gen_test_info_usage = "moonc gen-test-info [options] <input files>"
let postprocess_ast = Driver_util.postprocess_ast ~diagnostics
let write_s name sexp = if not !no_intermediate_file then Io.write_s name sexp

let tast_of_ast ~name ~build_context (asts : Parse.output list) :
    Typedtree.output * Global_env.t =
  let write_opt suffix sexp = write_s (name ^ suffix) sexp in
  let export_tast tast =
    write_opt ".typedtree" (Typedtree.sexp_of_output tast)
  in
  let std_import = Driver_util.Std_Path !Basic_config.std_path in
  Driver_util.tast_of_ast ~diagnostics ~build_context ~imports:[] ~std_import
    ~quiet:!quiet
    ~genv_callback:(fun _ -> ())
    ~tast_callback:export_tast ~pkg_config:None ~import_kind:Normal asts

let core_invariant_check ~global_env:_ _ir_name _core = ()

let core_of_tast ~name genv (tast : Typedtree.output) : Core.program =
  let core_callback ~global_env stage prog =
    match stage with
    | `Core_Inline_Single_Use_Join ->
        core_invariant_check ~global_env "core+inline-single-use-join" prog
    | `Core_Remove_Let_Alias ->
        core_invariant_check ~global_env "core+remove-let-alias" prog
    | `Core_Stackalloc ->
        core_invariant_check ~global_env "core+stackalloc" prog
    | `Core_Lambda_Lift ->
        core_invariant_check ~global_env "core+lambda-lift" prog
    | `Core_DCE -> core_invariant_check ~global_env "core+dce" prog
    | `Core_Contify ->
        write_s (name ^ ".core.contified") (Core.sexp_of_program prog);
        core_invariant_check ~global_env "core+contify" prog
    | `Core_Start -> ()
    | `Core_End -> ()
  in
  Driver_util.core_of_tast ~debug:!Config.debug ~contification:!contification
    ~genv ~tast ~core_callback

let wasm_gen ~(name : string) c =
  let optimize_count = ref 0 in
  let s name =
    incr optimize_count;
    Printf.sprintf "%s_%02d" name !optimize_count
  in
  let write_opt suffix sexp = write_s (s name ^ suffix) sexp in
  let clam_callback stage clam =
    match stage with
    | `Clam_Start -> write_s (name ^ ".clam") (Clam.sexp_of_prog clam)
    | `Clam_Unused_Let -> write_opt ".unused_let.clam" (Clam.sexp_of_prog clam)
    | `Clam_End -> write_s (name ^ ".o.clam") (Clam.sexp_of_prog clam)
  in

  let target =
    match !Config.target with
    | Wasm_gc ->
        Driver_util.Wasm_gc { clam_callback; sexp_callback = (fun _ -> ()) }
  in
  Driver_util.wasm_gen ~elim_unused_let:!elim_unused_let c ~target

let source_loader ~pkg file =
  let path = Pkg_path_tbl.resolve_source Loc.pkg_path_tbl ~pkg ~file in
  let content = try Some (Io.load_file path) with _ -> None in
  (path, content)

let wasm_bin_gen ~file ~on_source_map (mod_ : Driver_util.wasm_rep) =
  let m =
    match mod_ with Driver_util.Wat sexp -> Dwarfsm_parse.module_ sexp
  in
  if !Config.debug then
    if !source_map then (
      let source_map_url =
        if !source_map_url = "" then None else Some !source_map_url
      in
      let wasm, source_map =
        Dwarfsm_encode.module_with_source_map ~file:(Filename.basename file)
          ?source_map_url ~source_loader m
      in
      on_source_map source_map;
      wasm)
    else Dwarfsm_encode.module_ ~emit_names:true m
  else Dwarfsm_encode.module_ ~emit_names:false m

let wat_gen (sexp : W.t list) =
  let buf = Buffer.create 1000 in
  (match sexp with
  | [] -> ()
  | s :: ss ->
      Buffer.add_string buf
        (W.to_string
           ~ignores:
             [ "source_name"; "source_pos"; "source_type"; "prologue_end" ]
           s);
      Lst.iter ss (fun s ->
          Buffer.add_string buf "\n";
          Buffer.add_string buf (W.to_string s)));
  Buffer.contents buf

let bundle_core () =
  let output_file = ref "" in
  let inputs = ref [] in
  Arg.parse_argv ~current:(ref 1) Sys.argv
    [ ("-o", Arg.Set_string output_file, "set output file") ]
    (fun input -> inputs := input :: !inputs)
    bundle_core_usage;
  if !output_file = "" then (
    prerr_string "unspecified output file";
    exit 2);
  Core_format.bundle ~inputs:(List.rev !inputs) ~path:!output_file

let link_core () =
  let output_file = Driver_config.Linkcore_Opt.output_file in
  let link_main = Driver_config.Linkcore_Opt.link_main in
  let pkg_config_path = Driver_config.Linkcore_Opt.pkg_config_path in
  let input_files : Driver_util.core_input Basic_vec.t = Basic_vec.empty () in
  let exported_functions = Driver_config.Linkcore_Opt.exported_functions in
  let link_core_spec = Driver_config.Linkcore_Opt.spec in
  Arg.parse_argv ~current:(ref 1) Sys.argv link_core_spec
    (fun input_file ->
      Basic_vec.push input_files (Driver_util.Core_Path input_file))
    link_core_usage;
  if Basic_vec.is_empty input_files && !output_file = "" then (
    Arg.usage link_core_spec link_core_usage;
    exit 0);
  Basic_config.current_package := !link_main;
  (if Sys.file_exists !pkg_config_path then
     let json_data =
       Json_parse.parse_json_from_file ~diagnostics:(Diagnostics.make ())
         ~fname:(Filename.basename !pkg_config_path)
         !pkg_config_path
     in
     Pkg_config_util.link_core_load_pkg_config json_data);
  let on_source_map blob = Io.write (!output_file ^ ".map") blob in
  let sexp_callback sexp =
    match !output_file with
    | "" ->
        prerr_string "unspecified output file";
        exit 2
    | filename ->
        if Filename.check_suffix filename ".wat" then
          Io.write filename (wat_gen sexp)
        else if Filename.check_suffix filename ".wasm" then
          Io.write filename
            (wasm_bin_gen ~file:filename ~on_source_map (Driver_util.Wat sexp))
        else prerr_string (filename ^ ": unrecognized file type")
  in
  let target =
    match !Config.target with
    | Wasm_gc ->
        Driver_util.Wasm_gc { clam_callback = (fun _ _ -> ()); sexp_callback }
  in
  Driver_util.link_core ~shrink_wasm:!shrink_wasm
    ~elim_unused_let:!elim_unused_let ~core_inputs:input_files
    ~exported_functions ~target

let build_package () =
  let input_files = ref [] in
  let mi_files = Driver_config.Buildpkg_Opt.mi_files in
  let output_file = Driver_config.Buildpkg_Opt.output_file in
  let no_mi = Driver_config.Buildpkg_Opt.no_mi in
  let blackbox_test = Driver_config.Buildpkg_Opt.blackbox_test in
  let whitebox_test = Driver_config.Buildpkg_Opt.whitebox_test in
  Compile_env.reset ();
  let build_package_spec = Driver_config.Buildpkg_Opt.spec in
  Arg.parse_argv ~current:(ref 1) Sys.argv build_package_spec
    (fun input_file -> input_files := input_file :: !input_files)
    build_packae_usage;
  if !input_files = [] && !output_file = "" then (
    Arg.usage build_package_spec build_packae_usage;
    exit 0);
  if !output_file = "" then
    output_file := Filename.basename !Basic_config.current_package ^ ".core";
  let output_dir = Filename.dirname !output_file in
  let pkg_name = !Basic_config.current_package in
  make_directory output_dir;
  let std_import = Driver_util.Std_Path !Basic_config.std_path in
  let imports = Lst.map !mi_files (fun imp -> Driver_util.Import_Path imp) in
  let mbt_files =
    Lst.map !input_files (fun path -> Driver_util.File_Path path)
  in
  let profile_callback asts = asts in
  let debug_source_callback _ _ = ()
  in
  let genv_callback genv =
    if (not (Diagnostics.has_fatal_errors diagnostics)) && not !no_mi then
      let mi_path = Filename.remove_extension !output_file ^ ".mi" in
      Global_env.export_mi ~action:(Write_file mi_path) ~pkg_name genv
  in
  let tast_callback _ = () in
  let core_callback ~global_env stage prog =
    match stage with
    | `Core_Contify -> core_invariant_check ~global_env "core+contify" prog
    | `Core_Inline_Single_Use_Join ->
        core_invariant_check ~global_env "core+inline-single-use-join" prog
    | `Core_Remove_Let_Alias ->
        core_invariant_check ~global_env "core+remove-let-alias" prog
    | `Core_Stackalloc ->
        core_invariant_check ~global_env "core+stackalloc" prog
    | `Core_Lambda_Lift ->
        core_invariant_check ~global_env "core+lambda-lift" prog
    | `Core_DCE -> core_invariant_check ~global_env "core+dce" prog
    | `Core_Start -> ()
    | `Core_End ->
        let core_path = Filename.remove_extension !output_file ^ ".core" in
        Core_format.export ~action:(Write_file core_path) ~pkg_name
          ~program:prog ~genv:global_env
  in
  let pkg_config_file =
    match
      Pkg_path_tbl.find_pkg Loc.pkg_path_tbl !Basic_config.current_package
    with
    | Some pkg_src_dir ->
        let path = Filename.concat pkg_src_dir "moon.pkg.json" in
        if Sys.file_exists path then Some path else None
    | None -> None
  in
  let import_kind : Pkg_config_util.import_kind =
    if !blackbox_test then Bbtest else if !whitebox_test then Wbtest else Normal
  in
  Driver_util.build_package ~diagnostics ~std_import ~imports
    ~debug_tokens:!debug_tokens ~debug:!Config.debug
    ~contification:!contification ~is_main:!is_main ~quiet:!quiet
    ~profile_callback ~debug_source_callback ~genv_callback ~tast_callback
    ~core_callback ~pkg_config_file ~import_kind mbt_files
  |> ignore

let check () =
  let input_files = ref [] in
  let mi_files = Driver_config.Check_Opt.mi_files in
  let output_file = Driver_config.Check_Opt.output_file in
  let no_mi = Driver_config.Check_Opt.no_mi in
  let blackbox_test = Driver_config.Check_Opt.blackbox_test in
  let whitebox_test = Driver_config.Check_Opt.whitebox_test in
  let check_spec = Driver_config.Check_Opt.spec in
  Arg.parse_argv ~current:(ref 1) Sys.argv check_spec
    (fun input_file -> input_files := input_file :: !input_files)
    check_usage;
  if !input_files = [] && !output_file = "" then (
    Arg.usage check_spec check_usage;
    exit 0);
  if !output_file = "" then
    output_file := Filename.basename !Basic_config.current_package ^ ".mi";
  let output_dir = Filename.dirname !output_file in
  if not !no_mi then make_directory output_dir;
  let std_import = Driver_util.Std_Path !Basic_config.std_path in
  let imports = Lst.map !mi_files (fun imp -> Driver_util.Import_Path imp) in
  let mbt_files =
    Lst.map !input_files (fun path -> Driver_util.File_Path path)
  in
  let genv_callback genv =
    if not !no_mi then
      Global_env.export_mi ~action:(Write_file !output_file)
        ~pkg_name:!Basic_config.current_package
        genv
  in
  let pkg_config_file =
    match
      Pkg_path_tbl.find_pkg Loc.pkg_path_tbl !Basic_config.current_package
    with
    | Some pkg_src_dir ->
        let path = Filename.concat pkg_src_dir "moon.pkg.json" in
        if Sys.file_exists path then Some path else None
    | None -> None
  in
  let import_kind : Pkg_config_util.import_kind =
    if !blackbox_test then Bbtest else if !whitebox_test then Wbtest else Normal
  in
  Driver_util.check ~diagnostics ~std_import ~imports
    ~debug_tokens:!debug_tokens ~is_main:!is_main ~quiet:!quiet ~genv_callback
    ~tast_callback:(fun _ -> ())
    ~pkg_config_file ~import_kind mbt_files
  |> ignore;
  Diagnostics.check_diagnostics diagnostics

let compile () =
  let src = ref [] in
  let output_file = Driver_config.Compile_Opt.output_file in
  let extra_deps = Driver_config.Compile_Opt.extra_deps in
  let export_functions_info = Driver_config.Compile_Opt.export_functions_info in
  let wat_stats = Driver_config.Compile_Opt.wat_stats in
  let stop_after_parsing = Driver_config.Compile_Opt.stop_after_parsing in
  let stop_after_typing = Driver_config.Compile_Opt.stop_after_typing in
  let spec = Driver_config.Compile_Opt.spec in
  let compile name =
    let exception Exit in
    try
      Compile_env.reset ();
      Diagnostics.reset diagnostics;
      let ast =
        Parse.parse ~diagnostics ~debug_tokens:!debug_tokens ~transform:false
          name ~directive_handler:(fun directive ->
            match Basic_lst.assoc_str directive "build" with
            | None -> ()
            | Some d ->
                let build_flags =
                  Array.of_list
                    (Sys.executable_name :: String.split_on_char ' ' d)
                in
                (try Arg.parse_argv build_flags spec ignore ""
                 with Arg.Bad s -> prerr_string s);
                ())
      in
      let ast =
        postprocess_ast ast |-> fun c ->
        write_s (name ^ ".ast") (Parse.sexp_of_output c)
      in
      let ast = [ ast ] in
      if !stop_after_parsing then (
        Diagnostics.check_diagnostics diagnostics;
        raise_notrace Exit);
      let tast, genv = tast_of_ast ~build_context:SingleFile ~name ast in
      Diagnostics.check_diagnostics diagnostics;
      if !stop_after_typing then raise_notrace Exit;
      let program = core_of_tast ~name genv tast in
      Basic_uuid.reset ();
      let targets = Basic_vec.empty () in
      !extra_deps
      |> List.iter (fun filename ->
             Basic_vec.push targets (Core_link.File_path filename));
      Basic_vec.push targets
        (Core_link.Core_format
           [|
             {
               program;
               types =
                 Global_env.get_toplevel_types genv |> Typing_info.get_all_types;
               traits =
                 Global_env.get_toplevel_types genv
                 |> Typing_info.get_all_traits;
               methods = Global_env.get_method_env genv;
               ext_methods = Global_env.get_ext_method_env genv;
               pkg_name = !Basic_config.current_package;
             };
           |]);
      let link_output = Core_link.link ~targets in
      let exported_functions =
        match !export_functions_info with
        | Some export -> export
        | None -> Exported_functions.Export_all_pubs
      in
      let mono_core =
        Driver_util.monofy_core_link ~link_output ~exported_functions
      in
      write_s (name ^ ".core.mono") (Mcore.sexp_of_t mono_core);
      let on_source_map blob = Io.write (!output_file ^ ".map") blob in
      let postprecess mod_ =
        let mod_ =
          match mod_ with
          | Driver_util.Wat sexp ->
              Driver_util.Wat
                (if !shrink_wasm then Pass_shrink_wasm.shrink sexp else sexp)
        in
        let sexp = match mod_ with Driver_util.Wat sexp -> wat_gen sexp in
        let write_wat_to filename =
          Io.write filename sexp;
          if !wat_stats then
            let bin = wasm_bin_gen ~file:filename ~on_source_map mod_ in
            let binary_size = Int.to_string (String.length bin) in
            let statsfile = filename ^ ".stats" in
            Io.write statsfile
              (filename ^ ": binary size " ^ binary_size ^ " bytes")
        in
        match !output_file with
        | "" ->
            let filename = name ^ ".wat" in
            write_wat_to filename
        | filename ->
            if Filename.check_suffix filename ".wat" then write_wat_to filename
            else if Filename.check_suffix filename ".wasm" then
              Io.write filename
                (wasm_bin_gen ~file:filename ~on_source_map mod_)
            else prerr_string (filename ^ ": unrecognized file type")
      in
      match !Config.target with
      | Wasm_gc ->
          let mod_ = wasm_gen ~name mono_core in
          postprecess mod_
    with Exit -> ()
  in
  Arg.parse_argv ~current:(ref 1) Sys.argv spec
    (fun name -> src := name :: !src)
    compile_usage;
  Basic_lst.rev_iter !src compile

let gen_test_info () =
  let input_files = Basic_vec.make ~dummy:"" 7 in
  let json = Driver_config.Gentestinfo_Opt.json in
  Compile_env.reset ();
  Arg.parse_argv ~current:(ref 1) Sys.argv Driver_config.Gentestinfo_Opt.spec
    (fun input_file -> Basic_vec.push input_files input_file)
    gen_test_info_usage;
  Basic_vec.map_into_list input_files (fun path -> Driver_util.File_Path path)
  |> Driver_util.gen_test_info ~diagnostics ~json:!json
  |> print_string

let subcommands =
  [
    ("bundle-core", "bundle multiple core IR files into one", bundle_core);
    ("link-core", "link core IR files and generate backend code", link_core);
    ("build-package", "build core IR file for a single package", build_package);
    ("check", "check the code of a single package", check);
    ("compile", "compile a single file and generate backend code", compile);
    ("gen-test-info", "generate test block info for a package", gen_test_info);
  ]

let print_help () =
  Printf.printf "Usage: %s subcommand\n" moonc_executable_name;
  Printf.printf "Available subcommands:\n";
  List.iter
    (fun (name, help, _) -> Printf.printf "  %s: %s\n" name help)
    subcommands

let run_main () =
  let argv_len = Array.length Sys.argv in
  if argv_len = 1 then print_help ()
  else
    let subcommand = Sys.argv.(1) in
    if subcommand = "help" then print_help ()
    else
      match
        List.find_opt (fun (name, _, _) -> name = subcommand) subcommands
      with
      | Some (_, _, command) ->
          let before, after =
            Driver_compenv.moonc_internal_params () |> Driver_compenv.parse_args
          in
          Driver_compenv.exec_args before;
          (try command () with
          | Arg.Help s -> print_string s
          | Arg.Bad s ->
              prerr_string s;
              exit 2
          | Diagnostics.Fatal_error -> exit 2);
          Driver_compenv.exec_args after
      | None ->
          Arg.parse
            [
              ( "-v",
                Arg.Unit
                  (fun _ ->
                    print_endline Version.version;
                    exit 0),
                " show version" );
            ]
            (fun name -> raise (Arg.Bad ("Dont'know what to do with " ^ name)))
            moonc_executable_name;
          exit 2

let () = Ice_catcher.run_with_protection (fun () -> run_main ())
