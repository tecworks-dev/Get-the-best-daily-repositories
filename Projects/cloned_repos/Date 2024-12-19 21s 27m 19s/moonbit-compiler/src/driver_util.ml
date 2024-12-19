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


let ( |-> ) obj callback =
  callback obj;
  obj

type core_passes =
  [ `Core_Contify
  | `Core_DCE
  | `Core_End
  | `Core_Inline_Single_Use_Join
  | `Core_Lambda_Lift
  | `Core_Remove_Let_Alias
  | `Core_Stackalloc
  | `Core_Start ]

type clam1_passes =
  [ `Clam1_End
  | `Clam1_RC_Drop_Spec
  | `Clam1_RC_End
  | `Clam1_RC_Insert_Ref
  | `Clam1_Start
  | `Clam1_Unused_Let ]

type clam_passes = [ `Clam_End | `Clam_Start | `Clam_Unused_Let ]
type mbt_input = File_Path of string | Name_Content of string * string
type mi_input = Import_Path of string | Import_Path_Content of string * string
type std_input = Std_Path of string | Std_Content of (string * string) list
type wasm_rep = Wat of W.t list

type target =
  | Wasm_gc of {
      clam_callback : clam_passes -> Clam.prog -> unit;
      sexp_callback : W.t list -> unit;
    }

let parse ~diagnostics ~(debug_tokens : bool) (input : mbt_input) :
    Parsing_parse.output =
  match input with
  | File_Path path ->
      Parsing_parse.parse ~diagnostics ~debug_tokens ~transform:false path
  | Name_Content (name, content) ->
      Parsing_parse.impl_of_string ~name ~debug_tokens ~diagnostics
        ~transform:false content

let postprocess_ast ~diagnostics (output : Parsing_parse.output) :
    Parsing_parse.output =
  { output with ast = Parsing_ast_lint.post_process ~diagnostics output.ast }

let tast_of_ast ~diagnostics ~(build_context : Typeutil.build_context)
    ~(imports : mi_input list) ~(quiet : bool) ?(std_import : std_input option)
    ~(genv_callback : Global_env.t -> unit)
    ~(tast_callback : Typedtree.output -> unit)
    ~(pkg_config : Json_types.t option)
    ~(import_kind : Pkg_config_util.import_kind)
    (asts : Parsing_parse.output list) : Typedtree.output * Global_env.t =
  let asts = Basic_lst.map asts (fun a -> a.ast) in
  let pkgs = Pkg.create_tbl () in
  let import_items =
    Pkg_config_util.parse_import_item ~import_kind pkg_config
  in
  Basic_lst.iter imports (function
    | Import_Path_Content (imp_str, mi_content) ->
        let imp = Parsing_import_path.parse imp_str in
        Pkg.load_mi ~import_items pkgs imp mi_content ~diagnostics
    | Import_Path imp_str ->
        let imp = Parsing_import_path.parse imp_str in
        let mi_content =
          Stdlib.In_channel.with_open_bin imp.path Stdlib.In_channel.input_all
        in
        Pkg.load_mi ~import_items pkgs imp mi_content ~diagnostics);
  (match std_import with
  | Some (Std_Content imports) ->
      Basic_lst.iter imports (fun (imp_str, mi_content) ->
          let imp = Parsing_import_path.parse imp_str in
          Pkg.load_mi ~import_items pkgs imp mi_content ~diagnostics)
  | Some (Std_Path std_path) ->
      if std_path <> "" then Pkg.load_std pkgs ~std_path ~diagnostics
  | None -> ());
  let top_output =
    Toplevel_typer.check_toplevel ~pkgs ~build_context asts ~diagnostics
  in
  let genv = top_output.global_env |-> genv_callback in
  let tast = Typer.type_check top_output ~diagnostics in
  if (not quiet) && not (Diagnostics.has_fatal_errors diagnostics) then
    Check_match.analyze ~diagnostics (genv, tast);
  let tast = Topo_sort.topo_sort tast ~diagnostics |-> tast_callback in
  Global_env.report_unused_pkg ~diagnostics genv;
  Dead_code.analyze_unused ~diagnostics ~import_items tast;
  (tast, genv)

let core_of_tast ~(debug : bool) ~(contification : bool) ~(genv : Global_env.t)
    ~(tast : Typedtree.output) ~core_callback : Core.program =
  let pass ~cond ~stage f prog =
    if cond then prog |> f |-> core_callback ~global_env:genv stage else prog
  in
  let prog = Core_of_tast.transl ~global_env:genv tast in
  prog
  |> pass ~cond:(not debug) ~stage:`Core_Inline_Single_Use_Join
       Pass_inline_single_use_join.inline_single_use_join
  |-> core_callback ~global_env:genv `Core_Start
  |> pass ~cond:contification ~stage:`Core_Contify Pass_contification.contify
  |> pass ~cond:(not debug) ~stage:`Core_Remove_Let_Alias
       Pass_let_alias.remove_let_alias
  |> pass ~cond:(not debug) ~stage:`Core_Stackalloc
       Pass_stackalloc.unbox_mut_records
  |> pass ~cond:(not debug) ~stage:`Core_Lambda_Lift Lambda_lift.lift_program
  |> pass ~cond:(not debug) ~stage:`Core_DCE Core_dce.eliminate_dead_code
  |-> core_callback ~global_env:genv `Core_End

let check ~diagnostics ?(std_import : std_input option)
    ~(imports : mi_input list) ~(debug_tokens : bool) ~(is_main : bool)
    ~(quiet : bool) ~(genv_callback : Global_env.t -> unit)
    ~(tast_callback : Typedtree.output -> unit)
    ~(pkg_config_file : string option)
    ~(import_kind : Pkg_config_util.import_kind) (mbt_files : mbt_input list) :
    Typedtree.output * Global_env.t =
  let outputs =
    mbt_files
    |> List.map (fun input ->
           let ast = parse ~diagnostics ~debug_tokens input in
           postprocess_ast ~diagnostics ast)
  in
  let pkg_config =
    match pkg_config_file with
    | Some path ->
        Some
          (Json_parse.parse_json_from_file ~diagnostics
             ~fname:(Filename.basename path) path)
    | None -> None
  in
  Pkg_config_util.parse_warn_alert pkg_config;
  let build_context : Typeutil.build_context =
    match Pkg_config_util.parse_is_main pkg_config with
    | Is_main loc -> Exec { is_main_loc = loc }
    | Not_main ->
        if is_main then Exec { is_main_loc = Loc.no_location } else Lib
  in
  tast_of_ast ~diagnostics ~build_context ~imports ~quiet ?std_import
    ~genv_callback ~tast_callback ~pkg_config ~import_kind outputs

let build_package ~diagnostics ?(std_import : std_input option)
    ~(imports : mi_input list) ~(debug_tokens : bool) ~(debug : bool)
    ~(contification : bool) ~(is_main : bool) ~(quiet : bool)
    ~(profile_callback : Parsing_parse.output list -> Parsing_parse.output list)
    ~(debug_source_callback : string -> Parsing_parse.output -> unit)
    ~(genv_callback : Global_env.t -> unit)
    ~(tast_callback : Typedtree.output -> unit) ~core_callback
    ~(pkg_config_file : string option)
    ~(import_kind : Pkg_config_util.import_kind) (mbt_files : mbt_input list) :
    Core.program =
  let asts =
    mbt_files
    |> List.map (parse ~diagnostics ~debug_tokens)
    |> profile_callback
    |> List.map (postprocess_ast ~diagnostics)
  in
  let pkg_config =
    match pkg_config_file with
    | Some path ->
        Some
          (Json_parse.parse_json_from_file ~diagnostics
             ~fname:(Filename.basename path) path)
    | None -> None
  in
  Pkg_config_util.parse_warn_alert pkg_config;
  let build_context : Typeutil.build_context =
    match Pkg_config_util.parse_is_main pkg_config with
    | Is_main loc -> Exec { is_main_loc = loc }
    | Not_main ->
        if is_main then Exec { is_main_loc = Loc.no_location } else Lib
  in
  Basic_lst.iter2 mbt_files asts (fun mbt_file ast ->
      match mbt_file with
      | Name_Content _ -> ()
      | File_Path path -> debug_source_callback path ast);
  ( asts
  |> tast_of_ast ~diagnostics ~build_context ~imports ~quiet ?std_import
       ~genv_callback ~tast_callback ~pkg_config ~import_kind
  |-> fun _ -> Diagnostics.check_diagnostics diagnostics )
  |> fun (tast, genv) ->
  core_of_tast ~debug ~contification ~genv ~tast ~core_callback

type core_input = Core_Path of string | Core_Content of string

let monofy_core_link ~(link_output : Core_link.output) ~exported_functions :
    Mcore.t =
  let monofy_env =
    Monofy_env.make ~regular_methods:link_output.methods
      ~extension_methods:link_output.ext_methods
  in
  let mono_core =
    Monofy.monofy ~monofy_env ~stype_defs:link_output.types ~exported_functions
      link_output.linked_program
  in
  let mono_core = Pass_layout.optimize_layout mono_core in
  mono_core

let clam1_of_mcore ~(elim_unused_let : bool) (core : Mcore.t) ~clam1_callback :
    Clam1.prog =
  let pass ~cond ~stage f prog =
    if cond then prog |> f |-> clam1_callback stage else prog
  in
  core |> Clam1_of_core.transl_prog
  |-> clam1_callback `Clam1_Start
  |> pass ~cond:elim_unused_let ~stage:`Clam1_Unused_Let
       Pass_unused_let1.unused_let_opt
  |-> clam1_callback `Clam1_End

let clam_of_mcore ~(elim_unused_let : bool) (core : Mcore.t) ~clam_callback :
    Clam.prog =
  let pass ~cond ~stage f prog =
    if cond then prog |> f |-> clam_callback stage else prog
  in
  core |> Clam_of_core.transl_prog |-> clam_callback `Clam_Start
  |> pass ~cond:elim_unused_let ~stage:`Clam_Unused_Let
       Pass_unused_let.unused_let_opt
  |-> clam_callback `Clam_End

let wasm_gen ~(elim_unused_let : bool) (core : Mcore.t) ~(target : target) =
  match target with
  | Wasm_gc { clam_callback; _ } ->
      core
      |> clam_of_mcore ~elim_unused_let ~clam_callback
      |> Wasm_of_clam_gc.compile
      |> fun sexp -> Wat sexp

let link_core ~(shrink_wasm : bool) ~(elim_unused_let : bool)
    ~(core_inputs : core_input Basic_vec.t)
    ~(exported_functions : string Basic_hash_string.t) ~(target : target) : unit
    =
  let targets : Core_link.linking_target Basic_vec.t = Basic_vec.empty () in
  Basic_vec.iter core_inputs (function
    | Core_Path path -> Basic_vec.push targets (Core_link.File_path path)
    | Core_Content content ->
        Basic_vec.push targets
          (Core_link.Core_format (Core_format.of_string content)));
  let link_output = Core_link.link ~targets in
  let mono_core =
    monofy_core_link ~link_output
      ~exported_functions:
        (Exported_functions.Export_selected exported_functions)
  in
  match target with
  | Wasm_gc { sexp_callback; _ } -> (
      let mod_and_callback = mono_core |> wasm_gen ~elim_unused_let ~target in
      match mod_and_callback with
      | Wat sexp ->
          (if shrink_wasm then Pass_shrink_wasm.shrink sexp else sexp)
          |> sexp_callback)

let gen_test_info ~(diagnostics : Diagnostics.t) ~(json : bool)
    (mbt_files : mbt_input list) : string =
  let parse_and_patch mbt_file =
    let ast = parse ~diagnostics ~debug_tokens:false mbt_file in
    ast
  in
  let inputs =
    Basic_lst.map mbt_files (fun mbt_file ->
        match mbt_file with
        | File_Path path -> (path, (parse_and_patch mbt_file).ast)
        | Name_Content (name, _) -> (name, (parse_and_patch mbt_file).ast))
  in
  if json then Gen_test_info.gen_test_info_json inputs
  else Gen_test_info.gen_test_info inputs
