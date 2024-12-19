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


module Import_path = Parsing_import_path
module Longident = Basic_longident
module H = Basic_hash_string
module Lst = Basic_lst
module Type_path = Basic_type_path
module Arr = Basic_arr
module Hash_string = Basic_hash_string

type pkg_tbl_entry = Pkg_info.mi_view option lazy_t

include struct
  let _ = fun (_ : pkg_tbl_entry) -> ()

  let sexp_of_pkg_tbl_entry =
    (fun x__001_ ->
       Moon_sexp_conv.sexp_of_lazy_t
         (Moon_sexp_conv.sexp_of_option Pkg_info.sexp_of_mi_view)
         x__001_
      : pkg_tbl_entry -> S.t)

  let _ = sexp_of_pkg_tbl_entry
end

type direct_uses = Direct_use_all | Values of (string * Loc.t) list

include struct
  let _ = fun (_ : direct_uses) -> ()

  let sexp_of_direct_uses =
    (function
     | Direct_use_all -> S.Atom "Direct_use_all"
     | Values arg0__006_ ->
         let res0__007_ =
           Moon_sexp_conv.sexp_of_list
             (fun (arg0__002_, arg1__003_) ->
               let res0__004_ = Moon_sexp_conv.sexp_of_string arg0__002_
               and res1__005_ = Loc.sexp_of_t arg1__003_ in
               S.List [ res0__004_; res1__005_ ])
             arg0__006_
         in
         S.List [ S.Atom "Values"; res0__007_ ]
      : direct_uses -> S.t)

  let _ = sexp_of_direct_uses
end

type pkg_info = {
  mi_info : pkg_tbl_entry;
  usage : bool ref;
  alias_to : string option;
  direct_uses : direct_uses;
  loc : Loc.t;
}

include struct
  let _ = fun (_ : pkg_info) -> ()

  let sexp_of_pkg_info =
    (fun {
           mi_info = mi_info__009_;
           usage = usage__011_;
           alias_to = alias_to__013_;
           direct_uses = direct_uses__015_;
           loc = loc__017_;
         } ->
       let bnds__008_ = ([] : _ Stdlib.List.t) in
       let bnds__008_ =
         let arg__018_ = Loc.sexp_of_t loc__017_ in
         (S.List [ S.Atom "loc"; arg__018_ ] :: bnds__008_ : _ Stdlib.List.t)
       in
       let bnds__008_ =
         let arg__016_ = sexp_of_direct_uses direct_uses__015_ in
         (S.List [ S.Atom "direct_uses"; arg__016_ ] :: bnds__008_
           : _ Stdlib.List.t)
       in
       let bnds__008_ =
         let arg__014_ =
           Moon_sexp_conv.sexp_of_option Moon_sexp_conv.sexp_of_string
             alias_to__013_
         in
         (S.List [ S.Atom "alias_to"; arg__014_ ] :: bnds__008_
           : _ Stdlib.List.t)
       in
       let bnds__008_ =
         let arg__012_ =
           Moon_sexp_conv.sexp_of_ref Moon_sexp_conv.sexp_of_bool usage__011_
         in
         (S.List [ S.Atom "usage"; arg__012_ ] :: bnds__008_ : _ Stdlib.List.t)
       in
       let bnds__008_ =
         let arg__010_ = sexp_of_pkg_tbl_entry mi_info__009_ in
         (S.List [ S.Atom "mi_info"; arg__010_ ] :: bnds__008_
           : _ Stdlib.List.t)
       in
       S.List bnds__008_
      : pkg_info -> S.t)

  let _ = sexp_of_pkg_info
end

type pkg_tbl = pkg_info H.t

include struct
  let _ = fun (_ : pkg_tbl) -> ()

  let sexp_of_pkg_tbl =
    (fun x__019_ -> H.sexp_of_t sexp_of_pkg_info x__019_ : pkg_tbl -> S.t)

  let _ = sexp_of_pkg_tbl
end

let create_tbl () = H.create 17

let add_pkg_entry ~(import_items : Pkg_config_util.import_items) (tbl : pkg_tbl)
    (imp : Import_path.t) name (entry : pkg_tbl_entry) ~diagnostics =
  let get_default_alias name =
    match imp.alias with
    | Some alias -> alias
    | None -> Import_path.default_abbrev name
  in
  let noloc = Loc.no_location_with_pkg !Basic_config.current_package in
  let add_default_alias () =
    let alias = get_default_alias name in
    let alias_item =
      {
        mi_info = entry;
        usage = ref true;
        alias_to = Some name;
        loc = noloc;
        direct_uses = Values [];
      }
    in
    if not (H.mem tbl alias) then H.add tbl alias alias_item
      [@@inline]
  in
  match Basic_map_string.find_opt import_items name with
  | Some info -> (
      let usage =
        ref ((not info.need_analyze_usage) || info.direct_uses <> [])
      in
      let item =
        {
          mi_info = entry;
          usage;
          alias_to = None;
          loc = info.pkg_loc;
          direct_uses = Values info.direct_uses;
        }
      in
      H.add tbl name item;
      match info.alias_info with
      | Some (alias, loc) when alias <> "" ->
          let usage = ref (not info.need_analyze_usage) in
          let alias_item =
            {
              mi_info = entry;
              usage;
              alias_to = Some name;
              loc;
              direct_uses = Values [];
            }
          in
          if not (H.mem tbl alias) then H.add tbl alias alias_item
      | Some (_, loc) ->
          Diagnostics.add_warning diagnostics
            { loc; kind = Empty_package_alias };
          add_default_alias ()
      | None -> add_default_alias ())
  | None ->
      let direct_uses =
        if
          !Basic_config.blackbox_test_import_all
          && !Basic_config.current_package = name ^ "_blackbox_test"
        then Direct_use_all
        else Values []
      in
      H.add tbl name
        {
          mi_info = entry;
          usage = ref true;
          alias_to = None;
          loc = noloc;
          direct_uses;
        };
      add_default_alias ()

let load_pkg_info ~(import_items : Pkg_config_util.import_items) tbl imp
    (pkg_info : Pkg_info.mi_view) ~diagnostics =
  add_pkg_entry ~import_items tbl imp pkg_info.name
    (Lazy.from_val (Some pkg_info))
    ~diagnostics

let mark_pkg_as_used tbl pkg =
  match H.find_opt tbl pkg with
  | Some { usage; alias_to; _ } -> (
      usage := true;
      match alias_to with
      | Some alias -> (
          match H.find_opt tbl alias with
          | Some { usage; _ } -> usage := true
          | None -> ())
      | None -> ())
  | None -> ()

let load_mi ~(import_items : Pkg_config_util.import_items) tbl
    (imp : Import_path.t) mi ~diagnostics =
  match Mi_format.input_mi ~path:imp.path mi with
  | Error e -> Diagnostics.add_error diagnostics e
  | Ok info -> load_pkg_info ~import_items tbl imp info ~diagnostics

let find_pkg_opt (tbl : pkg_tbl) ~pkg =
  mark_pkg_as_used tbl pkg;
  match H.find_opt tbl pkg with
  | Some { mi_info = (lazy (Some p)); _ } -> Some p
  | Some { mi_info = (lazy None); _ } | None -> None

let lazy_load_mi tbl (imp : Import_path.t) ~full_name ~(mi : unit -> _)
    ~diagnostics =
  add_pkg_entry ~import_items:Basic_map_string.empty ~diagnostics tbl imp
    full_name
    (lazy
      (match Mi_format.input_mi ~path:imp.path (mi ()) with
      | Ok pkg_info -> Some pkg_info
      | Error err ->
          Diagnostics.add_error diagnostics err;
          None))

let pkg_path_concat pkg_path filename =
  let l = String.length pkg_path in
  if l = 0 || pkg_path.[l - 1] = '/' then pkg_path ^ filename
  else pkg_path ^ "/" ^ filename

let load_std pkgs ~std_path ~diagnostics =
  let load path =
    Stdlib.In_channel.with_open_bin path (fun ic ->
        Stdlib.In_channel.input_all ic)
  in
  let rec load_dir ~pkg_path dir =
    let files = Sys.readdir dir in
    Basic_arr.iter files (fun file_name ->
        let path = Filename.concat dir file_name in
        if Sys.is_directory path then
          load_dir ~pkg_path:(pkg_path_concat pkg_path file_name) path
        else if String.ends_with ~suffix:".mi" file_name then
          let full_name = pkg_path_concat "moonbitlang/core" pkg_path in
          let imp : Import_path.t = { path; alias = Some pkg_path } in
          lazy_load_mi pkgs imp ~full_name
            ~mi:(fun () -> load path)
            ~diagnostics)
  in
  load_dir ~pkg_path:"" std_path

type find_value_result =
  | Not_exist
  | Found_value of Value_info.toplevel
  | Found_method of (Type_path.t * Method_env.method_info) list

let find_value_in_mi_view ~allow_method (mi_view : Pkg_info.mi_view) id :
    find_value_result =
  let p = mi_view in
  match
    Arr.find_map p.values (fun ((x : string), vd) ->
        if x = id then Some vd else None)
  with
  | Some vd -> Found_value vd
  | None ->
      if allow_method then
        Found_method
          (Method_env.find_methods_by_name p.method_env ~method_name:id)
      else Not_exist

let find_value ~allow_method (tbl : pkg_tbl) ({ pkg; id } : Longident.qual_name)
    ~loc : Value_info.t Local_diagnostics.info =
  match find_pkg_opt tbl ~pkg with
  | Some p -> (
      match find_value_in_mi_view ~allow_method p id with
      | Found_value vd -> Ok (Toplevel_value vd)
      | Found_method ms -> (
          match ms with
          | [] -> Error (Errors.unbound_value ~name:(Ldot { pkg; id }) ~loc)
          | (_, m) :: [] -> Ok (Method_env.to_value_info m)
          | ms ->
              let type_locs =
                Lst.map ms (fun (type_name, m) -> (type_name, m.loc))
              in
              Error (Errors.ambiguous_method ~name:id ~type_locs ~loc))
      | Not_exist -> Error (Errors.unbound_value ~name:(Ldot { pkg; id }) ~loc))
  | None -> Error (Errors.pkg_not_loaded ~pkg ~loc)

let find_constructor_or_constant (tbl : pkg_tbl) ~pkg constr_name ~loc =
  match find_pkg_opt tbl ~pkg with
  | Some p -> (
      match H.find_opt p.external_constrs constr_name with
      | Some (constr :: []) -> Ok (`Constr constr)
      | Some (c1 :: c2 :: _) ->
          let first_ty =
            c1.cs_res |> Stype.extract_tpath_exn |> Type_path_util.name
          in
          let second_ty =
            c2.cs_res |> Stype.extract_tpath_exn |> Type_path_util.name
          in
          Error
            (Errors.ambiguous_constructor ~name:constr_name ~first_ty ~second_ty
               ~loc)
      | None | Some [] -> (
          match find_value_in_mi_view ~allow_method:false p constr_name with
          | Found_value c -> Ok (`Constant c)
          | Found_method _ -> assert false
          | Not_exist ->
              Error (Errors.constr_not_found ~ty:None ~constr:constr_name ~loc))
      )
  | None -> Error (Errors.pkg_not_loaded ~pkg ~loc)

let find_method_opt (tbl : pkg_tbl) ~pkg ~type_name ~method_name :
    Method_env.method_table_entry option =
  match find_pkg_opt tbl ~pkg with
  | Some p -> Method_env.find_method_opt p.method_env ~type_name ~method_name
  | None -> None

let find_regular_method (tbl : pkg_tbl) ~pkg ~type_name ~method_name :
    Method_env.method_info option =
  match find_pkg_opt tbl ~pkg with
  | Some p ->
      Method_env.find_regular_method p.method_env ~type_name ~method_name
  | None -> None

let find_methods_by_name (tbl : pkg_tbl) ~pkg ~method_name :
    (Type_path.t * Method_env.method_info) list =
  match find_pkg_opt tbl ~pkg with
  | Some p -> Method_env.find_methods_by_name p.method_env ~method_name
  | None -> []

let iter_methods_by_type (tbl : pkg_tbl) ~pkg ~type_name f =
  match find_pkg_opt tbl ~pkg with
  | Some p -> Method_env.iter_methods_by_type p.method_env ~type_name f
  | None -> ()

let find_ext_method_opt (tbl : pkg_tbl) ~pkg ~trait ~self_type ~method_name :
    Ext_method_env.method_info option =
  match find_pkg_opt tbl ~pkg with
  | None -> None
  | Some p ->
      Ext_method_env.find_method p.ext_method_env ~trait ~self_type ~method_name

let find_trait_impl (tbl : pkg_tbl) ~pkg ~trait ~type_name =
  match find_pkg_opt tbl ~pkg with
  | None -> None
  | Some p -> Trait_impl.find_impl p.trait_impls ~trait ~type_name

let find_type_alias (tbl : pkg_tbl) ({ pkg; id } : Longident.qual_name) =
  match find_pkg_opt tbl ~pkg with
  | None -> None
  | Some p -> Hash_string.find_opt p.external_type_alias id

let find_type (tbl : pkg_tbl) ({ pkg; id } : Longident.qual_name) ~loc :
    Typedecl_info.t Local_diagnostics.info =
  match find_pkg_opt tbl ~pkg with
  | Some p -> (
      match
        Array.find_map
          (fun (name, td) -> if id = name then Some td else None)
          p.external_types
      with
      | Some td -> Ok td
      | None -> Error (Errors.unbound_type ~name:(Ldot { pkg; id }) ~loc))
  | None -> Error (Errors.pkg_not_loaded ~pkg ~loc)

let find_trait (tbl : pkg_tbl) ({ pkg; id } : Longident.qual_name) ~loc :
    Trait_decl.t Local_diagnostics.info =
  match find_pkg_opt tbl ~pkg with
  | Some p -> (
      match
        Array.find_map
          (fun (name, trait) -> if id = name then Some trait else None)
          p.external_traits
      with
      | Some trait -> Ok trait
      | None -> Error (Errors.unbound_trait ~name:(Ldot { pkg; id }) ~loc))
  | None -> Error (Errors.pkg_not_loaded ~pkg ~loc)

let find_type_or_trait (tbl : pkg_tbl) ~pkg id ~loc =
  match find_pkg_opt tbl ~pkg with
  | Some p -> (
      match
        Array.find_map
          (fun (name, td) -> if id = name then Some td else None)
          p.external_types
      with
      | Some td -> `Type td
      | None -> (
          match
            Array.find_map
              (fun (name, trait) -> if id = name then Some trait else None)
              p.external_traits
          with
          | Some trait -> `Trait trait
          | None ->
              `Error
                (Errors.unbound_type_or_trait ~name:(Ldot { pkg; id }) ~loc)))
  | None -> `Error (Errors.pkg_not_loaded ~pkg ~loc)

let find_type_opt (tbl : pkg_tbl) ~pkg id =
  match find_pkg_opt tbl ~pkg with
  | Some p ->
      let type_ =
        Array.find_map
          (fun (name, td) -> if id = name then Some td else None)
          p.external_types
      in
      type_
  | None -> None

let iter (tbl : pkg_tbl) f =
  H.iter tbl (fun (name, { mi_info = (lazy pkg_info); _ }) ->
      match pkg_info with Some pkg_info -> f (name, pkg_info) | None -> ())

let load_direct_uses (tbl : pkg_tbl) (values : Typing_info.values)
    (type_alias : Typedecl_info.alias Hash_string.t)
    ~(diagnostics : Diagnostics.t) =
  let use_all_mi = Basic_vec.empty () in
  H.iter tbl (fun (pkg, { mi_info; direct_uses; _ }) ->
      match direct_uses with
      | Direct_use_all -> (
          match mi_info with
          | (lazy None) -> ()
          | (lazy (Some mi_view)) -> Basic_vec.push use_all_mi mi_view)
      | Values [] -> ()
      | Values direct_uses -> (
          match mi_info with
          | (lazy None) -> ()
          | (lazy (Some mi_view)) ->
              Lst.iter direct_uses (fun (id, loc) ->
                  match Typing_info.find_value values id with
                  | Some vd -> (
                      match[@warning "-fragile-match"] vd.direct_use_loc_ with
                      | Explicit_import loc1 ->
                          Diagnostics.add_error diagnostics
                            (Errors.direct_use_redeclare ~name:id
                               ~first_loc:loc1 ~second_loc:loc)
                      | _ -> assert false)
                  | None -> (
                      let not_found () =
                        Diagnostics.add_error diagnostics
                          (Errors.direct_use_not_found ~pkg ~id ~loc)
                      in
                      match
                        find_value_in_mi_view ~allow_method:true mi_view id
                      with
                      | Found_value vd ->
                          Typing_info.add_value values
                            { vd with direct_use_loc_ = Explicit_import loc }
                      | Found_method ms -> (
                          match ms with
                          | [] -> not_found ()
                          | (_, m) :: [] -> (
                              match[@warning "-fragile-match"]
                                Method_env.to_value_info m
                              with
                              | Toplevel_value v ->
                                  Typing_info.add_value values
                                    {
                                      v with
                                      direct_use_loc_ = Explicit_import loc;
                                    }
                              | _ -> assert false)
                          | ms ->
                              let type_locs =
                                Lst.map ms (fun (type_name, m) ->
                                    (type_name, m.loc))
                              in
                              Diagnostics.add_error diagnostics
                                (Errors.direct_use_ambiguous_method ~name:id
                                   ~type_locs ~pkg ~loc))
                      | Not_exist -> not_found ()))));
  Basic_vec.iter use_all_mi (fun mi_view ->
      Array.iter
        (fun (id, (vd : Value_info.toplevel)) ->
          match Typing_info.find_value values id with
          | Some _ -> ()
          | None ->
              Typing_info.add_value values
                { vd with direct_use_loc_ = Implicit_import_all vd.loc_ })
        mi_view.values;
      Array.iter
        (fun (id, (td : Typedecl_info.t)) ->
          let alias : Typedecl_info.alias =
            {
              name = id;
              arity = td.ty_arity;
              ty_params = td.ty_params_;
              alias =
                T_constr
                  {
                    type_constructor = td.ty_constr;
                    tys = Tvar_env.to_list_map td.ty_params_ (fun p -> p.typ);
                    generic_ = not (Tvar_env.is_empty td.ty_params_);
                    only_tag_enum_ = td.ty_is_only_tag_enum_;
                    is_suberror_ = td.ty_is_suberror_;
                  };
              is_pub = false;
              doc_ = Docstring.empty;
              loc_ = td.ty_loc_;
            }
          in
          Hash_string.add type_alias id alias)
        mi_view.external_types;
      Array.iter
        (fun (id, (trait : Trait_decl.t)) ->
          let alias : Typedecl_info.alias =
            {
              name = id;
              arity = 0;
              ty_params = Tvar_env.empty;
              alias = T_trait trait.name;
              is_pub = false;
              doc_ = Docstring.empty;
              loc_ = trait.loc_;
            }
          in
          Hash_string.add type_alias id alias)
        mi_view.external_traits;
      Hash_string.iter mi_view.external_type_alias (fun (id, alias) ->
          Hash_string.add type_alias id alias))

let report_unused ~diagnostics (tbl : pkg_tbl) : unit =
  let is_core pkg =
    String.starts_with ~prefix:"moonbitlang/core/" pkg
      [@@inline]
  in
  H.iter2 tbl (fun name info ->
      if
        Lazy.is_val info.mi_info && (not !(info.usage))
        && (is_core !Basic_config.current_package || not (is_core name))
        && not (String.starts_with ~prefix:"_" name)
      then
        let report w loc =
          Diagnostics.add_warning diagnostics { loc; kind = w }
        in
        match info.alias_to with
        | None ->
            if
              not
                (name = "moonbitlang/core/coverage"
                || name = "moonbitlang/core/builtin")
            then report (Unused_package { name; is_alias = false }) info.loc
        | Some _ -> report (Unused_package { name; is_alias = true }) info.loc)

let find_regular_value (tbl : pkg_tbl) ~(pkg : string) (id : string) =
  match find_pkg_opt tbl ~pkg with
  | None -> None
  | Some p ->
      Array.find_map
        (fun ((x : string), vd) : Value_info.t option ->
          if x = id then Some (Toplevel_value vd) else None)
        p.values
