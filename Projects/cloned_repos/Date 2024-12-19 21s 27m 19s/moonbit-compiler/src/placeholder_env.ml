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


module Hash_string = Basic_hash_string
module Hashset_string = Basic_hashset_string
module Type_path = Basic_type_path
module Lst = Basic_lst
module Vec = Basic_vec
module Syntax = Parsing_syntax

type trait_info = {
  decl : Syntax.trait_decl;
  closure : Type_path.t list;
  object_safety_status : Trait_decl.object_safety_status;
}

type t = {
  types : Syntax.type_decl Hash_string.t;
  traits : trait_info Hash_string.t;
  newtype_deps : string list Hash_string.t;
  type_alias_deps : string list Hash_string.t;
}

let find_type_opt (env : t) (name : string) =
  Hash_string.find_opt env.types name

let find_trait_opt (env : t) (name : string) =
  Hash_string.find_opt env.traits name

let find_type_alias_deps (env : t) (name : string) =
  Hash_string.find_opt env.type_alias_deps name

let make ~foreign_types ~type_defs ~trait_defs =
  let newtype_deps = Hash_string.create 17 in
  let type_alias_deps = Hash_string.create 17 in
  Hash_string.iter type_defs (fun (name, (decl : Syntax.type_decl)) ->
      let collect_use (exclude : Hashset_string.t) (typ : Syntax.typ) =
        let result = ref [] in
        let rec go (typ : Syntax.typ) =
          match typ with
          | Ptype_any _ -> ()
          | Ptype_arrow { ty_arg; ty_res; ty_err } -> (
              Lst.iter ty_arg go;
              go ty_res;
              match ty_err with
              | Error_typ { ty = ty_err } -> go ty_err
              | No_error_typ | Default_error_typ _ -> ())
          | Ptype_tuple { tys } -> Lst.iter tys go
          | Ptype_name { constr_id; tys } -> (
              Lst.iter tys go;
              match constr_id.lid with
              | Lident name ->
                  if not (Hashset_string.mem exclude name) then
                    result := name :: !result
              | Ldot _ -> ())
          | Ptype_option { ty; _ } -> go ty
        in
        go typ;
        !result
      in
      match decl.components with
      | Ptd_newtype typ ->
          let exclude = Hashset_string.create 17 in
          Lst.iter decl.params (fun tvar ->
              match tvar.tvar_name with
              | Some tvar_name -> Hashset_string.add exclude tvar_name
              | None -> ());
          Hash_string.add newtype_deps name (collect_use exclude typ)
      | Ptd_alias typ ->
          let exclude = Hashset_string.create 17 in
          Lst.iter decl.params (fun tvar ->
              match tvar.tvar_name with
              | Some tvar_name -> Hashset_string.add exclude tvar_name
              | None -> ());
          Hash_string.add type_alias_deps name (collect_use exclude typ)
      | Ptd_abstract | Ptd_variant _ | Ptd_record _ | Ptd_error _ -> ());
  let object_safety_of_traits =
    Hash_string.create (Hash_string.length trait_defs)
  in
  Hash_string.iter trait_defs (fun (name, decl) ->
      Hash_string.add object_safety_of_traits name
        (Trait_decl.get_methods_object_safety decl));
  let traits = Hash_string.create (Hash_string.length trait_defs) in
  Hash_string.iter trait_defs (fun (name, decl) ->
      let visited = Type_path.Hashset.create 17 in
      let closure = Vec.empty () in
      let not_object_safe_supers = Vec.empty () in
      let name0 = name in
      let rec add_trait name =
        match Hash_string.find_opt trait_defs name with
        | Some decl ->
            let path =
              Type_path.toplevel_type ~pkg:!Basic_config.current_package name
            in
            if not (Type_path.Hashset.mem visited path) then (
              Type_path.Hashset.add visited path;
              Vec.push closure path;
              if
                Hash_string.find_exn object_safety_of_traits name <> []
                && name <> name0
              then Vec.push not_object_safe_supers path;
              Lst.iter decl.trait_supers (fun super ->
                  match super.tvc_trait with
                  | Lident name -> add_trait name
                  | Ldot _ as id -> add_foreign_trait id))
        | None -> add_foreign_trait (Lident name)
      and add_foreign_trait (name : Basic_longident.t) =
        match
          Global_env.All_types.find_trait foreign_types name
            ~loc:Rloc.no_location
        with
        | Error _ -> ()
        | Ok decl ->
            Lst.iter decl.closure (fun path ->
                if not (Type_path.Hashset.mem visited path) then (
                  Type_path.Hashset.add visited path;
                  Vec.push closure path;
                  if decl.object_safety_ <> [] then
                    Vec.push not_object_safe_supers path))
      in
      add_trait name;
      let object_safety_status =
        Lst.append
          (Hash_string.find_exn object_safety_of_traits name)
          (Vec.map_into_list not_object_safe_supers (fun super ->
               Trait_decl.Bad_super_trait super))
      in
      let info =
        { decl; object_safety_status; closure = Vec.to_list closure }
      in
      Hash_string.add traits name info);
  { types = type_defs; traits; newtype_deps; type_alias_deps }

let types_to_list_map (type a) (env : t) (f : string -> Syntax.type_decl -> a) :
    a list =
  Hash_string.to_list_with env.types f

let iter_types (env : t) (f : string -> Syntax.type_decl -> unit) =
  Hash_string.iter2 env.types f

let iter_traits (env : t) (f : string -> trait_info -> unit) : unit =
  Hash_string.iter2 env.traits f

let traits_to_list_map (type a) (env : t) (f : string -> trait_info -> a) :
    a list =
  Hash_string.to_list_with env.traits f

let newtype_in_cycle (env : t) (name : string) : bool =
  let visiting = Hashset_string.create 17 in
  let rec go (name : string) =
    if Hashset_string.mem visiting name then true
    else
      match Hash_string.find_opt env.newtype_deps name with
      | None -> false
      | Some deps ->
          Hashset_string.add visiting name;
          let result = Lst.exists deps (fun dep -> go dep) in
          Hashset_string.remove visiting name;
          result
  in
  go name
