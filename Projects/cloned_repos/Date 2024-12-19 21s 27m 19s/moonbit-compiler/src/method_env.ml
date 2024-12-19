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


module Type_path = Basic_type_path
module M = Basic_hash_string
module Vec = Basic_vec
module Arr = Basic_arr
module Lst = Basic_lst

module H = Basic_hashf.Make (struct
  type t = Type_path.t

  include struct
    let _ = fun (_ : t) -> ()
    let equal = (Type_path.equal : t -> t -> bool)
    let _ = equal

    let (hash_fold_t : Ppx_base.state -> t -> Ppx_base.state) =
      Type_path.hash_fold_t

    and (hash : t -> Ppx_base.hash_value) =
      let func = Type_path.hash in
      fun x -> func x

    let _ = hash_fold_t
    and _ = hash

    let sexp_of_t = (Type_path.sexp_of_t : t -> S.t)
    let _ = sexp_of_t
  end
end)

type method_name = string

include struct
  let _ = fun (_ : method_name) -> ()
  let sexp_of_method_name = (Moon_sexp_conv.sexp_of_string : method_name -> S.t)
  let _ = sexp_of_method_name
end

type type_name = Type_path.t

include struct
  let _ = fun (_ : type_name) -> ()
  let sexp_of_type_name = (Type_path.sexp_of_t : type_name -> S.t)
  let _ = sexp_of_type_name
end

type docstring = Docstring.t

include struct
  let _ = fun (_ : docstring) -> ()
  let sexp_of_docstring = (Docstring.sexp_of_t : docstring -> S.t)
  let _ = sexp_of_docstring
end

type method_info = {
  id : Basic_qual_ident.t;
  prim : Primitive.prim option;
  typ : Stype.t;
  pub : bool;
  loc : Loc.t;
  doc_ : docstring; [@sexp_drop_if Docstring.is_empty]
  ty_params_ : Tvar_env.t; [@sexp_drop_if Tvar_env.is_empty]
  arity_ : Fn_arity.t; [@sexp_drop_if Fn_arity.is_simple]
  param_names_ : string list;
}

include struct
  let _ = fun (_ : method_info) -> ()

  let sexp_of_method_info =
    (let (drop_if__015_ : docstring -> Stdlib.Bool.t) = Docstring.is_empty
     and (drop_if__020_ : Tvar_env.t -> Stdlib.Bool.t) = Tvar_env.is_empty
     and (drop_if__025_ : Fn_arity.t -> Stdlib.Bool.t) = Fn_arity.is_simple in
     fun {
           id = id__004_;
           prim = prim__006_;
           typ = typ__008_;
           pub = pub__010_;
           loc = loc__012_;
           doc_ = doc___016_;
           ty_params_ = ty_params___021_;
           arity_ = arity___026_;
           param_names_ = param_names___029_;
         } ->
       let bnds__003_ = ([] : _ Stdlib.List.t) in
       let bnds__003_ =
         let arg__030_ =
           Moon_sexp_conv.sexp_of_list Moon_sexp_conv.sexp_of_string
             param_names___029_
         in
         (S.List [ S.Atom "param_names_"; arg__030_ ] :: bnds__003_
           : _ Stdlib.List.t)
       in
       let bnds__003_ =
         if drop_if__025_ arity___026_ then bnds__003_
         else
           let arg__028_ = Fn_arity.sexp_of_t arity___026_ in
           let bnd__027_ = S.List [ S.Atom "arity_"; arg__028_ ] in
           (bnd__027_ :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         if drop_if__020_ ty_params___021_ then bnds__003_
         else
           let arg__023_ = Tvar_env.sexp_of_t ty_params___021_ in
           let bnd__022_ = S.List [ S.Atom "ty_params_"; arg__023_ ] in
           (bnd__022_ :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         if drop_if__015_ doc___016_ then bnds__003_
         else
           let arg__018_ = sexp_of_docstring doc___016_ in
           let bnd__017_ = S.List [ S.Atom "doc_"; arg__018_ ] in
           (bnd__017_ :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         let arg__013_ = Loc.sexp_of_t loc__012_ in
         (S.List [ S.Atom "loc"; arg__013_ ] :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         let arg__011_ = Moon_sexp_conv.sexp_of_bool pub__010_ in
         (S.List [ S.Atom "pub"; arg__011_ ] :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         let arg__009_ = Stype.sexp_of_t typ__008_ in
         (S.List [ S.Atom "typ"; arg__009_ ] :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         let arg__007_ =
           Moon_sexp_conv.sexp_of_option Primitive.sexp_of_prim prim__006_
         in
         (S.List [ S.Atom "prim"; arg__007_ ] :: bnds__003_ : _ Stdlib.List.t)
       in
       let bnds__003_ =
         let arg__005_ = Basic_qual_ident.sexp_of_t id__004_ in
         (S.List [ S.Atom "id"; arg__005_ ] :: bnds__003_ : _ Stdlib.List.t)
       in
       S.List bnds__003_
      : method_info -> S.t)

  let _ = sexp_of_method_info
end

type method_table_entry = Regular of method_info | Impl of method_info list

let sexp_of_method_table_entry = function
  | Regular mi | Impl (mi :: []) -> sexp_of_method_info mi
  | Impl mis -> List (Atom "Ambiguous" :: List.map sexp_of_method_info mis)

type t = {
  by_type : method_table_entry M.t H.t;
  by_name : (type_name * method_info) list M.t;
}

let sexp_of_t t = H.sexp_of_t (M.sexp_of_t sexp_of_method_table_entry) t.by_type
let empty () : t = { by_type = H.create 17; by_name = M.create 17 }

let find_regular_method (env : t) ~(type_name : type_name)
    ~(method_name : method_name) : method_info option =
  match H.find_opt env.by_type type_name with
  | Some method_table -> (
      match M.find_opt method_table method_name with
      | Some (Regular mi) -> Some mi
      | Some (Impl _) | None -> None)
  | None -> None

let find_method_opt (env : t) ~(type_name : type_name)
    ~(method_name : method_name) : method_table_entry option =
  match H.find_opt env.by_type type_name with
  | Some method_table -> M.find_opt method_table method_name
  | None -> None

let iter_methods_by_type (env : t) ~(type_name : type_name)
    (f : method_name -> method_info -> unit) =
  match H.find_opt env.by_type type_name with
  | Some method_table ->
      M.iter2 method_table (fun method_name entry ->
          match entry with
          | Regular mi | Impl (mi :: []) -> f method_name mi
          | Impl _ -> ())
  | None -> ()

let find_methods_by_name (env : t) ~method_name : (type_name * method_info) list
    =
  M.find_default env.by_name method_name []

let add_method (env : t) ~(type_name : type_name) ~(method_name : method_name)
    ~(method_info : method_info) : unit =
  (match H.find_opt env.by_type type_name with
  | Some method_table ->
      M.replace method_table method_name (Regular method_info)
  | None ->
      let method_table = M.create 17 in
      M.add method_table method_name (Regular method_info);
      H.add env.by_type type_name method_table);
  M.add_or_update env.by_name method_name
    [ (type_name, method_info) ]
    ~update:(fun methods -> (type_name, method_info) :: methods)
  |> ignore

let add_impl (env : t) ~(type_name : type_name) ~(method_name : method_name)
    ~(method_info : method_info) : unit =
  match H.find_opt env.by_type type_name with
  | Some method_table ->
      M.add_or_update method_table method_name (Impl [ method_info ])
        ~update:(fun entry ->
          match entry with
          | Regular _ -> entry
          | Impl mis -> Impl (method_info :: mis))
      |> ignore
  | None ->
      let method_table = M.create 17 in
      M.add method_table method_name (Impl [ method_info ]);
      H.add env.by_type type_name method_table

let to_value_info (m : method_info) : Value_info.t =
  Toplevel_value
    {
      id = m.id;
      typ = m.typ;
      pub = m.pub;
      kind = (match m.prim with None -> Normal | Some prim -> Prim prim);
      loc_ = m.loc;
      doc_ = m.doc_;
      ty_params_ = m.ty_params_;
      arity_ = Some m.arity_;
      param_names_ = m.param_names_;
      direct_use_loc_ = Not_direct_use;
    }

let iter (env : t) f =
  H.iter2 env.by_type (fun type_name tbl ->
      M.iter2 tbl (fun method_name entry ->
          match entry with
          | Regular mi | Impl (mi :: []) -> f type_name method_name mi
          | Impl _ -> ()))

let iter_by_name (env : t)
    (f : method_name * (type_name * method_info) list -> unit) =
  M.iter env.by_name f

type method_array = (type_name * (method_name * method_table_entry) array) array

include struct
  let _ = fun (_ : method_array) -> ()

  let sexp_of_method_array =
    (fun x__039_ ->
       Moon_sexp_conv.sexp_of_array
         (fun (arg0__035_, arg1__036_) ->
           let res0__037_ = sexp_of_type_name arg0__035_
           and res1__038_ =
             Moon_sexp_conv.sexp_of_array
               (fun (arg0__031_, arg1__032_) ->
                 let res0__033_ = sexp_of_method_name arg0__031_
                 and res1__034_ = sexp_of_method_table_entry arg1__032_ in
                 S.List [ res0__033_; res1__034_ ])
               arg1__036_
           in
           S.List [ res0__037_; res1__038_ ])
         x__039_
      : method_array -> S.t)

  let _ = sexp_of_method_array
end

let export (env : t) ~export_private : method_array =
  let result =
    Vec.make (H.length env.by_type)
      ~dummy:(Type_path.Builtin.type_path_int, [||])
  in
  H.iter2 env.by_type (fun self_type methods ->
      if
        export_private
        || (not (Type_path_util.is_foreign self_type))
        || !Basic_config.current_package = Basic_config.builtin_package
      then (
        let methods_vec = Vec.make (M.length methods) ~dummy:("", Impl []) in
        M.iter2 methods (fun method_name entry ->
            match entry with
            | Regular method_info ->
                if export_private || method_info.pub then
                  Vec.push methods_vec (method_name, entry)
            | Impl mis -> (
                match Lst.filter mis (fun mi -> export_private || mi.pub) with
                | [] -> ()
                | mis -> Vec.push methods_vec (method_name, Impl mis)));
        if Vec.length methods_vec > 0 then
          Vec.push result (self_type, Vec.to_array methods_vec)));
  Vec.to_array result

let export_regular_methods (env : t) :
    (type_name * (method_name * method_info) array) array =
  let result =
    Vec.make (H.length env.by_type)
      ~dummy:(Type_path.Builtin.type_path_int, [||])
  in
  H.iter2 env.by_type (fun self_type methods ->
      if
        (not (Type_path_util.is_foreign self_type))
        || !Basic_config.current_package = Basic_config.builtin_package
      then (
        let methods_vec = Vec.empty () in
        M.iter2 methods (fun method_name entry ->
            match entry with
            | Regular method_info ->
                Vec.push methods_vec (method_name, method_info)
            | Impl _ -> ());
        if Vec.length methods_vec > 0 then
          Vec.push result (self_type, Vec.to_array methods_vec)));
  Vec.to_array result

let import (methods_by_type : method_array) : t =
  let by_name = M.create 17 in
  let by_type = H.create (Array.length methods_by_type * 3 / 2) in
  Arr.iter methods_by_type (fun (type_name, methods) ->
      let tbl = M.create (Array.length methods * 3 / 2) in
      Arr.iter methods (fun (method_name, entry) ->
          (match entry with
          | Regular method_info ->
              M.add_or_update by_name method_name
                [ (type_name, method_info) ]
                ~update:(List.cons (type_name, method_info))
              |> ignore
          | Impl _ -> ());
          M.add tbl method_name entry);
      H.add by_type type_name tbl);
  { by_type; by_name }
