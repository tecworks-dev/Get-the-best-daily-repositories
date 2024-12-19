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
module Type_path = Basic_type_path
module Lst = Basic_lst

type id = string

include struct
  let _ = fun (_ : id) -> ()
  let sexp_of_id = (Moon_sexp_conv.sexp_of_string : id -> S.t)
  let _ = sexp_of_id
end

module Id_hash : Basic_hash_intf.S with type key = id = Basic_hash_string

let id_to_string (s : id) = s
let id_to_tid (s : id) = Basic_ty_ident.of_string s

type t =
  | T_int
  | T_char
  | T_bool
  | T_unit
  | T_byte
  | T_int64
  | T_uint
  | T_uint64
  | T_float
  | T_double
  | T_string
  | T_bytes
  | T_optimized_option of { elem : t }
  | T_func of { params : t list; return : t }
  | T_tuple of { tys : t list }
  | T_fixedarray of { elem : t }
  | T_constr of id
  | T_trait of id
  | T_any of { name : id }
  | T_maybe_uninit of t
  | T_error_value_result of { ok : t; err : t; id : id }
[@@warning "+4"]

include struct
  let _ = fun (_ : t) -> ()

  let rec sexp_of_t =
    (function
     | T_int -> S.Atom "T_int"
     | T_char -> S.Atom "T_char"
     | T_bool -> S.Atom "T_bool"
     | T_unit -> S.Atom "T_unit"
     | T_byte -> S.Atom "T_byte"
     | T_int64 -> S.Atom "T_int64"
     | T_uint -> S.Atom "T_uint"
     | T_uint64 -> S.Atom "T_uint64"
     | T_float -> S.Atom "T_float"
     | T_double -> S.Atom "T_double"
     | T_string -> S.Atom "T_string"
     | T_bytes -> S.Atom "T_bytes"
     | T_optimized_option { elem = elem__002_ } ->
         let bnds__001_ = ([] : _ Stdlib.List.t) in
         let bnds__001_ =
           let arg__003_ = sexp_of_t elem__002_ in
           (S.List [ S.Atom "elem"; arg__003_ ] :: bnds__001_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_optimized_option" :: bnds__001_)
     | T_func { params = params__005_; return = return__007_ } ->
         let bnds__004_ = ([] : _ Stdlib.List.t) in
         let bnds__004_ =
           let arg__008_ = sexp_of_t return__007_ in
           (S.List [ S.Atom "return"; arg__008_ ] :: bnds__004_
             : _ Stdlib.List.t)
         in
         let bnds__004_ =
           let arg__006_ = Moon_sexp_conv.sexp_of_list sexp_of_t params__005_ in
           (S.List [ S.Atom "params"; arg__006_ ] :: bnds__004_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_func" :: bnds__004_)
     | T_tuple { tys = tys__010_ } ->
         let bnds__009_ = ([] : _ Stdlib.List.t) in
         let bnds__009_ =
           let arg__011_ = Moon_sexp_conv.sexp_of_list sexp_of_t tys__010_ in
           (S.List [ S.Atom "tys"; arg__011_ ] :: bnds__009_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_tuple" :: bnds__009_)
     | T_fixedarray { elem = elem__013_ } ->
         let bnds__012_ = ([] : _ Stdlib.List.t) in
         let bnds__012_ =
           let arg__014_ = sexp_of_t elem__013_ in
           (S.List [ S.Atom "elem"; arg__014_ ] :: bnds__012_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_fixedarray" :: bnds__012_)
     | T_constr arg0__015_ ->
         let res0__016_ = sexp_of_id arg0__015_ in
         S.List [ S.Atom "T_constr"; res0__016_ ]
     | T_trait arg0__017_ ->
         let res0__018_ = sexp_of_id arg0__017_ in
         S.List [ S.Atom "T_trait"; res0__018_ ]
     | T_any { name = name__020_ } ->
         let bnds__019_ = ([] : _ Stdlib.List.t) in
         let bnds__019_ =
           let arg__021_ = sexp_of_id name__020_ in
           (S.List [ S.Atom "name"; arg__021_ ] :: bnds__019_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_any" :: bnds__019_)
     | T_maybe_uninit arg0__022_ ->
         let res0__023_ = sexp_of_t arg0__022_ in
         S.List [ S.Atom "T_maybe_uninit"; res0__023_ ]
     | T_error_value_result { ok = ok__025_; err = err__027_; id = id__029_ } ->
         let bnds__024_ = ([] : _ Stdlib.List.t) in
         let bnds__024_ =
           let arg__030_ = sexp_of_id id__029_ in
           (S.List [ S.Atom "id"; arg__030_ ] :: bnds__024_ : _ Stdlib.List.t)
         in
         let bnds__024_ =
           let arg__028_ = sexp_of_t err__027_ in
           (S.List [ S.Atom "err"; arg__028_ ] :: bnds__024_ : _ Stdlib.List.t)
         in
         let bnds__024_ =
           let arg__026_ = sexp_of_t ok__025_ in
           (S.List [ S.Atom "ok"; arg__026_ ] :: bnds__024_ : _ Stdlib.List.t)
         in
         S.List (S.Atom "T_error_value_result" :: bnds__024_)
      : t -> S.t)

  let _ = sexp_of_t
end

let is_numeric (t : t) =
  match t with
  | T_unit | T_int | T_uint | T_char | T_bool | T_byte | T_int64 | T_uint64
  | T_float | T_double ->
      true
  | T_optimized_option { elem } -> (
      match elem with
      | T_char | T_bool | T_byte | T_unit | T_int | T_uint -> true
      | T_int64 | T_uint64 | T_float | T_double | T_optimized_option _ | T_bytes
      | T_string | T_func _ | T_tuple _ | T_fixedarray _ | T_maybe_uninit _
      | T_constr _ | T_trait _ | T_any _ | T_error_value_result _ ->
          false)
  | T_bytes | T_string | T_func _ | T_tuple _ | T_fixedarray _
  | T_maybe_uninit _ | T_constr _ | T_trait _ | T_any _ | T_error_value_result _
    ->
      false

type field_name = Named of string | Indexed of int

include struct
  let _ = fun (_ : field_name) -> ()

  let sexp_of_field_name =
    (function
     | Named arg0__031_ ->
         let res0__032_ = Moon_sexp_conv.sexp_of_string arg0__031_ in
         S.List [ S.Atom "Named"; res0__032_ ]
     | Indexed arg0__033_ ->
         let res0__034_ = Moon_sexp_conv.sexp_of_int arg0__033_ in
         S.List [ S.Atom "Indexed"; res0__034_ ]
      : field_name -> S.t)

  let _ = sexp_of_field_name
end

let field_index0 = Indexed 0
let field_index1 = Indexed 1
let field_index2 = Indexed 2
let field_index3 = Indexed 3

let field_indexed i =
  match i with
  | 0 -> field_index0
  | 1 -> field_index1
  | 2 -> field_index2
  | 3 -> field_index3
  | n -> Indexed n

type field_info = { field_type : t; name : field_name; mut : bool }

include struct
  let _ = fun (_ : field_info) -> ()

  let sexp_of_field_info =
    (fun { field_type = field_type__036_; name = name__038_; mut = mut__040_ } ->
       let bnds__035_ = ([] : _ Stdlib.List.t) in
       let bnds__035_ =
         let arg__041_ = Moon_sexp_conv.sexp_of_bool mut__040_ in
         (S.List [ S.Atom "mut"; arg__041_ ] :: bnds__035_ : _ Stdlib.List.t)
       in
       let bnds__035_ =
         let arg__039_ = sexp_of_field_name name__038_ in
         (S.List [ S.Atom "name"; arg__039_ ] :: bnds__035_ : _ Stdlib.List.t)
       in
       let bnds__035_ =
         let arg__037_ = sexp_of_t field_type__036_ in
         (S.List [ S.Atom "field_type"; arg__037_ ] :: bnds__035_
           : _ Stdlib.List.t)
       in
       S.List bnds__035_
      : field_info -> S.t)

  let _ = sexp_of_field_info
end

type constr_info = { payload : field_info list; tag : Tag.t }

include struct
  let _ = fun (_ : constr_info) -> ()

  let sexp_of_constr_info =
    (fun { payload = payload__043_; tag = tag__045_ } ->
       let bnds__042_ = ([] : _ Stdlib.List.t) in
       let bnds__042_ =
         let arg__046_ = Tag.sexp_of_t tag__045_ in
         (S.List [ S.Atom "tag"; arg__046_ ] :: bnds__042_ : _ Stdlib.List.t)
       in
       let bnds__042_ =
         let arg__044_ =
           Moon_sexp_conv.sexp_of_list sexp_of_field_info payload__043_
         in
         (S.List [ S.Atom "payload"; arg__044_ ] :: bnds__042_
           : _ Stdlib.List.t)
       in
       S.List bnds__042_
      : constr_info -> S.t)

  let _ = sexp_of_constr_info
end

type method_info = {
  params_ty : t list;
  return_ty : t;
  index : int;
  name : string;
}

include struct
  let _ = fun (_ : method_info) -> ()

  let sexp_of_method_info =
    (fun {
           params_ty = params_ty__048_;
           return_ty = return_ty__050_;
           index = index__052_;
           name = name__054_;
         } ->
       let bnds__047_ = ([] : _ Stdlib.List.t) in
       let bnds__047_ =
         let arg__055_ = Moon_sexp_conv.sexp_of_string name__054_ in
         (S.List [ S.Atom "name"; arg__055_ ] :: bnds__047_ : _ Stdlib.List.t)
       in
       let bnds__047_ =
         let arg__053_ = Moon_sexp_conv.sexp_of_int index__052_ in
         (S.List [ S.Atom "index"; arg__053_ ] :: bnds__047_ : _ Stdlib.List.t)
       in
       let bnds__047_ =
         let arg__051_ = sexp_of_t return_ty__050_ in
         (S.List [ S.Atom "return_ty"; arg__051_ ] :: bnds__047_
           : _ Stdlib.List.t)
       in
       let bnds__047_ =
         let arg__049_ =
           Moon_sexp_conv.sexp_of_list sexp_of_t params_ty__048_
         in
         (S.List [ S.Atom "params_ty"; arg__049_ ] :: bnds__047_
           : _ Stdlib.List.t)
       in
       S.List bnds__047_
      : method_info -> S.t)

  let _ = sexp_of_method_info
end

type info =
  | Placeholder
  | Externref
  | Variant of { constrs : constr_info list }
  | Variant_constr
  | Constant_variant_constr
  | Record of { fields : field_info list }
  | Trait of { methods : method_info list }

include struct
  let _ = fun (_ : info) -> ()

  let sexp_of_info =
    (function
     | Placeholder -> S.Atom "Placeholder"
     | Externref -> S.Atom "Externref"
     | Variant { constrs = constrs__057_ } ->
         let bnds__056_ = ([] : _ Stdlib.List.t) in
         let bnds__056_ =
           let arg__058_ =
             Moon_sexp_conv.sexp_of_list sexp_of_constr_info constrs__057_
           in
           (S.List [ S.Atom "constrs"; arg__058_ ] :: bnds__056_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Variant" :: bnds__056_)
     | Variant_constr -> S.Atom "Variant_constr"
     | Constant_variant_constr -> S.Atom "Constant_variant_constr"
     | Record { fields = fields__060_ } ->
         let bnds__059_ = ([] : _ Stdlib.List.t) in
         let bnds__059_ =
           let arg__061_ =
             Moon_sexp_conv.sexp_of_list sexp_of_field_info fields__060_
           in
           (S.List [ S.Atom "fields"; arg__061_ ] :: bnds__059_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Record" :: bnds__059_)
     | Trait { methods = methods__063_ } ->
         let bnds__062_ = ([] : _ Stdlib.List.t) in
         let bnds__062_ =
           let arg__064_ =
             Moon_sexp_conv.sexp_of_list sexp_of_method_info methods__063_
           in
           (S.List [ S.Atom "methods"; arg__064_ ] :: bnds__062_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Trait" :: bnds__062_)
      : info -> S.t)

  let _ = sexp_of_info
end

type defs = { defs : info Id_hash.t; ext_tags : int Hash_string.t }

include struct
  let _ = fun (_ : defs) -> ()

  let sexp_of_defs =
    (fun { defs = defs__066_; ext_tags = ext_tags__068_ } ->
       let bnds__065_ = ([] : _ Stdlib.List.t) in
       let bnds__065_ =
         let arg__069_ =
           Hash_string.sexp_of_t Moon_sexp_conv.sexp_of_int ext_tags__068_
         in
         (S.List [ S.Atom "ext_tags"; arg__069_ ] :: bnds__065_
           : _ Stdlib.List.t)
       in
       let bnds__065_ =
         let arg__067_ = Id_hash.sexp_of_t sexp_of_info defs__066_ in
         (S.List [ S.Atom "defs"; arg__067_ ] :: bnds__065_ : _ Stdlib.List.t)
       in
       S.List bnds__065_
      : defs -> S.t)

  let _ = sexp_of_defs
end

let sexp_of_defs defs =
  if Hash_string.length defs.ext_tags = 0 then
    (fun x__070_ -> Id_hash.sexp_of_t sexp_of_info x__070_) defs.defs
  else sexp_of_defs defs

let find_stype_exn (stype_defs : Typing_info.stype_defs) (pkg : string)
    (id : string) : Typedecl_info.t =
  let types = Hash_string.find_exn stype_defs pkg in
  Typing_info.find_type_exn types id

let find_trait_exn (stype_defs : Typing_info.stype_defs) (pkg : string)
    (id : string) : Trait_decl.t =
  let types = Hash_string.find_exn stype_defs pkg in
  Typing_info.find_trait_exn types id

let is_optimizable_option_elem t mtype_defs =
  match t with
  | T_char | T_bool | T_int | T_uint | T_unit | T_byte | T_string | T_bytes
  | T_tuple _ | T_fixedarray _ | T_trait _ ->
      true
  | T_constr id -> (
      match Id_hash.find_opt mtype_defs id with
      | Some Externref -> false
      | None | Some _ -> true)
  | T_int64 | T_uint64 | T_func _ -> false
  | T_float | T_double | T_any _ | T_optimized_option _ | T_maybe_uninit _ ->
      false
  | T_error_value_result _ -> false

let error_mid = Type_args.mangle_ty Stype.error

let make_suberror_mid pkg name =
  if pkg = "" then (error_mid ^ "." ^ name : Stdlib.String.t)
  else Stdlib.String.concat "" [ error_mid; "."; pkg; "."; name ]

let from_stype (stype : Stype.t) ~(stype_defs : Typing_info.stype_defs)
    ~(mtype_defs : defs) : t =
  let rec go (stype : Stype.t) : t =
    let stype = Stype.type_repr stype in
    match stype with
    | T_builtin T_unit -> T_unit
    | T_builtin T_bool -> T_bool
    | T_builtin T_byte -> T_byte
    | T_builtin T_char -> T_char
    | T_builtin T_int -> T_int
    | T_builtin T_int64 -> T_int64
    | T_builtin T_uint -> T_uint
    | T_builtin T_uint64 -> T_uint64
    | T_builtin T_float -> T_float
    | T_builtin T_double -> T_double
    | T_builtin T_string -> T_string
    | T_builtin T_bytes -> T_bytes
    | Tarrow { params_ty; ret_ty; err_ty = None } ->
        T_func { params = Lst.map params_ty go; return = go ret_ty }
    | Tarrow { params_ty; ret_ty; err_ty = Some err_ty } ->
        T_func
          {
            params = Lst.map params_ty go;
            return = go (Stype.make_multi_value_result_ty ~ok_ty:ret_ty ~err_ty);
          }
    | T_constr
        {
          type_constructor = Toplevel { pkg; id };
          tys = [];
          is_suberror_ = true;
        } ->
        let mid = make_suberror_mid pkg id in
        (if not (Id_hash.mem mtype_defs.defs mid) then
           let typedecl_info = find_stype_exn stype_defs pkg id in
           add_type_def stype mid typedecl_info);
        T_constr error_mid
    | T_constr { type_constructor = T_error; _ } ->
        (match Id_hash.find_opt mtype_defs.defs error_mid with
        | None ->
            Id_hash.add mtype_defs.defs error_mid (Variant { constrs = [] })
        | Some _ -> ());
        T_constr error_mid
    | T_constr { type_constructor = T_option as c; tys = t :: [] } ->
        let elem_ty = go t in
        if is_optimizable_option_elem elem_ty mtype_defs.defs then
          T_optimized_option { elem = elem_ty }
        else
          let type_name =
            Type_path.export_name ~cur_pkg_name:!Basic_config.current_package c
          in
          let s = Type_args.mangle_ty stype in
          (if not (Id_hash.mem mtype_defs.defs s) then
             match Typing_info.find_type Builtin.builtin_types type_name with
             | Some typedecl_info -> add_type_def stype s typedecl_info
             | None -> assert false);
          T_constr s
    | T_constr { type_constructor; tys = _ }
      when Type_path.equal type_constructor
             Type_path.Builtin.type_path_foreach_result ->
        let s = Type_args.mangle_ty stype in
        if not (Id_hash.mem mtype_defs.defs s) then
          add_type_def stype s Foreach_util.foreach_result;
        T_constr s
    | T_constr { type_constructor; tys; only_tag_enum_; is_suberror_ } -> (
        let newtype_info =
          match type_constructor with
          | Tuple _ | T_unit | T_bool | T_byte | T_char | T_int | T_int64
          | T_uint | T_uint64 | T_float | T_double | T_string | T_option
          | T_result | T_error_value_result | T_fixedarray | T_bytes | T_ref
          | T_error | Constr _ ->
              None
          | Toplevel { pkg; id } -> (
              let typedecl_info = find_stype_exn stype_defs pkg id in
              match typedecl_info.ty_desc with
              | Extern_type | Abstract_type | Record_type _ | Variant_type _
              | Error_type _ | ErrorEnum_type _ ->
                  None
              | New_type info -> Some info)
        in
        match newtype_info with
        | Some { newtype_constr; recursive; _ } -> (
            if recursive then T_any { name = Type_args.mangle_ty stype }
            else
              let newtype_ty, newtype_ty_field =
                Poly_type.instantiate_constr newtype_constr
              in
              Ctype.unify_exn newtype_ty stype;
              match newtype_ty_field with t :: [] -> go t | _ -> assert false)
        | None -> (
            if only_tag_enum_ then T_int
            else
              match type_constructor with
              | T_int | T_char | T_bool | T_byte | T_unit | T_int64 | T_uint
              | T_uint64 | T_float | T_double | T_bytes | T_string ->
                  assert false
              | T_error -> T_constr error_mid
              | T_fixedarray -> (
                  match tys with
                  | elem_ty :: [] ->
                      let elem = go elem_ty in
                      if elem = T_byte then T_bytes else T_fixedarray { elem }
                  | [] | _ :: _ -> assert false)
              | T_option | T_ref | T_result | T_error_value_result ->
                  let type_name =
                    Type_path.export_name
                      ~cur_pkg_name:!Basic_config.current_package
                      type_constructor
                  in
                  let s = Type_args.mangle_ty stype in
                  (if not (Id_hash.mem mtype_defs.defs s) then
                     match
                       Typing_info.find_type Builtin.builtin_types type_name
                     with
                     | Some typedecl_info -> add_type_def stype s typedecl_info
                     | None -> assert false);
                  if Type_path.is_multi_value_result type_constructor then
                    match tys with
                    | [ ok_ty; err_ty ] ->
                        T_error_value_result
                          { ok = go ok_ty; err = go err_ty; id = s }
                    | _ -> assert false
                  else T_constr s
              | Tuple _ -> T_tuple { tys = Lst.map tys go }
              | _
                when Type_path.equal type_constructor
                       Type_path.Builtin.type_path_sourceloc ->
                  T_string
              | Toplevel { pkg; id = "UnsafeMaybeUninit" }
                when pkg = Basic_config.builtin_package -> (
                  match tys with
                  | t :: [] ->
                      let t = go t in
                      if is_numeric t then t else T_maybe_uninit t
                  | _ -> assert false)
              | Toplevel { pkg; id } ->
                  let s = Type_args.mangle_ty stype in
                  if not (Id_hash.mem mtype_defs.defs s) then (
                    let typedecl_info = find_stype_exn stype_defs pkg id in
                    match typedecl_info.ty_desc with
                    | Record_type { fields = [] } -> T_unit
                    | _ ->
                        add_type_def stype s typedecl_info;
                        T_constr s)
                  else T_constr s
              | Constr
                  {
                    tag =
                      Extensible_tag
                        { pkg; type_name; name; total = _; index = _ };
                    _;
                  } ->
                  let tag_str =
                    Basic_constr_info.ext_tag_to_str ~pkg ~type_name ~name
                  in
                  let mid = (error_mid ^ "." ^ tag_str : Stdlib.String.t) in
                  if not (Id_hash.mem mtype_defs.defs mid) then
                    Id_hash.add mtype_defs.defs mid Variant_constr;
                  let error_type_mid = make_suberror_mid pkg type_name in
                  (if not (Id_hash.mem mtype_defs.defs error_type_mid) then
                     let typedecl_info =
                       find_stype_exn stype_defs pkg type_name
                     in
                     add_type_def stype error_type_mid typedecl_info);
                  T_constr mid
              | Constr
                  { ty; tag = Constr_tag_regular { name_; is_constant_; _ } }
                -> (
                  let enum_ty : Stype.t =
                    T_constr
                      {
                        type_constructor = ty;
                        tys;
                        generic_ = false;
                        only_tag_enum_ = false;
                        is_suberror_;
                      }
                  in
                  match go enum_ty with
                  | T_constr id ->
                      let id = id ^ "." ^ name_ in
                      Id_hash.replace mtype_defs.defs id
                        (if is_constant_ then Constant_variant_constr
                         else Variant_constr);
                      T_constr id
                  | T_error_value_result { id; ok; err } ->
                      let id = id ^ "." ^ name_ in
                      Id_hash.replace mtype_defs.defs id Variant_constr;
                      T_error_value_result { id; ok; err }
                  | ( T_int | T_char | T_bool | T_unit | T_byte | T_int64
                    | T_uint | T_uint64 | T_float | T_double | T_string
                    | T_bytes | T_optimized_option _ | T_func _ | T_tuple _
                    | T_fixedarray _ | T_trait _ | T_any _ | T_maybe_uninit _ )
                    as ty ->
                      ty)))
    | Tvar _ | Tparam _ | T_blackhole -> assert false
    | T_trait type_path ->
        let s = Type_args.mangle_ty stype in
        (if not (Id_hash.mem mtype_defs.defs s) then
           let trait_info =
             match type_path with
             | Toplevel { pkg; id } -> find_trait_exn stype_defs pkg id
             | _ -> assert false
           in
           add_trait_def s trait_info);
        T_trait s
  and add_type_def (stype : Stype.t) (mid : id) (decl : Typedecl_info.t) =
    Id_hash.add mtype_defs.defs mid Placeholder;
    match decl.ty_desc with
    | Extern_type -> Id_hash.replace mtype_defs.defs mid Externref
    | Abstract_type | New_type _ -> ()
    | Error_type c -> (
        Id_hash.replace mtype_defs.defs mid Variant_constr;
        let constr =
          let tag = Tag.of_core_tag mtype_defs.ext_tags c.cs_tag in
          let payload =
            Lst.mapi c.cs_args (fun i ty ->
                let field_type = go ty in
                { field_type; name = field_indexed i; mut = false })
          in
          { payload; tag }
        in
        match Id_hash.find_opt mtype_defs.defs error_mid with
        | None ->
            Id_hash.add mtype_defs.defs error_mid
              (Variant { constrs = [ constr ] })
        | Some (Variant { constrs = cs }) ->
            Id_hash.replace mtype_defs.defs error_mid
              (Variant { constrs = constr :: cs })
        | _ -> assert false)
    | ErrorEnum_type cs -> (
        Id_hash.replace mtype_defs.defs mid Variant_constr;
        let constrs =
          Lst.map cs (fun c ->
              let tag = Tag.of_core_tag mtype_defs.ext_tags c.cs_tag in
              let arity = c.cs_arity_ in
              let payload =
                Fn_arity.to_list_map2 arity c.cs_args (fun param_kind ty ->
                    let field_type = go ty in
                    match param_kind with
                    | Labelled { label; is_mut = mut; _ } ->
                        { field_type; name = Named label; mut }
                    | Positional index ->
                        { field_type; name = field_indexed index; mut = false }
                    | Optional _ | Autofill _ | Question_optional _ ->
                        assert false)
              in
              { payload; tag })
        in
        match Id_hash.find_opt mtype_defs.defs error_mid with
        | None -> Id_hash.add mtype_defs.defs error_mid (Variant { constrs })
        | Some (Variant { constrs = cs }) ->
            Id_hash.replace mtype_defs.defs error_mid
              (Variant { constrs = constrs @ cs })
        | _ -> assert false)
    | Record_type { fields = fs } ->
        let _, fs' =
          Poly_type.instantiate_record ~ty_record:(`Known stype) fs
        in
        let fields =
          Lst.map fs' (fun f ->
              {
                field_type = go f.ty_field;
                name = Named f.field_name;
                mut = f.mut;
              })
        in
        Id_hash.replace mtype_defs.defs mid (Record { fields })
    | Variant_type cs ->
        let constrs =
          Lst.map cs (fun c ->
              let tag = Tag.of_core_tag_no_ext c.cs_tag in
              let arity = c.cs_arity_ in
              let enum_ty, constr_tys = Poly_type.instantiate_constr c in
              Ctype.unify_exn stype enum_ty;
              let payload =
                Fn_arity.to_list_map2 arity constr_tys (fun param_kind ty ->
                    let field_type = go ty in
                    match param_kind with
                    | Labelled { label; is_mut = mut; _ } ->
                        { field_type; name = Named label; mut }
                    | Positional index ->
                        { field_type; name = field_indexed index; mut = false }
                    | Optional _ | Autofill _ | Question_optional _ ->
                        assert false)
              in
              { payload; tag })
        in
        Id_hash.replace mtype_defs.defs mid (Variant { constrs })
  and add_trait_def (mid : id) (decl : Trait_decl.t) =
    Id_hash.add mtype_defs.defs mid Placeholder;
    let methods =
      Lst.mapi decl.closure_methods (fun i (_, m) ->
          match m.method_typ with
          | Stype.Tarrow { params_ty = _ :: params_ty; ret_ty; err_ty } ->
              let params_ty = Lst.map params_ty go in
              let return_ty =
                match err_ty with
                | None -> go ret_ty
                | Some err_ty ->
                    go (Stype.make_multi_value_result_ty ~ok_ty:ret_ty ~err_ty)
              in
              { params_ty; return_ty; index = i; name = m.method_name }
          | _ -> assert false)
    in
    Id_hash.replace mtype_defs.defs mid (Trait { methods })
  in
  go stype

let get_constr_tid_exn (t : t) : id =
  match t with
  | T_constr tid -> tid
  | T_int | T_char | T_bool | T_unit | T_byte | T_int64 | T_uint | T_uint64
  | T_float | T_double | T_string | T_bytes | T_optimized_option _ | T_func _
  | T_tuple _ | T_fixedarray _ | T_trait _ | T_any _ | T_maybe_uninit _
  | T_error_value_result _ ->
      assert false

let is_func (t : t) : bool =
  match t with
  | T_func _ -> true
  | T_int | T_char | T_bool | T_unit | T_byte | T_int64 | T_uint | T_uint64
  | T_float | T_double | T_string | T_bytes | T_constr _ | T_optimized_option _
  | T_tuple _ | T_fixedarray _ | T_trait _ | T_any _ | T_maybe_uninit _
  | T_error_value_result _ ->
      false

let get_fixedarray_elem_exn (t : t) : t =
  match t with
  | T_fixedarray { elem } -> elem
  | T_int | T_char | T_bool | T_unit | T_byte | T_int64 | T_uint | T_uint64
  | T_float | T_double | T_string | T_bytes | T_constr _ | T_optimized_option _
  | T_tuple _ | T_trait _ | T_func _ | T_any _ | T_maybe_uninit _
  | T_error_value_result _ ->
      assert false

let is_uninit (t : t) : bool =
  match t with
  | T_maybe_uninit _ -> true
  | T_func _ | T_int | T_char | T_bool | T_unit | T_byte | T_int64 | T_uint
  | T_uint64 | T_float | T_double | T_string | T_bytes | T_constr _
  | T_optimized_option _ | T_tuple _ | T_fixedarray _ | T_trait _ | T_any _
  | T_error_value_result _ ->
      false
