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


module Lst = Basic_lst
module Tid = Basic_ty_ident
module Hash_tid = Basic_ty_ident.Hash
module Hash_mid = Mtype.Id_hash

let rec mtype_to_ltype ~(type_defs : Ltype.type_defs_with_context)
    ~(mtype_defs : Mtype.defs) (t : Mtype.t) : Ltype.t =
  let rec go (t : Mtype.t) : Ltype.t =
    match t with
    | T_int -> I32_Int
    | T_char -> I32_Char
    | T_bool -> I32_Bool
    | T_unit -> I32_Unit
    | T_byte -> I32_Byte
    | T_uint -> U32
    | T_optimized_option { elem } -> (
        match elem with
        | T_char | T_byte | T_bool | T_unit -> I32_Option_Char
        | T_int | T_uint -> I64
        | T_int64 | T_uint64 | T_float | T_double | T_func _ | T_any _
        | T_optimized_option _ | T_maybe_uninit _ | T_error_value_result _ ->
            assert false
        | T_string | T_bytes | T_tuple _ | T_fixedarray _ | T_trait _
        | T_constr _ -> (
            match go elem with
            | Ref { tid } -> Ref_nullable { tid }
            | Ref_lazy_init { tid } -> Ref_nullable { tid }
            | Ref_string -> Ref_nullable { tid = Ltype.tid_string }
            | Ref_bytes -> Ref_nullable { tid = Ltype.tid_bytes }
            | _ -> assert false))
    | T_int64 -> I64
    | T_uint64 -> U64
    | T_float -> F32
    | T_double -> F64
    | T_string -> Ref_string
    | T_bytes -> Ref_bytes
    | T_fixedarray { elem = T_int } -> Ltype.ref_array_i32
    | T_fixedarray { elem = T_int64 } -> Ltype.ref_array_i64
    | T_fixedarray { elem = T_double } -> Ltype.ref_array_f64
    | T_fixedarray { elem } ->
        let name = Name_mangle.make_type_name t in
        (if not (Hash_tid.mem type_defs.defs name) then
           let ltype = Ltype.Ref_array { elem = go elem } in
           Hash_tid.add type_defs.defs name ltype);
        Ref { tid = name }
    | T_maybe_uninit t -> (
        match go t with
        | Ref { tid } when not (Tid.equal tid Ltype.tid_enum) ->
            Ref_lazy_init { tid }
        | t -> t)
    | T_func { params; return } -> (
        let fn_sig : Ltype.fn_sig =
          {
            params = Lst.map params go;
            ret = mtype_to_ltype_return ~mtype_defs ~type_defs return;
          }
        in
        match Ltype.FnSigHash.find_opt type_defs.fn_sig_tbl fn_sig with
        | Some name -> Ref { tid = name }
        | None ->
            let name = Name_mangle.make_type_name t in
            let ltype = Ltype.Ref_closure_abstract { fn_sig } in
            Ltype.FnSigHash.add type_defs.fn_sig_tbl fn_sig name;
            assert (not (Hash_tid.mem type_defs.defs name));
            Hash_tid.add type_defs.defs name ltype;
            Ref { tid = name })
    | T_tuple { tys } ->
        let name = Name_mangle.make_type_name t in
        (if not (Hash_tid.mem type_defs.defs name) then
           let ltype =
             Ltype.Ref_struct { fields = Lst.map tys (fun t -> (go t, false)) }
           in
           Hash_tid.add type_defs.defs name ltype);
        Ref { tid = name }
    | T_constr id | T_trait id -> (
        match Hash_mid.find_exn mtype_defs.defs id with
        | Placeholder -> assert false
        | Externref -> Ref_extern
        | Variant _ | Constant_variant_constr -> Ltype.ref_enum
        | Record _ | Trait _ | Variant_constr ->
            let tid = Mtype.id_to_tid id in
            Ref { tid })
    | T_any _ -> Ref_any
    | T_error_value_result _ -> assert false
  in
  go t

and mtype_to_ltype_return ~(type_defs : Ltype.type_defs_with_context)
    ~(mtype_defs : Mtype.defs) (t : Mtype.t) : Ltype.return_type =
  match t with
  | T_error_value_result { ok; err; id = _ } ->
      let ok_ty = mtype_to_ltype ~type_defs ~mtype_defs ok in
      let err_ty = mtype_to_ltype ~type_defs ~mtype_defs err in
      Ret_error { ok_ty; err_ty }
  | T_int | T_char | T_bool | T_unit | T_byte | T_int64 | T_uint | T_uint64
  | T_float | T_double | T_string | T_bytes | T_optimized_option _ | T_func _
  | T_tuple _ | T_fixedarray _ | T_constr _ | T_trait _ | T_any _
  | T_maybe_uninit _ ->
      Ret_single (mtype_to_ltype ~type_defs ~mtype_defs t)

let transl_mtype_defs (mtype_defs : Mtype.defs) : Ltype.type_defs_with_context =
  let type_defs = Hash_tid.of_list Ltype.predefs in
  let type_defs_with_context : Ltype.type_defs_with_context =
    { defs = type_defs; fn_sig_tbl = Ltype.FnSigHash.create 17 }
  in
  let mtype_to_ltype =
    mtype_to_ltype ~type_defs:type_defs_with_context ~mtype_defs
  in
  let add_type (id, info) =
    match info with
    | Mtype.Placeholder | Externref -> ()
    | Variant { constrs } ->
        Lst.iter constrs (fun { payload; tag } ->
            let constr_name =
              Name_mangle.make_constr_name (Mtype.id_to_tid id) tag
            in
            let args =
              Lst.map payload (fun { field_type; mut; _ } ->
                  (mtype_to_ltype field_type, mut))
            in
            Hash_tid.add type_defs constr_name (Ref_constructor { args }))
    | Variant_constr | Constant_variant_constr -> ()
    | Record { fields } ->
        let fields =
          Lst.map fields (fun { field_type; mut; _ } ->
              (mtype_to_ltype field_type, mut))
        in
        Hash_tid.add type_defs (Mtype.id_to_tid id) (Ref_struct { fields })
    | Trait { methods } ->
        let methods =
          Lst.map methods (fun { params_ty; return_ty; _ } : Ltype.fn_sig ->
              {
                params = Lst.map params_ty mtype_to_ltype;
                ret =
                  mtype_to_ltype_return ~type_defs:type_defs_with_context
                    ~mtype_defs return_ty;
              })
        in
        Hash_tid.add type_defs (Mtype.id_to_tid id) (Ref_object { methods })
  in
  Hash_mid.iter mtype_defs.defs add_type;
  type_defs_with_context

let constr_to_ltype ~(tag : Tag.t) (t : Mtype.t) : Tid.t =
  let name = Name_mangle.make_type_name t in
  Name_mangle.make_constr_name name tag
