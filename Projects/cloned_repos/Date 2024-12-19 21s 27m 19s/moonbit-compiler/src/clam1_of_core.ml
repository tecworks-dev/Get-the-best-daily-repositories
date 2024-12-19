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


module Ident = Clam1_ident
module Fn_address = Basic_fn_address
module Tid = Basic_ty_ident
module Ident_hashset = Basic_core_ident.Hashset
module Lst = Basic_lst
module Core_ident = Basic_core_ident
module Qual_ident = Basic_qual_ident

type binds = Clam1.top_func_item list

let base = ref Loc.no_location
let binds_init : binds ref = ref []
let new_top (x : Clam1.top_func_item) = binds_init := x :: !binds_init
let clam_unit : Clam1.lambda = Lconst (C_int { v = 0l; repr = None })
let local_non_well_knowns : Core_ident.Hashset.t = Core_ident.Hashset.create 17

let fix_var =
  object
    inherit [_] Clam1.map

    method! visit_var (self, replace) var =
      if Ident.equal var self then replace else var
  end

let fix_single_var ~self ~replace body =
  fix_var#visit_lambda (self, replace) body

let transl_params ~mtype_defs ~type_defs (params : Mcore.param list) :
    Ident.t list =
  Lst.map params (fun { binder; ty } ->
      Ident.of_core_ident binder
        ~ty:(Transl_mtype.mtype_to_ltype ~mtype_defs ~type_defs ty))

let transl_ident (id : Core_ident.t) ~(addr_tbl : Addr_table.t) ~mtype_defs
    ~type_defs mty : Ident.t =
  match Addr_table.find_opt addr_tbl id with
  | Some (Toplevel { name_as_closure = Some id; _ }) -> id
  | Some (Toplevel fn_info) -> (
      let ty = Transl_mtype.mtype_to_ltype ~mtype_defs ~type_defs mty in
      match[@warning "-fragile-match"] id with
      | Pdot qual_name ->
          let id =
            Ident.of_qual_ident ~ty
              (Qual_ident.map qual_name ~f:(fun name : Stdlib.String.t ->
                   name ^ ".clo"))
          in
          fn_info.name_as_closure <- Some id;
          id
      | _ -> assert false)
  | Some (Local (_addr, ty)) -> Ident.of_core_ident ~ty id
  | None ->
      let ty = Transl_mtype.mtype_to_ltype ~mtype_defs ~type_defs mty in
      Ident.of_core_ident ~ty id

let rec transl_expr ~name_hint ~mtype_defs ~addr_tbl ~type_defs ~object_methods
    (x : Mcore.expr) : Clam1.lambda =
  let transl_type mtype =
    Transl_mtype.mtype_to_ltype mtype ~type_defs ~mtype_defs
      [@@inline]
  in
  let transl_type_as_named_exn mtype =
    match transl_type mtype with
    | Ref_lazy_init { tid } | Ltype.Ref { tid } | Ref_nullable { tid } -> tid
    | Ref_bytes -> Ltype.tid_bytes
    | _ -> assert false
      [@@inline]
  in
  let transl_constr_type tag mtype =
    Transl_mtype.constr_to_ltype ~tag mtype
      [@@inline]
  in
  let go x =
    transl_expr ~name_hint ~mtype_defs ~addr_tbl ~type_defs ~object_methods x
      [@@inline]
  in
  let bind (rhs : Mcore.expr) cont : Clam1.lambda =
    match rhs with
    | Cexpr_var { id; ty; _ } ->
        let ty = transl_type ty in
        cont (Ident.of_core_ident ~ty id)
    | _ ->
        let ty = transl_type (Mcore.type_of_expr rhs) in
        let name = Ident.fresh ~ty "*bind" in
        Llet { name; e = go rhs; body = cont name }
      [@@inline]
  in
  let to_value (rhs : Mcore.expr) (cont : Clam1.value -> _) : Clam1.lambda =
    match rhs with
    | Cexpr_var { id = (Pident _ | Pdot _ | Plocal_method _) as id; ty; _ } ->
        cont (Vvar (transl_ident ~addr_tbl ~mtype_defs ~type_defs id ty))
    | Cexpr_const { c; _ } -> cont (Vconst c)
    | _ ->
        let name_hint =
          match rhs with
          | Cexpr_constr { constr; _ } -> constr.constr_name.name
          | Cexpr_tuple _ -> "tuple"
          | Cexpr_field { accessor = Label { label_name }; _ } -> label_name
          | _ -> "tmp"
        in
        let ty = transl_type (Mcore.type_of_expr rhs) in
        let name = Ident.fresh name_hint ~ty in
        Llet { name; e = go rhs; body = cont (Vvar name) }
      [@@inline]
  in
  let rec expr_list_to_value (exprs : Mcore.expr list)
      (cont : Clam1.value list -> _) =
    match exprs with
    | [] -> cont []
    | expr :: exprs ->
        to_value expr (fun value ->
            expr_list_to_value exprs (fun values -> cont (value :: values)))
  in
  let append_name_hint new_name : Stdlib.String.t =
    name_hint ^ "." ^ Core_ident.base_name new_name
      [@@inline]
  in
  let handle_abstract_closure_type ~(name : Core_ident.t option) (fn : Mcore.fn)
      (address : Fn_address.t) : Tid.t =
    let params = transl_params ~mtype_defs ~type_defs fn.params in
    let return_type_ =
      Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
        (Mcore.type_of_expr fn.body)
    in
    let sig_ : Ltype.fn_sig =
      { params = Lst.map params Ident.get_type; ret = return_type_ }
    in
    let abs_closure_tid =
      match Ltype.FnSigHash.find_opt type_defs.fn_sig_tbl sig_ with
      | Some tid -> tid
      | None ->
          let tid =
            Name_mangle.make_type_name
              (T_func
                 {
                   params = Lst.map fn.params (fun p -> p.ty);
                   return = Mcore.type_of_expr fn.body;
                 })
          in
          Ltype.FnSigHash.add type_defs.fn_sig_tbl sig_ tid;
          Tid.Hash.add type_defs.defs tid
            (Ltype.Ref_closure_abstract { fn_sig = sig_ });
          tid
    in
    let abs_closure_ty = Ltype.Ref { tid = abs_closure_tid } in
    (match name with
    | Some name ->
        Addr_table.add_local_fn_addr_and_type addr_tbl name address
          abs_closure_ty
    | None -> ());
    abs_closure_tid
  in
  let expr : Clam1.lambda =
    match x with
    | Cexpr_const { c; ty = _ } -> Lconst c
    | Cexpr_unit _ -> clam_unit
    | Cexpr_var { id; ty; prim = _ } ->
        let var = transl_ident id ty ~addr_tbl ~mtype_defs ~type_defs in
        Lvar { var }
    | Cexpr_prim { prim = Psequand; args = [ lhs; rhs ]; ty = _ } ->
        Lif
          {
            pred = go lhs;
            ifso = go rhs;
            ifnot = Lconst (C_bool false);
            type_ = Ret_single I32_Bool;
          }
    | Cexpr_prim { prim = Psequor; args = [ lhs; rhs ]; ty = _ } ->
        Lif
          {
            pred = go lhs;
            ifso = Lconst (C_bool true);
            ifnot = go rhs;
            type_ = Ret_single I32_Bool;
          }
    | Cexpr_prim { prim = Pcatch; args = [ body; on_exception ]; ty } ->
        Lcatch
          {
            body = go body;
            on_exception = go on_exception;
            type_ = transl_type ty;
          }
    | Cexpr_prim { prim; args; ty } -> (
        let make_prim fn =
          expr_list_to_value args (fun args -> Clam1.Lprim { fn; args })
            [@@inline]
        in
        match prim with
        | Parray_make ->
            let arr_type =
              let tid = Mtype.get_constr_tid_exn ty in
              match Mtype.Id_hash.find_opt mtype_defs.defs tid with
              | Some (Record { fields = arr_field :: _ }) ->
                  arr_field.field_type
              | _ -> assert false
            in
            let fixed_array =
              Mcore.prim
                (Pfixedarray_make { kind = EverySingleElem })
                args ~ty:arr_type
            in
            let vec_tid = transl_type_as_named_exn ty in
            to_value fixed_array (fun fixed_arr_value ->
                Lallocate
                  {
                    kind = Struct;
                    tid = vec_tid;
                    fields =
                      [
                        fixed_arr_value;
                        Vconst
                          (C_int
                             {
                               v = Int32.of_int (List.length args);
                               repr = None;
                             });
                      ];
                  })
        | Pfixedarray_length ->
            let arr_type = Mcore.type_of_expr (List.hd args) in
            if arr_type = T_bytes then make_prim Pbyteslength
            else make_prim Pfixedarray_length
        | Pfixedarray_make { kind } ->
            if ty = T_bytes && kind = LenAndInit then make_prim Pmakebytes
            else
              expr_list_to_value args (fun elems ->
                  Lmake_array { kind; tid = transl_type_as_named_exn ty; elems })
        | Pfixedarray_get_item { kind } ->
            let arr_type = Mcore.type_of_expr (List.hd args) in
            if arr_type = T_bytes then make_prim Pgetbytesitem
            else
              let tid = transl_type_as_named_exn arr_type in
              expr_list_to_value args (fun args ->
                  match[@warning "-fragile-match"] args with
                  | [ arr; index ] -> Larray_get_item { tid; arr; index; kind }
                  | _ -> assert false)
        | Pfixedarray_set_item { set_kind } ->
            let arr_type = Mcore.type_of_expr (List.hd args) in
            if arr_type = T_bytes then make_prim Psetbytesitem
            else
              expr_list_to_value args (fun args ->
                  let arr, index, item =
                    match args with
                    | [ arr; index ] -> (arr, index, None)
                    | [ arr; index; item ] -> (arr, index, Some item)
                    | _ -> assert false
                  in
                  Larray_set_item
                    {
                      arr;
                      index;
                      item;
                      tid = transl_type_as_named_exn arr_type;
                      kind = set_kind;
                    })
        | Prefeq ->
            let mty = Mcore.type_of_expr (List.hd args) in
            let lty = transl_type mty in
            make_prim
              (match lty with
              | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
              | I32_Option_Char ->
                  Pcomparison { operator = Eq; operand_type = I32 }
              | U32 -> Pcomparison { operator = Eq; operand_type = U32 }
              | I64 -> Pcomparison { operator = Eq; operand_type = I64 }
              | U64 -> Pcomparison { operator = Eq; operand_type = U64 }
              | F32 -> Pcomparison { operator = Eq; operand_type = F32 }
              | F64 -> Pcomparison { operator = Eq; operand_type = F64 }
              | Ref _ | Ref_lazy_init _ | Ref_nullable _ | Ref_extern
              | Ref_string | Ref_bytes | Ref_func | Ref_any ->
                  Prefeq)
        | Pcast _ -> (
            match[@warning "-fragile-match"] args with
            | arg :: [] -> Lcast { expr = go arg; target_type = transl_type ty }
            | _ -> assert false)
        | Penum_field { index; tag } ->
            let tid =
              Name_mangle.make_type_name (Mcore.type_of_expr (List.hd args))
            in
            expr_list_to_value args (fun args ->
                match[@warning "-fragile-match"] args with
                | Clam1.Vvar obj :: [] ->
                    let tag = Tag.of_core_tag mtype_defs.ext_tags tag in
                    Lget_field { kind = Enum { tag }; obj; index; tid }
                | _ -> assert false)
        | Pset_enum_field { index; tag } ->
            let tid =
              Name_mangle.make_type_name (Mcore.type_of_expr (List.hd args))
            in
            expr_list_to_value args (fun args ->
                match[@warning "-fragile-match"] args with
                | [ Clam1.Vvar obj; field ] ->
                    let tag = Tag.of_core_tag mtype_defs.ext_tags tag in
                    Lset_field { kind = Enum { tag }; obj; field; index; tid }
                | _ -> assert false)
        | Pmake_value_or_error { tag } -> (
            let tag = Tag.of_core_tag_no_ext tag in
            match[@warning "-fragile-match"] args with
            | arg :: [] ->
                to_value arg (fun argv ->
                    Lmake_multi_result
                      {
                        value = argv;
                        tag;
                        type_ =
                          Transl_mtype.mtype_to_ltype_return ~type_defs
                            ~mtype_defs ty;
                      })
            | _ -> assert false)
        | Pcall_object_method { method_index; _ } -> (
            let method_ty =
              transl_type
                (T_func
                   { params = Lst.map args Mcore.type_of_expr; return = ty })
            in
            match[@warning "-fragile-match"] args with
            | self :: args ->
                bind self (fun obj ->
                    expr_list_to_value args (fun args ->
                        Lapply
                          {
                            fn = Object { obj; method_index; method_ty };
                            args;
                            prim = None;
                          }))
            | _ -> assert false)
        | _ -> make_prim prim)
    | Cexpr_let { name; rhs; body; ty = _ } ->
        let ty_rhs = Mcore.type_of_expr rhs in
        let name = Ident.of_core_ident ~ty:(transl_type ty_rhs) name in
        Llet { name; e = go rhs; body = go body }
    | Cexpr_letfn { name; fn; body; kind; ty } -> (
        match kind with
        | Nonrec ->
            let name_hint = append_name_hint name in
            let address =
              Fn_address.fresh (name_hint ^ ".fn" : Stdlib.String.t)
            in
            if Core_ident.Hashset.mem local_non_well_knowns name then
              let abs_closure_tid =
                handle_abstract_closure_type ~name:(Some name) fn address
              in
              let closure, ty =
                closure_of_fn ~addr_tbl ~type_defs ~object_methods ~address
                  ~name_hint ~mtype_defs ~self:None ~abs_closure_tid fn
              in
              let body = go body in
              let name = Ident.of_core_ident name ~ty in
              Llet { name; e = Lclosure closure; body }
            else
              let tuple, binder =
                well_known_closure_of_fn ~mtype_defs ~addr_tbl ~type_defs
                  ~object_methods ~name_hint ~address ~self:name ~is_rec:false
                  fn
              in
              let body = go body in
              Llet { name = binder; e = tuple; body }
        | Rec ->
            let name_hint = append_name_hint name in
            let address =
              Fn_address.fresh (name_hint ^ ".fn" : Stdlib.String.t)
            in
            if Core_ident.Hashset.mem local_non_well_knowns name then
              let abs_closure_tid =
                handle_abstract_closure_type ~name:(Some name) fn address
              in
              let closure, ty =
                closure_of_fn ~addr_tbl ~type_defs ~object_methods ~name_hint
                  ~mtype_defs ~self:(Some name) ~address ~abs_closure_tid fn
              in
              let body = go body in
              let name = Ident.of_core_ident name ~ty in
              Llet { name; e = Lclosure closure; body }
            else
              let tuple, binder =
                well_known_closure_of_fn ~mtype_defs ~addr_tbl ~type_defs
                  ~object_methods ~name_hint ~address ~self:name ~is_rec:true fn
              in
              let body = go body in
              Llet { name = binder; e = tuple; body }
        | Tail_join | Nontail_join ->
            let name = Join.of_core_ident name in
            let params = transl_params ~mtype_defs ~type_defs fn.params in
            let join_body = go fn.body in
            let body = go body in
            let kind : Clam1.join_kind =
              match kind with
              | Tail_join -> Tail_join
              | Nontail_join -> Nontail_join
              | _ -> assert false
            in
            let type_ =
              Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty
            in
            Ljoinlet { name; params; e = join_body; body; kind; type_ })
    | Cexpr_function { func; ty = _ } ->
        let address = Fn_address.fresh (name_hint ^ ".fn" : Stdlib.String.t) in
        let abs_closure_tid =
          handle_abstract_closure_type ~name:None func address
        in
        let closure, _ =
          closure_of_fn ~addr_tbl ~type_defs ~mtype_defs ~object_methods
            ~name_hint ~address ~self:None ~abs_closure_tid func
        in
        Lclosure closure
    | Cexpr_apply { func; args; kind = Join; ty = _; prim = _ } ->
        let name = Join.of_core_ident func in
        expr_list_to_value args (fun args -> Ljoinapply { name; args })
    | Cexpr_apply
        { func; args = core_args; kind = Normal { func_ty }; ty = _; prim } ->
        expr_list_to_value core_args (fun args ->
            match Addr_table.find_opt addr_tbl func with
            | Some (Toplevel { addr; _ }) ->
                let prim : Clam1.intrinsic option =
                  match prim with
                  | Some (Pintrinsic FixedArray_copy) -> (
                      match core_args with
                      | dst :: _ :: src :: _ ->
                          let dst_tid =
                            transl_type_as_named_exn (Mcore.type_of_expr dst)
                          in
                          let src_tid =
                            transl_type_as_named_exn (Mcore.type_of_expr src)
                          in
                          Some (FixedArray_copy { src_tid; dst_tid })
                      | _ -> assert false)
                  | Some (Pintrinsic FixedArray_fill) -> (
                      match core_args with
                      | arr :: _ ->
                          let arr_tid =
                            transl_type_as_named_exn (Mcore.type_of_expr arr)
                          in
                          Some (FixedArray_fill { tid = arr_tid })
                      | _ -> assert false)
                  | Some
                      (Pintrinsic
                        ( Char_to_string | F64_to_string | String_substring
                        | FixedArray_join | FixedArray_iter | FixedArray_iteri
                        | FixedArray_map | FixedArray_fold_left | Iter_map
                        | Iter_iter | Iter_from_array | Iter_take | Iter_reduce
                        | Iter_flat_map | Iter_repeat | Iter_filter
                        | Iter_concat | Array_length | Array_get
                        | Array_unsafe_get | Array_set | Array_unsafe_set )) ->
                      None
                  | Some _ -> assert false
                  | None -> None
                in
                Lapply { fn = StaticFn addr; args; prim }
            | Some (Local (addr, self_ty)) ->
                let self = Ident.of_core_ident func ~ty:self_ty in
                Lapply
                  { fn = StaticFn addr; args = Vvar self :: args; prim = None }
            | None ->
                let fn = Ident.of_core_ident ~ty:(transl_type func_ty) func in
                Lapply { fn = Dynamic fn; args; prim = None })
    | Cexpr_object { self; methods_key; ty = _ } ->
        let ({ trait; type_ } : Object_util.object_key) = methods_key in
        let tid = Tid.concrete_object_type ~trait ~type_name:type_ in
        let methods = Object_util.Hash.find_exn object_methods methods_key in
        bind self (fun self ->
            Lclosure { tid; address = Object methods; captures = [ self ] })
    | Cexpr_letrec { bindings; body; ty = _ } ->
        let addresses =
          Lst.map bindings (fun (name, fn) ->
              let address =
                Fn_address.fresh
                  (append_name_hint name ^ ".fn" : Stdlib.String.t)
              in
              if Core_ident.Hashset.mem local_non_well_knowns name then
                let abs_closure_tid =
                  handle_abstract_closure_type ~name:(Some name) fn address
                in
                (address, Some abs_closure_tid)
              else
                let ty = Ltype.Ref { tid = Tid.capture_of_function address } in
                Addr_table.add_local_fn_addr_and_type addr_tbl name address ty;
                (address, None))
        in
        let fns, names =
          Lst.split_map2 bindings addresses
            (fun (name, fn) (address, abs_closure_tid_opt) ->
              if Core_ident.Hashset.mem local_non_well_knowns name then
                let abs_closure_tid = Option.get abs_closure_tid_opt in
                let closure, ty =
                  closure_of_fn ~addr_tbl ~type_defs ~mtype_defs ~object_methods
                    ~address ~name_hint:(append_name_hint name)
                    ~self:(Some name) ~abs_closure_tid fn
                in
                let self = Ident.of_core_ident name ~ty in
                (closure, self)
              else
                well_known_closure_of_mut_rec_fn ~addr_tbl ~type_defs
                  ~mtype_defs ~object_methods ~address
                  ~name_hint:(append_name_hint name) ~self:name fn)
        in
        Lletrec { names; fns; body = go body }
    | Cexpr_constr { constr = _; tag; args; ty } ->
        let tid = transl_constr_type tag ty in
        expr_list_to_value args (fun fields ->
            Lallocate { tid; fields; kind = Enum { tag } })
    | Cexpr_tuple { exprs; ty } ->
        let tid = transl_type_as_named_exn ty in
        expr_list_to_value exprs (fun fields ->
            Lallocate { tid; fields; kind = Tuple })
    | Cexpr_record { fields; ty } ->
        if fields = [] then clam_unit
        else
          let args = Lst.map fields (fun { expr; _ } -> expr) in
          let tid = transl_type_as_named_exn ty in
          expr_list_to_value args (fun fields ->
              Lallocate { tid; fields; kind = Struct })
    | Cexpr_record_update { record; fields; fields_num; ty } ->
        let t = transl_type_as_named_exn ty in
        let ty_fields =
          Ltype_util.get_fields t type_defs.defs |> Array.of_list
        in
        bind record (fun record_id ->
            let rec set_new_fields (prev_fields_rev : Clam1.value list) i =
              if i >= fields_num then
                Clam1.Lallocate
                  { tid = t; kind = Struct; fields = List.rev prev_fields_rev }
              else
                match Lst.find_first fields (fun { pos; _ } -> pos = i) with
                | Some { expr; _ } ->
                    to_value expr (fun v ->
                        set_new_fields (v :: prev_fields_rev) (i + 1))
                | None ->
                    let ty_field = ty_fields.(i) in
                    let tmp =
                      Ident.fresh ("field_" ^ string_of_int i) ~ty:ty_field
                    in
                    Llet
                      {
                        name = tmp;
                        e =
                          Lget_field
                            {
                              kind = Struct;
                              obj = record_id;
                              tid = t;
                              index = i;
                            };
                        body =
                          set_new_fields (Vvar tmp :: prev_fields_rev) (i + 1);
                      }
            in
            set_new_fields [] 0)
    | Cexpr_field { record; accessor; pos; ty = _ } ->
        to_value record (fun obj ->
            let t = transl_type_as_named_exn (Mcore.type_of_expr record) in
            match[@warning "-fragile-match"] obj with
            | Clam1.Vvar obj ->
                let kind : Clam1.aggregate_kind =
                  match accessor with
                  | Newtype -> assert false
                  | Label _ -> Struct
                  | Index _ -> Tuple
                in
                Lget_field { kind; obj; index = pos; tid = t }
            | _ -> assert false)
    | Cexpr_mutate { record; label = _; field; pos; ty = _ } ->
        let t = transl_type_as_named_exn (Mcore.type_of_expr record) in
        to_value record (fun record ->
            to_value field (fun field ->
                match[@warning "-fragile-match"] record with
                | Clam1.Vvar obj ->
                    Lset_field
                      { kind = Struct; obj; field; index = pos; tid = t }
                | _ -> assert false))
    | Cexpr_array { exprs; ty } ->
        expr_list_to_value exprs (fun elems ->
            Lmake_array
              {
                kind = EverySingleElem;
                tid = transl_type_as_named_exn ty;
                elems;
              })
    | Cexpr_assign { var; expr; ty = _ } ->
        let var =
          Ident.of_core_ident var ~ty:(transl_type (Mcore.type_of_expr expr))
        in
        Lassign { var; e = go expr }
    | Cexpr_sequence { expr1; expr2; ty = _ } ->
        Lsequence
          {
            expr1 = go expr1;
            expr2 = go expr2;
            expr1_type_ = transl_type (Mcore.type_of_expr expr1);
          }
    | Cexpr_if { cond; ifso; ifnot; ty } -> (
        let pred = go cond in
        let ifso = go ifso in
        match ifnot with
        | Some ifnot ->
            let type_ =
              Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty
            in
            Lif { pred; ifso; ifnot = go ifnot; type_ }
        | None ->
            Lif { pred; ifso; ifnot = clam_unit; type_ = Ret_single I32_Unit })
    | Cexpr_handle_error { obj; handle_kind; ty } ->
        let e = go obj in
        let ok_lty, err_lty =
          match
            Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
              (Mcore.type_of_expr obj)
          with
          | Ret_error { ok_ty; err_ty } -> (ok_ty, err_ty)
          | Ret_single _ -> assert false
        in
        let ok_var = Ident.fresh ~ty:ok_lty "*ok" in
        let ok_branch : Clam1.lambda =
          match handle_kind with
          | To_result ->
              let tag = Tag.of_core_tag_no_ext Builtin.constr_ok.cs_tag in
              let tid = transl_constr_type tag ty in
              Lallocate { kind = Enum { tag }; tid; fields = [ Vvar ok_var ] }
          | Joinapply _ | Return_err _ -> Lvar { var = ok_var }
        in
        let err_var = Ident.fresh ~ty:err_lty "*err" in
        let err_branch : Clam1.lambda =
          match handle_kind with
          | To_result ->
              let tag = Tag.of_core_tag_no_ext Builtin.constr_err.cs_tag in
              let tid = transl_constr_type tag ty in
              Lallocate { kind = Enum { tag }; tid; fields = [ Vvar err_var ] }
          | Joinapply err_join ->
              Ljoinapply
                { name = Join.of_core_ident err_join; args = [ Vvar err_var ] }
          | Return_err { ok_ty = ctx_ok_ty } ->
              let ctx_ok_ty =
                Transl_mtype.mtype_to_ltype ~mtype_defs ~type_defs ctx_ok_ty
              in
              let tag = Tag.of_core_tag_no_ext Builtin.constr_err.cs_tag in
              Lreturn
                (Lmake_multi_result
                   {
                     value = Vvar err_var;
                     tag;
                     type_ = Ret_error { ok_ty = ctx_ok_ty; err_ty = err_lty };
                   })
        in
        Lhandle_error
          {
            obj = e;
            ok_branch = (ok_var, ok_branch);
            err_branch = (err_var, err_branch);
            type_ = Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty;
          }
    | Cexpr_switch_constr { obj; cases; default; ty } ->
        let obj_ty = Mcore.type_of_expr obj in
        bind obj (fun obj_id ->
            let transl_action tag binder body : Clam1.lambda =
              match binder with
              | None -> go body
              | Some binder ->
                  let constr_tid = transl_constr_type tag obj_ty in
                  let constr_ty : Ltype.t = Ref { tid = constr_tid } in
                  Clam1.Llet
                    {
                      name = Ident.of_core_ident binder ~ty:constr_ty;
                      e =
                        Lcast
                          {
                            expr = Lvar { var = obj_id };
                            target_type = constr_ty;
                          };
                      body = go body;
                    }
                [@@inline]
            in
            let cases, default =
              Lst.fold_right cases
                ([], Option.map go default)
                (fun (tag, binder, action) (cases, default) ->
                  let action = transl_action tag binder action in
                  match default with
                  | Some _ -> ((tag, action) :: cases, default)
                  | None -> (cases, Some action))
            in
            match (cases, default) with
            | [], Some default -> default
            | [], None -> Lprim { fn = Praise; args = [] }
            | _ :: _, None -> assert false
            | _, Some default ->
                let type_ =
                  Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty
                in
                Lswitch { obj = obj_id; cases; default; type_ })
    | Cexpr_switch_constant { obj; cases; default; ty } ->
        let type_ =
          Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty
        in
        bind obj (fun obj ->
            match cases with
            | (C_string _, _) :: _ ->
                let cases =
                  Lst.map cases (fun (c, action) ->
                      match c with
                      | C_string s -> (s, go action)
                      | _ -> assert false)
                in
                Lswitchstring { obj; cases; default = go default; type_ }
            | (C_int _, _) :: _ ->
                let cases =
                  Lst.map cases (fun (c, action) ->
                      match c with
                      | C_int { v; repr = _ } -> (Int32.to_int v, go action)
                      | _ -> assert false)
                in
                Lswitchint { obj; cases; default = go default; type_ }
            | (c, _) :: _ ->
                let default = go default in
                let equal =
                  match c with
                  | C_char _ -> Primitive.equal_char
                  | C_int _ -> Primitive.equal_int
                  | C_int64 _ -> Primitive.equal_int64
                  | C_uint _ -> Primitive.equal_uint
                  | C_uint64 _ -> Primitive.equal_uint64
                  | C_float _ -> Primitive.equal_float
                  | C_double _ -> Primitive.equal_float64
                  | C_bool _ -> Primitive.equal_bool
                  | C_bigint _ -> assert false
                  | C_string _ -> assert false
                  | C_bytes _ -> assert false
                in
                Lst.fold_right cases default (fun (c, action) rest ->
                    let pred : Clam1.lambda =
                      Lprim { fn = equal; args = [ Vvar obj; Vconst c ] }
                    in
                    Lif { pred; ifso = go action; ifnot = rest; type_ })
            | [] -> go default)
    | Cexpr_loop { params; body; args; label; ty } ->
        let params = transl_params ~mtype_defs ~type_defs params in
        let body = go body in
        let type_ =
          Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ty
        in
        expr_list_to_value args (fun args ->
            Lloop { params; body; args; label; type_ })
    | Cexpr_break { arg; label; ty = _ } -> (
        match arg with
        | None -> Lbreak { arg = None; label }
        | Some arg -> to_value arg (fun arg -> Lbreak { arg = Some arg; label })
        )
    | Cexpr_continue { args; label; ty = _ } ->
        expr_list_to_value args (fun args -> Lcontinue { args; label })
    | Cexpr_return { expr; return_kind; ty = _ } -> (
        match return_kind with
        | Single_value -> Lreturn (go expr)
        | Error_result { is_error; return_ty } ->
            let tag =
              Tag.of_core_tag_no_ext
                (if is_error then Builtin.constr_err.cs_tag
                 else Builtin.constr_ok.cs_tag)
            in
            let type_ =
              Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs
                return_ty
            in
            to_value expr (fun value ->
                Lreturn (Lmake_multi_result { value; tag; type_ })))
  in
  if !Basic_config.debug then
    let expr_loc = Mcore.loc_of_expr x in
    Clam1.event ~loc_:(Rloc.to_loc ~base:!base expr_loc) expr
  else expr

and closure_of_fn ~mtype_defs ~addr_tbl ~type_defs ~object_methods ~name_hint
    ~(address : Fn_address.t) ~(self : Core_ident.t option)
    ~(abs_closure_tid : Tid.t) (fn : Mcore.fn) : Clam1.closure * Ltype.t =
  let exclude =
    match self with
    | Some self -> Core_ident.Set.singleton self
    | None -> Core_ident.Set.empty
  in
  let fvs =
    Core_ident.Map.fold (Mcore_util.free_vars ~exclude fn) [] (fun id mty acc ->
        transl_ident id mty ~addr_tbl ~mtype_defs ~type_defs :: acc)
  in
  let params = transl_params ~mtype_defs ~type_defs fn.params in
  let return_type_ =
    Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
      (Mcore.type_of_expr fn.body)
  in
  let abs_closure_ty = Ltype.Ref { tid = abs_closure_tid } in
  match (fvs, self) with
  | [], None ->
      let env_id = Ident.fresh ~ty:abs_closure_ty "*env" in
      let body =
        transl_expr ~mtype_defs ~addr_tbl ~name_hint ~type_defs ~object_methods
          fn.body
      in
      let fn : Clam1.fn = { params = env_id :: params; body; return_type_ } in
      new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
      ( { captures = fvs; address = Normal address; tid = abs_closure_tid },
        abs_closure_ty )
  | _ ->
      let closure_cap_tid = Tid.capture_of_function address in
      let closure_type_def =
        Ltype.Ref_closure
          {
            fn_sig_tid = abs_closure_tid;
            captures = Lst.map fvs Ident.get_type;
          }
      in
      let closure_ty = Ltype.Ref { tid = closure_cap_tid } in
      Tid.Hash.add type_defs.defs closure_cap_tid closure_type_def;
      let env_id =
        match self with
        | Some self -> Ident.of_core_ident self ~ty:abs_closure_ty
        | None -> Ident.fresh ~ty:abs_closure_ty "*env"
      in
      let casted_env_id = Ident.fresh ~ty:closure_ty "*casted_env" in
      let env = Clam1.Lvar { var = env_id } in
      let body =
        transl_expr ~mtype_defs ~addr_tbl ~name_hint ~type_defs ~object_methods
          fn.body
      in
      let body_with_captures =
        Lst.fold_left_with_offset fvs body 0 (fun fv rest index ->
            let rhs =
              Clam1.Lclosure_field
                { obj = casted_env_id; index; tid = closure_cap_tid }
            in
            Clam1.Llet { name = fv; e = rhs; body = rest })
      in
      let body : Clam1.lambda =
        Llet
          {
            name = casted_env_id;
            e = Lcast { expr = env; target_type = closure_ty };
            body = body_with_captures;
          }
      in
      let fn : Clam1.fn = { params = env_id :: params; body; return_type_ } in
      new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
      ( { captures = fvs; address = Normal address; tid = closure_cap_tid },
        abs_closure_ty )

and well_known_closure_of_fn ~mtype_defs ~addr_tbl ~type_defs ~object_methods
    ~name_hint ~(address : Fn_address.t) ~(self : Core_ident.t) ~(is_rec : bool)
    (fn : Mcore.fn) : Clam1.lambda * Ident.t =
  let params = transl_params ~mtype_defs ~type_defs fn.params in
  let return_type_ =
    Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
      (Mcore.type_of_expr fn.body)
  in
  let exclude =
    if is_rec then Core_ident.Set.singleton self else Core_ident.Set.empty
  in
  let fvs =
    Core_ident.Map.fold (Mcore_util.free_vars ~exclude fn) [] (fun id mty acc ->
        transl_ident id mty ~addr_tbl ~mtype_defs ~type_defs :: acc)
  in
  let insert_let_self_for_recursive_case self rhs ty body =
    if is_rec then
      let self = Ident.of_core_ident ~ty self in
      Clam1.Llet { name = self; e = rhs; body }
    else body
      [@@inline]
  in
  let update_type_table ty =
    Addr_table.add_local_fn_addr_and_type addr_tbl self address ty
      [@@inline]
  in
  match fvs with
  | [] ->
      let env = Ident.fresh ~ty:I32_Unit "*env" in
      Addr_table.add_local_fn_addr_and_type addr_tbl self address I32_Unit;
      let body =
        transl_expr ~mtype_defs ~addr_tbl ~name_hint ~type_defs ~object_methods
          fn.body
      in
      let body =
        insert_let_self_for_recursive_case self clam_unit I32_Unit body
      in
      let fn : Clam1.fn = { params = env :: params; body; return_type_ } in
      new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
      (clam_unit, Ident.of_core_ident ~ty:I32_Unit self)
  | fv :: [] ->
      let fv_ty = Ident.get_type fv in
      let fv_var = Clam1.Lvar { var = fv } in
      update_type_table fv_ty;
      let body =
        transl_expr ~mtype_defs ~addr_tbl ~name_hint ~type_defs ~object_methods
          fn.body
      in
      let body =
        if is_rec then
          let self = Ident.of_core_ident ~ty:I32_Unit self in
          fix_single_var ~self ~replace:fv body
        else body
      in
      let fn : Clam1.fn = { params = fv :: params; body; return_type_ } in
      new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
      (fv_var, Ident.of_core_ident self ~ty:fv_ty)
  | _ :: _ :: _ ->
      let ty_tuple_def =
        Ltype.Ref_struct
          { fields = Lst.map fvs (fun id -> (Ident.get_type id, false)) }
      in
      let ty_tuple_tid = Tid.capture_of_function address in
      Basic_ty_ident.Hash.add type_defs.defs ty_tuple_tid ty_tuple_def;
      let ty_tuple = Ltype.Ref { tid = ty_tuple_tid } in
      let env_id = Ident.fresh ~ty:ty_tuple "*env" in
      let env = Clam1.Lvar { var = env_id } in
      update_type_table ty_tuple;
      let body =
        transl_expr ~mtype_defs ~addr_tbl ~name_hint ~type_defs ~object_methods
          fn.body
      in
      let body =
        Lst.fold_left_with_offset fvs body 0 (fun fv rest i ->
            let rhs =
              Clam1.Lget_field
                { kind = Tuple; obj = env_id; index = i; tid = ty_tuple_tid }
            in
            Clam1.Llet { name = fv; e = rhs; body = rest })
      in
      let body = insert_let_self_for_recursive_case self env ty_tuple body in
      let fn : Clam1.fn = { params = env_id :: params; body; return_type_ } in
      new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
      ( Lallocate
          {
            tid = ty_tuple_tid;
            fields = Lst.map fvs (fun fv -> Clam1.Vvar fv);
            kind = Struct;
          },
        Ident.of_core_ident ~ty:ty_tuple self )

and well_known_closure_of_mut_rec_fn ~mtype_defs ~addr_tbl ~type_defs
    ~object_methods ~self ~name_hint ~address (fn : Mcore.fn) :
    Clam1.closure * Ident.t =
  let fvs =
    Core_ident.Map.fold
      (Mcore_util.free_vars ~exclude:(Core_ident.Set.singleton self) fn)
      []
      (fun id mty acc ->
        transl_ident id mty ~addr_tbl ~mtype_defs ~type_defs :: acc)
  in
  let params = transl_params ~mtype_defs ~type_defs fn.params in
  let return_type_ =
    Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
      (Mcore.type_of_expr fn.body)
  in
  let tid = Tid.capture_of_function address in
  let closure_type =
    Ltype.Ref_late_init_struct { fields = Lst.map fvs Ident.get_type }
  in
  Tid.Hash.add type_defs.defs tid closure_type;
  let name_hint =
    (name_hint ^ "." ^ Core_ident.base_name self : Stdlib.String.t)
  in
  let ty : Ltype.t = Ref { tid } in
  let self = Ident.of_core_ident ~ty self in
  let body =
    transl_expr ~mtype_defs ~addr_tbl ~type_defs ~name_hint ~object_methods
      fn.body
  in
  let env_id = Ident.fresh ~ty "*env" in
  let body =
    Lst.fold_left_with_offset fvs body 0 (fun fv rest i ->
        let rhs =
          Clam1.Lget_field { obj = env_id; index = i; tid; kind = Tuple }
        in
        Clam1.Llet { name = fv; e = rhs; body = rest })
  in
  let body = fix_single_var ~self ~replace:env_id body in
  let fn : Clam1.fn = { params = env_id :: params; body; return_type_ } in
  new_top { binder = address; fn_kind_ = Clam1.Top_private; fn };
  ({ captures = fvs; address = Well_known_mut_rec; tid }, self)

let make_top_fn_item ~mtype_defs ~addr_tbl ~type_defs ~object_methods
    (fn_address : Fn_address.t) params
    (body : [ `Clam1 of Clam1.lambda | `Core of Mcore.expr ]) return_type_
    ~export_info_ : Clam1.top_func_item =
  let fn_kind_ =
    match export_info_ with
    | Some export_name -> Clam1.Top_pub export_name
    | None -> Clam1.Top_private
  in
  let fn_body =
    match body with
    | `Core body ->
        transl_expr ~addr_tbl ~type_defs
          ~name_hint:(Fn_address.to_string fn_address)
          ~mtype_defs ~object_methods body
    | `Clam1 body -> body
  in
  {
    binder = fn_address;
    fn_kind_;
    fn = { params; body = fn_body; return_type_ };
  }

let make_top_closure_item (addr : Fn_address.t) (params : Ident.t list)
    (return_type_ : Ltype.return_type) ~(closure_ty : Ltype.t) :
    Clam1.top_func_item =
  let closure_address = Fn_address.make_closure_wrapper addr in
  let closure_body =
    let args = Lst.map params (fun p -> Clam1.Vvar p) in
    Clam1.Lapply { fn = Clam1.StaticFn addr; args; prim = None }
  in
  let env_id = Ident.fresh ~ty:closure_ty "*env" in
  {
    binder = closure_address;
    fn_kind_ = Top_private;
    fn = { params = env_id :: params; body = closure_body; return_type_ };
  }

let make_object_wrapper ~mtype_defs ~type_defs ~addr_tbl ~object_methods
    ~abstract_obj_tid ~concrete_obj_tid ~self_lty ~trait
    ({ method_id; method_prim; method_ty } : Object_util.object_method_item) =
  let params_ty, ret_ty =
    match method_ty with
    | T_func { params; return } -> (params, return)
    | T_char | T_bool | T_int | T_uint | T_unit | T_byte | T_string | T_bytes
    | T_tuple _ | T_fixedarray _ | T_trait _ | T_constr _ | T_int64 | T_uint64
    | T_float | T_double | T_any _ | T_optimized_option _ | T_maybe_uninit _
    | T_error_value_result _ ->
        assert false
  in
  match[@warning "-fragile-match"] params_ty with
  | self_ty :: params_ty -> (
      let self_id = Core_ident.fresh "*self" in
      let params_id = Lst.map params_ty (fun _ -> Core_ident.fresh "*param") in
      let body =
        let args =
          Mcore.var ~prim:None ~ty:self_ty self_id
          :: Lst.map2 params_id params_ty (fun id ty ->
                 Mcore.var ~prim:None ~ty id)
        in
        match method_prim with
        | Some (Pintrinsic _) | None ->
            Mcore.apply ~prim:method_prim ~ty:ret_ty
              ~kind:(Normal { func_ty = method_ty })
              method_id args
        | Some prim -> Mcore.prim ~ty:ret_ty prim args
      in
      let body =
        transl_expr ~name_hint:"" ~mtype_defs ~addr_tbl ~type_defs
          ~object_methods body
      in
      let self_id = Ident.of_core_ident ~ty:self_lty self_id in
      let obj_ty : Ltype.t = Ref { tid = abstract_obj_tid } in
      let obj_id = Ident.fresh ~ty:obj_ty "*obj" in
      let casted_obj_ty : Ltype.t = Ref { tid = concrete_obj_tid } in
      let casted_obj_id = Ident.fresh ~ty:casted_obj_ty "*casted_obj" in
      let body : Clam1.lambda =
        Llet
          {
            name = casted_obj_id;
            e =
              Lcast
                { expr = Lvar { var = obj_id }; target_type = casted_obj_ty };
            body =
              Llet
                {
                  name = self_id;
                  e =
                    Lclosure_field
                      { obj = casted_obj_id; tid = concrete_obj_tid; index = 0 };
                  body;
                };
          }
      in
      let fn : Clam1.fn =
        {
          params =
            obj_id
            :: Lst.map2 params_ty params_id (fun ty id ->
                   let ty =
                     Transl_mtype.mtype_to_ltype ~type_defs ~mtype_defs ty
                   in
                   Ident.of_core_ident ~ty id);
          return_type_ =
            Transl_mtype.mtype_to_ltype_return ~type_defs ~mtype_defs ret_ty;
          body;
        }
      in
      match[@warning "-fragile-match"] method_id with
      | Pdot qual_name ->
          let addr = Fn_address.make_object_wrapper qual_name ~trait in
          new_top { binder = addr; fn_kind_ = Top_private; fn };
          addr
      | _ -> assert false)
  | _ -> assert false

let transl_top_func ~(mtype_defs : Mtype.defs) ~(addr_tbl : Addr_table.t)
    ~type_defs ~object_methods (func : Mcore.fn) (binder : Core_ident.t) is_pub_
    =
  match[@warning "-fragile-match"] Addr_table.find_exn addr_tbl binder with
  | Toplevel { addr; params; return; _ } ->
      let fn_item =
        make_top_fn_item ~mtype_defs ~addr_tbl ~type_defs ~object_methods addr
          params (`Core func.body) return ~export_info_:is_pub_
      in
      new_top fn_item
  | _ -> assert false

let sequence (a : Clam1.lambda) (b : Clam1.lambda) ~expr1_type_ =
  if Basic_prelude.phys_equal (Clam1_util.no_located b) clam_unit then a
  else if Basic_prelude.phys_equal (Clam1_util.no_located a) clam_unit then b
  else Lsequence { expr1 = a; expr2 = b; expr1_type_ }

type translate_result = {
  globals : (Ident.t * Constant.t option) list;
  init : Clam1.lambda;
  test : Clam1.lambda;
}

let transl_top_item ~(mtype_defs : Mtype.defs) ~(addr_tbl : Addr_table.t)
    ~(type_defs : Ltype.type_defs_with_context) ~object_methods
    (top : Mcore.top_item) (acc : translate_result) : translate_result =
  match top with
  | Ctop_expr { expr; loc_ } ->
      let name_hint = "*init*" in
      base := loc_;
      let clam_expr =
        transl_expr ~mtype_defs ~name_hint ~addr_tbl ~type_defs ~object_methods
          expr
      in
      let expr1_type_ =
        Transl_mtype.mtype_to_ltype (Mcore.type_of_expr expr) ~type_defs
          ~mtype_defs
      in
      { acc with init = sequence clam_expr acc.init ~expr1_type_ }
  | Ctop_let { binder; expr; is_pub_ = _; loc_ } -> (
      base := loc_;
      match expr with
      | Cexpr_function { func; ty = _ } ->
          transl_top_func ~mtype_defs ~addr_tbl ~type_defs ~object_methods func
            binder None;
          acc
      | _ -> (
          let ty = Mcore.type_of_expr expr in
          let name =
            Ident.of_core_ident
              ~ty:(Transl_mtype.mtype_to_ltype ~mtype_defs ~type_defs ty)
              binder
          in
          let expr =
            transl_expr ~name_hint:(Ident.to_string name) ~mtype_defs ~addr_tbl
              ~type_defs ~object_methods expr
          in
          match Clam1_util.no_located expr with
          | Lconst
              ((C_bool _ | C_char _ | C_int _ | C_int64 _ | C_double _) as c) ->
              { acc with globals = (name, Some c) :: acc.globals }
          | _ ->
              {
                acc with
                globals = (name, None) :: acc.globals;
                init = Llet { name; e = expr; body = acc.init };
              }))
  | Ctop_fn { binder; func; export_info_; loc_ } ->
      base := loc_;
      transl_top_func ~mtype_defs ~addr_tbl ~type_defs ~object_methods func
        binder export_info_;
      acc
  | Ctop_stub { binder; func_stubs; params_ty; return_ty; export_info_; loc_ }
    -> (
      base := loc_;
      match[@warning "-fragile-match"] Addr_table.find_exn addr_tbl binder with
      | Toplevel { addr; params; return; _ } ->
          let params_ltype =
            Lst.map2 params params_ty (fun id ty : Ltype.t ->
                if Mtype.is_func ty then Ref_extern else Ident.get_type id)
          in
          let return_ltype =
            match return_ty with
            | None -> None
            | Some _ -> (
                match[@warning "-fragile-match"] return with
                | Ret_single return_ltype -> Some return_ltype
                | _ -> assert false)
          in
          let rec make_body args_rev params params_ty : Clam1.lambda =
            match (params, params_ty) with
            | [], [] ->
                let args = List.rev args_rev in
                Levent
                  {
                    expr =
                      Lstub_call
                        {
                          fn = func_stubs;
                          args;
                          params_ty = params_ltype;
                          return_ty = return_ltype;
                        };
                    loc_;
                  }
            | param :: params, ty :: params_ty ->
                if Mtype.is_func ty then
                  let tmp = Ident.fresh "extern" ~ty:(Ident.get_type param) in
                  Llet
                    {
                      name = tmp;
                      e =
                        Lprim
                          { fn = Pclosure_to_extern_ref; args = [ Vvar param ] };
                      body = make_body (Vvar tmp :: args_rev) params params_ty;
                    }
                else make_body (Vvar param :: args_rev) params params_ty
            | _ -> assert false
          in
          let body : Clam1.lambda = make_body [] params params_ty in
          let fn_item =
            make_top_fn_item ~mtype_defs ~addr_tbl ~type_defs ~object_methods
              addr params (`Clam1 body) return ~export_info_
          in
          new_top fn_item;
          acc
      | _ -> assert false)

let collect_top_func ~type_defs ~mtype_defs (item : Mcore.top_item)
    ~(addr_tbl : Addr_table.t) =
  match item with
  | Ctop_expr _ -> ()
  | Ctop_let { binder; expr = Cexpr_function { func; _ } }
  | Ctop_fn { binder; func; _ } ->
      let return =
        Transl_mtype.mtype_to_ltype_return ~mtype_defs ~type_defs
          (Mcore.type_of_expr func.body)
      in
      let params = transl_params ~mtype_defs func.params ~type_defs in
      Addr_table.add_toplevel_fn addr_tbl binder ~params ~return
  | Ctop_stub { binder; params_ty; return_ty; _ } ->
      let params =
        Lst.map params_ty (fun ty ->
            Ident.fresh "*param"
              ~ty:(Transl_mtype.mtype_to_ltype ty ~type_defs ~mtype_defs))
      in
      let return_ltype =
        match return_ty with
        | Some return_ty ->
            Transl_mtype.mtype_to_ltype ~type_defs ~mtype_defs return_ty
        | None -> I32_Unit
      in
      Addr_table.add_toplevel_fn addr_tbl binder ~params
        ~return:(Ret_single return_ltype)
  | Ctop_let _ -> ()

let non_well_knowns_obj =
  object
    inherit [_] Mcore.Iter.iter

    method! visit_Cexpr_var (ctx : Core_ident.Hashset.t) id _ty _prim _loc =
      Core_ident.Hashset.add ctx id
  end

let collect_local_non_well_knowns (ctx : Core_ident.Hashset.t) (prog : Mcore.t)
    =
  Lst.iter prog.body (non_well_knowns_obj#visit_top_item ctx);
  match prog.main with
  | None -> ()
  | Some (main, _) -> non_well_knowns_obj#visit_expr ctx main

let transl_prog ({ body; main; types; object_methods = _ } as prog : Mcore.t) :
    Clam1.prog =
  binds_init := [];
  Ident_hashset.reset local_non_well_knowns;
  let type_defs = Transl_mtype.transl_mtype_defs types in
  collect_local_non_well_knowns local_non_well_knowns prog;
  let addr_tbl = Addr_table.create 17 in
  Lst.iter body (collect_top_func ~type_defs ~mtype_defs:types ~addr_tbl);
  let object_methods = Object_util.Hash.create 17 in
  Object_util.Hash.iter2 prog.object_methods
    (fun obj_key { self_ty; methods } ->
      let ({ trait; type_ } : Object_util.object_key) = obj_key in
      let abstract_obj_tid = Tid.of_type_path trait in
      let concrete_obj_tid = Tid.concrete_object_type ~trait ~type_name:type_ in
      let self_lty =
        Transl_mtype.mtype_to_ltype ~type_defs ~mtype_defs:types self_ty
      in
      Tid.Hash.add type_defs.defs concrete_obj_tid
        (Ref_closure { fn_sig_tid = abstract_obj_tid; captures = [ self_lty ] });
      let addrs =
        Lst.map methods
          (make_object_wrapper ~type_defs ~mtype_defs:types ~addr_tbl
             ~object_methods ~abstract_obj_tid ~concrete_obj_tid ~self_lty
             ~trait)
      in
      Object_util.Hash.add object_methods obj_key addrs);
  (match main with
  | Some (main, loc_) ->
      collect_top_func ~type_defs ~mtype_defs:types
        (Ctop_expr { expr = main; loc_ })
        ~addr_tbl
  | None -> ());
  let acc = { globals = []; init = clam_unit; test = clam_unit } in
  let main =
    match main with
    | None -> None
    | Some (main, loc_) ->
        base := loc_;
        Some
          (transl_expr ~mtype_defs:types ~name_hint:"*main*" ~addr_tbl
             ~type_defs ~object_methods main)
  in
  let { globals; init; test } =
    List.fold_right
      (transl_top_item ~mtype_defs:types ~addr_tbl ~type_defs ~object_methods)
      body acc
  in
  let start = sequence init test ~expr1_type_:I32_Unit in
  let globals, init, fns =
    Addr_table.fold addr_tbl (globals, start, !binds_init)
      (fun _fn_binder fn_info (globals, start, fns) ->
        match fn_info with
        | Toplevel { name_as_closure = Some name; params; return; addr } -> (
            let closure_ty : Ltype.t =
              match Ident.get_type name with
              | Ref_any ->
                  let sig_ : Ltype.fn_sig =
                    { params = Lst.map params Ident.get_type; ret = return }
                  in
                  Ref
                    { tid = Ltype.FnSigHash.find_exn type_defs.fn_sig_tbl sig_ }
              | ty -> ty
            in
            let closure_item =
              make_top_closure_item addr params return ~closure_ty
            in
            let address = closure_item.binder in
            match[@warning "-fragile-match"] closure_ty with
            | Ref { tid } ->
                let fn_closure =
                  Clam1.Lclosure
                    { captures = []; address = Normal address; tid }
                in
                ( (name, None) :: globals,
                  Clam1.Llet { name; e = fn_closure; body = start },
                  closure_item :: fns )
            | _ -> assert false)
        | Toplevel { name_as_closure = None; _ } | Local _ ->
            (globals, start, fns))
  in
  { fns; init; main; globals; type_defs = type_defs.defs }
