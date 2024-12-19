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


module Tid = Basic_ty_ident
module Ltype = Ltype_gc
module Lst = Basic_lst

let tid_to_wasm_name (tid : Tid.t) = Tid.to_wasm_name tid
let tid_to_string = tid_to_wasm_name

let need_late_init_tid_to_sexp (tid : Tid.t) : W.t =
  List
    (List.cons
       (Atom "ref" : W.t)
       (List.cons (Atom "null" : W.t) ([ Atom (tid_to_string tid) ] : W.t list)))

let tid_to_sexp (tid : Tid.t) : W.t =
  List (List.cons (Atom "ref" : W.t) ([ Atom (tid_to_string tid) ] : W.t list))

let ltype_to_sexp (t : Ltype.t) : W.t =
  match t with
  | Ref_extern -> Atom "externref"
  | F32 -> Atom "f32"
  | F64 -> Atom "f64"
  | I64 -> Atom "i64"
  | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
  | I32_Option_Char ->
      Atom "i32"
  | Ref_bytes ->
      (List
         (List.cons (Atom "ref" : W.t) ([ Atom "$moonbit.bytes" ] : W.t list))
        : W.t)
  | Ref_string ->
      if !Basic_config.use_js_builtin_string then
        (List (List.cons (Atom "ref" : W.t) ([ Atom "extern" ] : W.t list))
          : W.t)
      else
        (List
           (List.cons
              (Atom "ref" : W.t)
              ([ Atom "$moonbit.string" ] : W.t list))
          : W.t)
  | Ref { tid } ->
      (List
         (List.cons
            (Atom "ref" : W.t)
            ([ Atom (tid_to_string tid) ] : W.t list))
        : W.t)
  | Ref_nullable { tid } ->
      if !Basic_config.use_js_builtin_string && Tid.equal tid Ltype.tid_string
      then Atom "externref"
      else
        (List
           (List.cons
              (Atom "ref" : W.t)
              (List.cons
                 (Atom "null" : W.t)
                 ([ Atom (tid_to_string tid) ] : W.t list)))
          : W.t)
  | Ref_lazy_init { tid } ->
      (List
         (List.cons
            (Atom "ref" : W.t)
            (List.cons
               (Atom "null" : W.t)
               ([ Atom (tid_to_string tid) ] : W.t list)))
        : W.t)
  | Ref_func -> Atom "funcref"
  | Ref_any -> (Atom "anyref" : W.t)

let need_late_init_ltype_to_sexp (t : Ltype.t) : W.t =
  match t with
  | Ref_extern -> Atom "externref"
  | F32 -> Atom "f32"
  | F64 -> Atom "f64"
  | I64 -> Atom "i64"
  | I32_Int | I32_Char | I32_Bool | I32_Unit | I32_Byte | I32_Tag
  | I32_Option_Char ->
      Atom "i32"
  | Ref_bytes ->
      (List
         (List.cons
            (Atom "ref" : W.t)
            (List.cons
               (Atom "null" : W.t)
               ([ Atom "$moonbit.bytes" ] : W.t list)))
        : W.t)
  | Ref_string ->
      if !Basic_config.use_js_builtin_string then Atom "externref"
      else
        (List
           (List.cons
              (Atom "ref" : W.t)
              (List.cons
                 (Atom "null" : W.t)
                 ([ Atom "$moonbit.string" ] : W.t list)))
          : W.t)
  | Ref_nullable { tid } ->
      (List
         (List.cons
            (Atom "ref" : W.t)
            (List.cons
               (Atom "null" : W.t)
               ([ Atom (tid_to_string tid) ] : W.t list)))
        : W.t)
  | Ref_lazy_init { tid } | Ref { tid } ->
      (List
         (List.cons
            (Atom "ref" : W.t)
            (List.cons
               (Atom "null" : W.t)
               ([ Atom (tid_to_string tid) ] : W.t list)))
        : W.t)
  | Ref_func -> Atom "funcref"
  | Ref_any -> (Atom "anyref" : W.t)

let compile_typedef ~(type_defs : Ltype.type_defs) (tid : Tid.t)
    (def : Ltype.def) =
  match def with
  | Ref_constructor { args = [] } | Ref_struct { fields = [] } -> []
  | Ref_array { elem } ->
      ([
         List
           (List.cons
              (Atom "type" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 ([
                    List
                      (List.cons
                         (Atom "array" : W.t)
                         ([
                            List
                              (List.cons
                                 (Atom "mut" : W.t)
                                 ([ ltype_to_sexp elem ] : W.t list));
                          ]
                           : W.t list));
                  ]
                   : W.t list)));
       ]
        : W.t list)
  | Ref_struct { fields } ->
      let fields =
        Lst.map fields (fun (field_ty, mut) ->
            if mut then
              (List
                 (List.cons
                    (Atom "field" : W.t)
                    ([
                       List
                         (List.cons
                            (Atom "mut" : W.t)
                            ([ ltype_to_sexp field_ty ] : W.t list));
                     ]
                      : W.t list))
                : W.t)
            else
              (List
                 (List.cons
                    (Atom "field" : W.t)
                    ([ ltype_to_sexp field_ty ] : W.t list))
                : W.t))
      in
      ([
         List
           (List.cons
              (Atom "type" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 ([ List (List.cons (Atom "struct" : W.t) (fields : W.t list)) ]
                   : W.t list)));
       ]
        : W.t list)
  | Ref_late_init_struct { fields } ->
      let fields =
        Lst.map fields (fun p : W.t ->
            List
              (List.cons
                 (Atom "field" : W.t)
                 ([
                    List
                      (List.cons
                         (Atom "mut" : W.t)
                         ([ need_late_init_ltype_to_sexp p ] : W.t list));
                  ]
                   : W.t list)))
      in
      ([
         List
           (List.cons
              (Atom "type" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 ([ List (List.cons (Atom "struct" : W.t) (fields : W.t list)) ]
                   : W.t list)));
       ]
        : W.t list)
  | Ref_constructor { args } ->
      let tid_enum = Ltype.tid_enum in
      let fields =
        Lst.map args (fun (field_ty, mut) ->
            if mut then
              (List
                 (List.cons
                    (Atom "field" : W.t)
                    ([
                       List
                         (List.cons
                            (Atom "mut" : W.t)
                            ([ ltype_to_sexp field_ty ] : W.t list));
                     ]
                      : W.t list))
                : W.t)
            else
              (List
                 (List.cons
                    (Atom "field" : W.t)
                    ([ ltype_to_sexp field_ty ] : W.t list))
                : W.t))
      in
      ([
         List
           (List.cons
              (Atom "type" : W.t)
              (List.cons
                 (Atom (tid_to_string tid) : W.t)
                 ([
                    List
                      (List.cons
                         (Atom "sub" : W.t)
                         (List.cons
                            (Atom "final" : W.t)
                            (List.cons
                               (Atom (tid_to_string tid_enum) : W.t)
                               ([
                                  List
                                    (List.cons
                                       (Atom "struct" : W.t)
                                       (List.cons
                                          (List
                                             (List.cons
                                                (Atom "field" : W.t)
                                                ([ Atom "i32" ] : W.t list))
                                            : W.t)
                                          (fields : W.t list)));
                                ]
                                 : W.t list))));
                  ]
                   : W.t list)));
       ]
        : W.t list)
  | Ref_closure_abstract { fn_sig = { params; ret } } ->
      let tid_code_ptr = Tid.code_pointer_of_closure tid in
      let params =
        Lst.map params (fun p : W.t ->
            List
              (List.cons (Atom "param" : W.t) ([ ltype_to_sexp p ] : W.t list)))
      in
      let result =
        Lst.map ret (fun p : W.t ->
            List
              (List.cons (Atom "result" : W.t) ([ ltype_to_sexp p ] : W.t list)))
      in
      ([
         List
           (List.cons
              (Atom "rec" : W.t)
              (List.cons
                 (List
                    (List.cons
                       (Atom "type" : W.t)
                       (List.cons
                          (Atom (tid_to_string tid_code_ptr) : W.t)
                          ([
                             List
                               (List.cons
                                  (Atom "func" : W.t)
                                  (List.cons
                                     (List
                                        (List.cons
                                           (Atom "param" : W.t)
                                           ([ tid_to_sexp tid ] : W.t list))
                                       : W.t)
                                     (List.append
                                        (params : W.t list)
                                        (result : W.t list))));
                           ]
                            : W.t list)))
                   : W.t)
                 ([
                    List
                      (List.cons
                         (Atom "type" : W.t)
                         (List.cons
                            (Atom (tid_to_string tid) : W.t)
                            ([
                               List
                                 (List.cons
                                    (Atom "sub" : W.t)
                                    ([
                                       List
                                         (List.cons
                                            (Atom "struct" : W.t)
                                            ([
                                               List
                                                 (List.cons
                                                    (Atom "field" : W.t)
                                                    ([
                                                       List
                                                         (List.cons
                                                            (Atom "mut" : W.t)
                                                            ([
                                                               need_late_init_tid_to_sexp
                                                                 tid_code_ptr;
                                                             ]
                                                              : W.t list));
                                                     ]
                                                      : W.t list));
                                             ]
                                              : W.t list));
                                     ]
                                      : W.t list));
                             ]
                              : W.t list)));
                  ]
                   : W.t list)));
       ]
        : W.t list)
  | Ref_object { methods } ->
      let method_tids =
        Lst.mapi methods (fun method_index _ ->
            Tid.method_of_object tid method_index)
      in
      let method_sigs =
        Lst.map2 method_tids methods (fun method_tid { params; ret } ->
            let params =
              Lst.map params (fun p : W.t ->
                  List
                    (List.cons
                       (Atom "param" : W.t)
                       ([ ltype_to_sexp p ] : W.t list)))
            in
            let result =
              Lst.map ret (fun p : W.t ->
                  List
                    (List.cons
                       (Atom "result" : W.t)
                       ([ ltype_to_sexp p ] : W.t list)))
            in
            (List
               (List.cons
                  (Atom "type" : W.t)
                  (List.cons
                     (Atom (tid_to_string method_tid) : W.t)
                     ([
                        List
                          (List.cons
                             (Atom "func" : W.t)
                             (List.cons
                                (List
                                   (List.cons
                                      (Atom "param" : W.t)
                                      ([ tid_to_sexp tid ] : W.t list))
                                  : W.t)
                                (List.append
                                   (params : W.t list)
                                   (result : W.t list))));
                      ]
                       : W.t list)))
              : W.t))
      in
      let method_fields =
        Lst.map method_tids (fun method_tid : W.t ->
            List
              (List.cons
                 (Atom "field" : W.t)
                 ([ tid_to_sexp method_tid ] : W.t list)))
      in
      ([
         List
           (List.cons
              (Atom "rec" : W.t)
              (List.cons
                 (List
                    (List.cons
                       (Atom "type" : W.t)
                       (List.cons
                          (Atom (tid_to_string tid) : W.t)
                          ([
                             List
                               (List.cons
                                  (Atom "sub" : W.t)
                                  ([
                                     List
                                       (List.cons
                                          (Atom "struct" : W.t)
                                          (method_fields : W.t list));
                                   ]
                                    : W.t list));
                           ]
                            : W.t list)))
                   : W.t)
                 (method_sigs : W.t list)));
       ]
        : W.t list)
  | Ref_closure { fn_sig_tid = _; captures = [] } -> []
  | Ref_closure { fn_sig_tid; captures } -> (
      let tid_capture = tid in
      let tid_base = fn_sig_tid in
      let make_result ~need_late_init code_ptr_fields =
        let captures =
          Lst.map captures (fun p ->
              if need_late_init then
                (List
                   (List.cons
                      (Atom "field" : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "mut" : W.t)
                              ([ need_late_init_ltype_to_sexp p ] : W.t list));
                       ]
                        : W.t list))
                  : W.t)
              else
                (List
                   (List.cons
                      (Atom "field" : W.t)
                      ([ ltype_to_sexp p ] : W.t list))
                  : W.t))
        in
        ([
           List
             (List.cons
                (Atom "type" : W.t)
                (List.cons
                   (Atom (tid_to_string tid_capture) : W.t)
                   ([
                      List
                        (List.cons
                           (Atom "sub" : W.t)
                           (List.cons
                              (Atom (tid_to_string tid_base) : W.t)
                              ([
                                 List
                                   (List.cons
                                      (Atom "struct" : W.t)
                                      (List.append
                                         (code_ptr_fields : W.t list)
                                         (captures : W.t list)));
                               ]
                                : W.t list)));
                    ]
                     : W.t list)));
         ]
          : W.t list)
          [@@local]
      in
      match Tid.Hash.find_exn type_defs fn_sig_tid with
      | Ref_struct _ | Ref_late_init_struct _ | Ref_constructor _ | Ref_array _
      | Ref_closure _ ->
          assert false
      | Ref_closure_abstract _ ->
          let tid_code_ptr = Tid.code_pointer_of_closure fn_sig_tid in
          let code_ptr_fields =
            ([
               List
                 (List.cons
                    (Atom "field" : W.t)
                    ([
                       List
                         (List.cons
                            (Atom "mut" : W.t)
                            ([ need_late_init_tid_to_sexp tid_code_ptr ]
                              : W.t list));
                     ]
                      : W.t list));
             ]
              : W.t list)
          in
          make_result ~need_late_init:true code_ptr_fields
      | Ref_object { methods } ->
          let code_ptr_fields =
            Lst.mapi methods (fun method_index _ ->
                let tid_method = Tid.method_of_object fn_sig_tid method_index in
                (List
                   (List.cons
                      (Atom "field" : W.t)
                      ([ tid_to_sexp tid_method ] : W.t list))
                  : W.t))
          in
          make_result ~need_late_init:false code_ptr_fields)

let compile_group_type_defs (type_defs : Ltype.type_defs) =
  let grouped_types = Grouped_typedefs.group_typedefs type_defs in
  Lst.concat_map grouped_types (fun def ->
      match def with
      | Rec defs ->
          let types = ref [] in
          let rec_types = ref [] in
          Lst.iter defs (fun (tid, def) ->
              let instrs = compile_typedef tid def ~type_defs in
              Lst.iter instrs (fun instr ->
                  match instr with
                  | (List (Atom "rec" :: instrs) : W.t) ->
                      rec_types := instrs @ !rec_types
                  | _ -> types := instr :: !types));
          let rec_types = !rec_types in
          let types = List.rev !types in
          ([
             List
               (List.cons
                  (Atom "rec" : W.t)
                  (List.append (rec_types : W.t list) (types : W.t list)));
           ]
            : W.t list)
      | Nonrec (tid, def) -> compile_typedef tid def ~type_defs)
