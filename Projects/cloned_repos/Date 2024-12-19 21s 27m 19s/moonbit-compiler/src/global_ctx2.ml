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


module Hashset_int = Basic_hashset_int
module Tid = Basic_ty_ident
module Ltype = Ltype_gc
module Vec = Basic_vec
module Lst = Basic_lst
module Fn_addr = Basic_fn_address
module Fn_addr_hashset = Fn_addr.Hashset

let make_string_constant_id index : Stdlib.String.t =
  "$moonbit.js_string_constant." ^ Int.to_string index

type default_info = Arr of Tid.t | Enum | Js_string

let string_default_name = "$moonbit.string.default"
let bytes_default_name = "$moonbit.bytes.default"
let enum_default_name = "$moonbit.enum.default"

module Import = struct
  type t = {
    module_name : string;
    func_name : string;
    params_ty : Ltype.t list;
    return_ty : Ltype.t option;
  }

  include struct
    let _ = fun (_ : t) -> ()

    let sexp_of_t =
      (fun {
             module_name = module_name__002_;
             func_name = func_name__004_;
             params_ty = params_ty__006_;
             return_ty = return_ty__008_;
           } ->
         let bnds__001_ = ([] : _ Stdlib.List.t) in
         let bnds__001_ =
           let arg__009_ =
             Moon_sexp_conv.sexp_of_option Ltype.sexp_of_t return_ty__008_
           in
           (S.List [ S.Atom "return_ty"; arg__009_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         let bnds__001_ =
           let arg__007_ =
             Moon_sexp_conv.sexp_of_list Ltype.sexp_of_t params_ty__006_
           in
           (S.List [ S.Atom "params_ty"; arg__007_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         let bnds__001_ =
           let arg__005_ = Moon_sexp_conv.sexp_of_string func_name__004_ in
           (S.List [ S.Atom "func_name"; arg__005_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         let bnds__001_ =
           let arg__003_ = Moon_sexp_conv.sexp_of_string module_name__002_ in
           (S.List [ S.Atom "module_name"; arg__003_ ] :: bnds__001_
             : _ Stdlib.List.t)
         in
         S.List bnds__001_
        : t -> S.t)

    let _ = sexp_of_t

    let (hash_fold_t : Ppx_base.state -> t -> Ppx_base.state) =
     fun hsv arg ->
      let hsv =
        let hsv =
          let hsv =
            let hsv = hsv in
            Ppx_base.hash_fold_string hsv arg.module_name
          in
          Ppx_base.hash_fold_string hsv arg.func_name
        in
        Ppx_base.hash_fold_list Ltype.hash_fold_t hsv arg.params_ty
      in
      Ppx_base.hash_fold_option Ltype.hash_fold_t hsv arg.return_ty

    let _ = hash_fold_t

    let (hash : t -> Ppx_base.hash_value) =
      let func arg =
        Ppx_base.get_hash_value
          (let hsv = Ppx_base.create () in
           hash_fold_t hsv arg)
      in
      fun x -> func x

    let _ = hash

    let equal =
      (fun a__010_ b__011_ ->
         if Stdlib.( == ) a__010_ b__011_ then true
         else
           Stdlib.( && )
             (Stdlib.( = ) (a__010_.module_name : string) b__011_.module_name)
             (Stdlib.( && )
                (Stdlib.( = ) (a__010_.func_name : string) b__011_.func_name)
                (Stdlib.( && )
                   (Ppx_base.equal_list
                      (fun a__012_ b__013_ -> Ltype.equal a__012_ b__013_)
                      a__010_.params_ty b__011_.params_ty)
                   (match (a__010_.return_ty, b__011_.return_ty) with
                   | None, None -> true
                   | None, Some _ -> false
                   | Some _, None -> false
                   | Some __option_x, Some __option_y ->
                       (fun a__014_ b__015_ -> Ltype.equal a__014_ b__015_)
                         __option_x __option_y)))
        : t -> t -> bool)

    let _ = equal
  end

  let make_closure : t =
    {
      module_name = "moonbit:ffi";
      func_name = "make_closure";
      params_ty = [ Ref_func; Ref_any ];
      return_ty = Some Ref_extern;
    }
end

module Import_hash = Basic_hashf.Make (Import)

type t = {
  const_table : Const_table.t;
  tags : Hashset_int.t;
  imports : string Import_hash.t;
  inlines : (string * W.t) Vec.t;
  defaults : (string * default_info) Tid.Hash.t;
  func_refs : Fn_addr_hashset.t;
}

let make_constr_id (tag : int) = W.Atom ("$moonbit.constr/" ^ Int.to_string tag)

let create () =
  {
    const_table = Const_table.create ();
    tags = Hashset_int.create 17;
    imports = Import_hash.create 17;
    inlines = Vec.empty ();
    defaults = Tid.Hash.create 17;
    func_refs = Fn_addr_hashset.create 17;
  }

let i32_to_sexp = Wasm_util.i32_to_sexp
let tid_to_string = Tid.to_wasm_name

let compile_string_literal ~(global_ctx : t) (s : string) : W.t =
  if !Basic_config.use_js_builtin_string then
    let index =
      Const_table.find_js_builtin_str_const global_ctx.const_table s
    in
    let name = make_string_constant_id index in
    (List (List.cons (Atom "global.get" : W.t) ([ Atom name ] : W.t list))
      : W.t)
  else if s = "" then (
    let tid = Ltype.tid_string in
    if not (Tid.Hash.mem global_ctx.defaults tid) then
      Tid.Hash.add global_ctx.defaults tid
        (string_default_name, Arr Ltype.tid_string);
    (List
       (List.cons
          (Atom "global.get" : W.t)
          ([ Atom string_default_name ] : W.t list))
      : W.t))
  else
    let utf16 = Basic_strutil.string_utf16_of_utf8 s in
    let offset, index =
      Const_table.find_str_const global_ctx.const_table utf16
    in
    let len = String.length utf16 asr 1 in
    (List
       (List.cons
          (Atom "call" : W.t)
          (List.cons
             (Atom "$moonbit.string_literal" : W.t)
             (List.cons
                (i32_to_sexp index : W.t)
                (List.cons
                   (i32_to_sexp offset : W.t)
                   ([ i32_to_sexp len ] : W.t list)))))
      : W.t)

let compile_bytes_literal ~(global_ctx : t) (utf8str : string) : W.t =
  let offset = Const_table.find_bytes_const global_ctx.const_table utf8str in
  let len = String.length utf8str in
  (List
     (List.cons
        (Atom "array.new_data" : W.t)
        (List.cons
           (Atom "$moonbit.bytes" : W.t)
           (List.cons
              (Atom "$moonbit.const_data" : W.t)
              (List.cons
                 (i32_to_sexp offset : W.t)
                 ([ i32_to_sexp len ] : W.t list)))))
    : W.t)

let compile_int32_array_literal ~(global_ctx : t) (xs : int32 list)
    (tid : Tid.t) : W.t =
  let buf = Buffer.create 16 in
  let len = List.length xs in
  List.iter (fun x -> Buffer.add_int32_le buf x) xs;
  let s = Buffer.contents buf in
  let offset = Const_table.find_array_const global_ctx.const_table s in
  (List
     (List.cons
        (Atom "array.new_data" : W.t)
        (List.cons
           (Atom (tid_to_string tid) : W.t)
           (List.cons
              (Atom "$moonbit.const_data" : W.t)
              (List.cons
                 (i32_to_sexp offset : W.t)
                 ([ i32_to_sexp len ] : W.t list)))))
    : W.t)

let compile_int64_array_literal ~(global_ctx : t) (xs : int64 list)
    (tid : Tid.t) : W.t =
  let buf = Buffer.create 16 in
  let len = List.length xs in
  List.iter (fun x -> Buffer.add_int64_le buf x) xs;
  let s = Buffer.contents buf in
  let offset = Const_table.find_array_const global_ctx.const_table s in
  (List
     (List.cons
        (Atom "array.new_data" : W.t)
        (List.cons
           (Atom (tid_to_string tid) : W.t)
           (List.cons
              (Atom "$moonbit.const_data" : W.t)
              (List.cons
                 (i32_to_sexp offset : W.t)
                 ([ i32_to_sexp len ] : W.t list)))))
    : W.t)

let compile_constant_constr ~(global_ctx : t) ~(tag : int) : W.t =
  Hashset_int.add global_ctx.tags tag;
  let id = make_constr_id tag in
  (List (List.cons (Atom "global.get" : W.t) ([ id ] : W.t list)) : W.t)

let compile_defaults (global_ctx : t) =
  let default_names = Vec.empty () in
  Tid.Hash.iter global_ctx.defaults (fun (_, (name, elem_tid)) ->
      Vec.push default_names (name, elem_tid));
  Vec.map_into_list default_names (fun (name, default_info) ->
      match default_info with
      | Arr tid ->
          (List
             (List.cons
                (Atom "global" : W.t)
                (List.cons
                   (Atom name : W.t)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "ref" : W.t)
                            ([ Atom (tid_to_string tid) ] : W.t list))
                        : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "array.new_fixed" : W.t)
                              (List.cons
                                 (Atom (tid_to_string tid) : W.t)
                                 ([ Atom "0" ] : W.t list)));
                       ]
                        : W.t list))))
            : W.t)
      | Js_string ->
          let s = "\"\"" in
          let const_string_module_name =
            Basic_strutil.esc_quote !Basic_config.const_string_module_name
          in
          (List
             (List.cons
                (Atom "global" : W.t)
                (List.cons
                   (Atom name : W.t)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "import" : W.t)
                            (List.cons
                               (Atom const_string_module_name : W.t)
                               ([ Atom s ] : W.t list)))
                        : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "ref" : W.t)
                              ([ Atom "extern" ] : W.t list));
                       ]
                        : W.t list))))
            : W.t)
      | Enum ->
          let tid = Ltype.tid_enum in
          (List
             (List.cons
                (Atom "global" : W.t)
                (List.cons
                   (Atom name : W.t)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "ref" : W.t)
                            ([ Atom (tid_to_string tid) ] : W.t list))
                        : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "struct.new" : W.t)
                              (List.cons
                                 (Atom (tid_to_string tid) : W.t)
                                 ([
                                    List
                                      (List.cons
                                         (Atom "i32.const" : W.t)
                                         ([ Atom "-1" ] : W.t list));
                                  ]
                                   : W.t list)));
                       ]
                        : W.t list))))
            : W.t))

let compile_to_globals (global_ctx : t) =
  let constant_constr =
    Hashset_int.fold global_ctx.tags [] (fun i acc ->
        let id = make_constr_id i in
        let tid_enum = Ltype.tid_enum in
        List.cons
          (List
             (List.cons
                (Atom "global" : W.t)
                (List.cons
                   (id : W.t)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "ref" : W.t)
                            ([ Atom (tid_to_string tid_enum) ] : W.t list))
                        : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "struct.new" : W.t)
                              (List.cons
                                 (Atom (tid_to_string tid_enum) : W.t)
                                 ([ i32_to_sexp i ] : W.t list)));
                       ]
                        : W.t list))))
            : W.t)
          (acc : W.t list))
  in
  let defaults = compile_defaults global_ctx in
  if !Basic_config.use_js_builtin_string then (
    let constant_strings = Vec.empty () in
    Const_table.iter_constant_string_with_index global_ctx.const_table
      (fun s i ->
        let name = make_string_constant_id i in
        let esc_quote_hex s = "\"" ^ W.escaped s ^ "\"" in
        let s = esc_quote_hex s in
        let const_string_module_name =
          Basic_strutil.esc_quote !Basic_config.const_string_module_name
        in
        let global =
          (List
             (List.cons
                (Atom "global" : W.t)
                (List.cons
                   (Atom name : W.t)
                   (List.cons
                      (List
                         (List.cons
                            (Atom "import" : W.t)
                            (List.cons
                               (Atom const_string_module_name : W.t)
                               ([ Atom s ] : W.t list)))
                        : W.t)
                      ([
                         List
                           (List.cons
                              (Atom "ref" : W.t)
                              ([ Atom "extern" ] : W.t list));
                       ]
                        : W.t list))))
            : W.t)
        in
        Vec.push constant_strings global);
    let constant_strings = Vec.to_list constant_strings in
    List.append
      (constant_constr : W.t list)
      (List.append (defaults : W.t list) (constant_strings : W.t list)))
  else
    let string_pool_len = Const_table.get_string_count global_ctx.const_table in
    let string_pool =
      (List
         (List.cons
            (Atom "global" : W.t)
            (List.cons
               (Atom "$moonbit.string_pool" : W.t)
               (List.cons
                  (List
                     (List.cons
                        (Atom "ref" : W.t)
                        ([ Atom "$moonbit.string_pool_type" ] : W.t list))
                    : W.t)
                  ([
                     List
                       (List.cons
                          (Atom "array.new_default" : W.t)
                          (List.cons
                             (Atom "$moonbit.string_pool_type" : W.t)
                             ([ i32_to_sexp string_pool_len ] : W.t list)));
                   ]
                    : W.t list))))
        : W.t)
    in
    List.cons
      (string_pool : W.t)
      (List.append (constant_constr : W.t list) (defaults : W.t list))

let compile_to_data (global_ctx : t) =
  let s = Const_table.to_wat_string global_ctx.const_table in
  if s = "" then []
  else
    let s = "\"" ^ W.escaped s ^ "\"" in
    ([
       List
         (List.cons
            (Atom "data" : W.t)
            (List.cons
               (Atom "$moonbit.const_data" : W.t)
               ([ Atom s ] : W.t list)));
     ]
      : W.t list)

let compile_func_ref_declare (global_ctx : t) =
  let addr_to_string = Basic_fn_address.to_wasm_name in
  let fns =
    Fn_addr_hashset.fold global_ctx.func_refs [] (fun addr acc ->
        (Atom (addr_to_string addr) : W.t) :: acc)
  in
  if fns = [] then []
  else
    ([
       List
         (List.cons
            (Atom "elem" : W.t)
            (List.cons
               (Atom "declare" : W.t)
               (List.cons (Atom "func" : W.t) (fns : W.t list))));
     ]
      : W.t list)

let add_import (global_ctx : t) (import : Import.t) =
  Import_hash.find_or_update global_ctx.imports import ~update:(fun _ ->
      let id = Import_hash.length global_ctx.imports in
      let module_name = Basic_strutil.mangle_wasm_name import.module_name in
      let func_name = Basic_strutil.mangle_wasm_name import.func_name in
      Stdlib.String.concat ""
        [ "$"; module_name; "."; func_name; "."; Int.to_string id ])

let add_inline (global_ctx : t) (func_body : W.t) =
  let uuid = Basic_uuid.next () in
  let name = ("$inline$" ^ Int.to_string uuid : Stdlib.String.t) in
  Vec.push global_ctx.inlines (name, func_body);
  name

let add_func_ref (global_ctx : t) (address : Fn_addr.t) =
  Fn_addr_hashset.add global_ctx.func_refs address

let compile_imports (global_ctx : t) =
  let ltype_to_sexp = Transl_type.ltype_to_sexp in
  Import_hash.fold global_ctx.imports []
    (fun { module_name; func_name; params_ty; return_ty } name acc ->
      let mname = Basic_strutil.esc_quote module_name in
      let fname = Basic_strutil.esc_quote func_name in
      let params =
        Lst.map params_ty (fun t : W.t ->
            List
              (List.cons (Atom "param" : W.t) ([ ltype_to_sexp t ] : W.t list)))
      in
      let return =
        match return_ty with
        | Some return_type ->
            ([
               List
                 (List.cons
                    (Atom "result" : W.t)
                    ([ ltype_to_sexp return_type ] : W.t list));
             ]
              : W.t list)
        | None -> []
      in
      List.cons
        (List
           (List.cons
              (Atom "func" : W.t)
              (List.cons
                 (Atom name : W.t)
                 (List.cons
                    (List
                       (List.cons
                          (Atom "import" : W.t)
                          (List.cons
                             (Atom mname : W.t)
                             ([ Atom fname ] : W.t list)))
                      : W.t)
                    (List.append (params : W.t list) (return : W.t list)))))
          : W.t)
        (acc : W.t list))

let compile_inlines (global_ctx : t) =
  Vec.map_into_list global_ctx.inlines (fun (name, body) ->
      match body with
      | (List (Atom "func" :: a) : W.t) ->
          (List
             (List.cons
                (Atom "func" : W.t)
                (List.cons (Atom name : W.t) (a : W.t list)))
            : W.t)
      | _ -> assert false)

let find_default_elem (global_ctx : t) (type_defs : Ltype.def Tid.Hash.t)
    (tid : Tid.t) =
  let add_if_not_exist tid default_name default =
    if not (Tid.Hash.mem global_ctx.defaults tid) then
      Tid.Hash.add global_ctx.defaults tid (default_name, default)
      [@@inline]
  in
  match Tid.Hash.find_opt type_defs tid with
  | Some (Ref_array { elem = Ref_string }) ->
      let default =
        if !Basic_config.use_js_builtin_string then Js_string
        else Arr Ltype.tid_string
      in
      add_if_not_exist Ltype.tid_string string_default_name default;
      Some string_default_name
  | Some (Ref_array { elem = Ref_bytes }) ->
      add_if_not_exist Ltype.tid_bytes bytes_default_name (Arr Ltype.tid_bytes);
      Some bytes_default_name
  | Some (Ref_array { elem = Ref { tid = elem_tid } })
    when Tid.equal elem_tid Ltype.tid_enum ->
      add_if_not_exist Ltype.tid_enum enum_default_name Enum;
      Some enum_default_name
  | _ -> None
