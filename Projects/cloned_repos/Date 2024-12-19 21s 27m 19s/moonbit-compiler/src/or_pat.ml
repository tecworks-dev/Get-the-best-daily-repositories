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
module I = Basic_ident
module Lst = Basic_lst

let rename_pat_obj =
  (object
     inherit [_] Typedtree.map

     method! visit_binder binder_tbl (b : Typedtree.binder) =
       match Hash_string.find_opt binder_tbl (I.base_name b.binder_id) with
       | Some id -> { binder_id = id; loc_ = b.loc_ }
       | None -> b
   end
    : < visit_pat : I.t Hash_string.t -> Typedtree.pat -> Typedtree.pat ; .. >)

let rename_pat_pat_binders =
  object
    inherit [_] Typedtree.map

    method! visit_binder (binder_tbl : Typedtree.pat_binders)
        (b : Typedtree.binder) =
      Lst.find_def binder_tbl
        (fun binder ->
          let binder_id = binder.binder.binder_id in
          if I.same_local_name binder_id b.binder_id then
            Some ({ binder_id; loc_ = b.loc_ } : Typedtree.binder)
          else None)
        b
  end

let rename_pat (p : Typedtree.pat) (binders : Typedtree.pat_binders) =
  match binders with
  | [] -> p
  | _ ->
      let len = List.length binders in
      if len < 4 then rename_pat_pat_binders#visit_pat binders p
      else
        let binder_tbl =
          Hash_string.of_list_map binders (fun x ->
              let binder_id = x.binder.binder_id in
              (I.base_name binder_id, binder_id))
        in
        rename_pat_obj#visit_pat binder_tbl p
