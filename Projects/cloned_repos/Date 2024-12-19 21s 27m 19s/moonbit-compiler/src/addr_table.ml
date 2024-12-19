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


module Fn_address = Basic_fn_address
module Ident_hash = Basic_core_ident.Hash
module Ident = Clam1_ident
module Core_ident = Basic_core_ident

type fn_info =
  | Toplevel of {
      addr : Fn_address.t;
      params : Ident.t list;
      return : Ltype.return_type;
      mutable name_as_closure : Ident.t option;
    }
  | Local of Fn_address.t * Ltype.t

type t = fn_info Ident_hash.t

let add_toplevel_fn (t : t) (name : Core_ident.t) ~params ~return =
  match[@warning "-fragile-match"] name with
  | Pdot qual_name ->
      let addr = Fn_address.of_qual_ident qual_name in
      Ident_hash.add t name
        (Toplevel { addr; params; return; name_as_closure = None })
  | _ -> assert false

let add_local_fn_addr_and_type (t : t) (name : Core_ident.t)
    (addr : Fn_address.t) ty =
  Ident_hash.add t name (Local (addr, ty))

let find_exn = Ident_hash.find_exn
let find_opt = Ident_hash.find_opt
let create = Ident_hash.create
let fold = Ident_hash.fold
