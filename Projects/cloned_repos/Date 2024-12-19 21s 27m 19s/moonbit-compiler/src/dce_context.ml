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


module Ident = Basic_core_ident

type def = Value of Core.expr | Func of Core.fn
type env = { used : Ident.Hashset.t; defs : def Ident.Hash.t }

let mark_as_used (env : env) id ~dce_visitor =
  if Ident.Hashset.check_add env.used id then
    Ident.Hash.update_if_exists env.defs id (fun def ->
        match def with
        | Value expr -> Value (dce_visitor env expr)
        | Func fn -> Func { fn with body = dce_visitor env fn.body })

let add_value env name rhs = Ident.Hash.add env.defs name (Value rhs)
let add_func env name fn = Ident.Hash.add env.defs name (Func fn)

let get_analyzed_value env id =
  match Ident.Hash.find_exn env.defs id with
  | Value expr -> expr
  | Func _ -> assert false

let get_analyzed_fn env id =
  match Ident.Hash.find_exn env.defs id with
  | Func fn -> fn
  | Value _ -> assert false

let make () = { used = Ident.Hashset.create 17; defs = Ident.Hash.create 17 }
let used env id = Ident.Hashset.mem env.used id
