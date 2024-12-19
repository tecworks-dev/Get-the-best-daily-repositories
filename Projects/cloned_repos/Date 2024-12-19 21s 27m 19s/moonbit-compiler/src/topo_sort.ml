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


module T = Basic_ident.Ordered_hash
module VI = Basic_vec_int
module Vec = Basic_vec
module Ident = Basic_ident
module Lst = Basic_lst

let add_error = Diagnostics.add_error

let vars_of_expr (e : Typedtree.expr) (tbl : _ T.t) : Ident.t list =
  let vars = ref [] in
  let obj =
    object
      inherit [_] Typedtree.iter

      method! visit_var _ v =
        if T.mem tbl v.var_id then vars := v.var_id :: !vars
    end
  in
  obj#visit_expr () e;
  !vars

type 'a impl_info = { impl : Typedtree.impl; info : 'a }
type binder_vars = { binder : Ident.t; vars : Ident.t list }

let topo_sort ~diagnostics (defs : Typedtree.output) : Typedtree.output =
  let (Output { value_defs = impls; type_defs; trait_defs }) = defs in
  let add_cycle cycle =
    let idents, locs = List.split cycle in
    let errors = Errors.cycle_definitions ~cycle:idents ~locs in
    Lst.iter errors (add_error diagnostics)
  in
  let tbl = T.create 17 in
  let extract_binder i (impl : Typedtree.impl) : Ident.t impl_info =
    match impl with
    | Timpl_stub_decl { binder; _ }
    | Timpl_fun_decl { fun_decl = { fn_binder = binder; _ }; _ } ->
        T.add tbl binder.binder_id i;
        { impl; info = binder.binder_id }
    | Timpl_letdef { binder; _ } ->
        let binder = binder.binder_id in
        T.add tbl binder i;
        { impl; info = binder }
    | Timpl_expr t ->
        let binder = Ident.of_qual_ident t.expr_id in
        T.add tbl binder i;
        { impl; info = binder }
  in
  let extract_vars (impl_info : Ident.t impl_info) : binder_vars impl_info =
    let impl = impl_info.impl in
    match impl with
    | Timpl_fun_decl { fun_decl = { fn; _ }; _ } ->
        {
          impl;
          info = { binder = impl_info.info; vars = vars_of_expr fn.body tbl };
        }
    | Timpl_stub_decl _ ->
        { impl; info = { binder = impl_info.info; vars = [] } }
    | Timpl_letdef { expr; loc_; _ } ->
        let binder = impl_info.info in
        let vars = vars_of_expr expr tbl in
        if Basic_lst.exists vars (fun v -> Ident.equal binder v) then
          add_cycle [ (binder, loc_) ];
        { impl; info = { binder; vars } }
    | Timpl_expr { expr; is_main = _; expr_id = _ } ->
        {
          impl;
          info = { binder = impl_info.info; vars = vars_of_expr expr tbl };
        }
  in
  let impl_array =
    impls |> Array.of_list |> Array.mapi extract_binder
    |> Array.map extract_vars
  in
  let nodes_num = Array.length impl_array in
  let adjacency_array = Array.init nodes_num (fun _ -> VI.empty ()) in
  let add_edge (binder : Ident.t) (var : Ident.t) =
    let src = T.find_value tbl binder in
    let tgt = T.find_value tbl var in
    VI.push adjacency_array.(src) tgt
  in
  Array.iter
    (fun impl -> List.iter (add_edge impl.info.binder) impl.info.vars)
    impl_array;
  let scc = Basic_scc.graph adjacency_array in
  let has_cycle (c : VI.t) : bool =
    let is_letdef impl =
      match impl with Typedtree.Timpl_letdef _ -> true | _ -> false
    in
    VI.length c > 1 && VI.exists (fun i -> is_letdef impl_array.(i).impl) c
  in
  let handle_cycle (c : VI.t) : unit =
    let cycle =
      VI.map_into_list c (fun i ->
          let impl = impl_array.(i) in
          (impl.info.binder, Typedtree.loc_of_impl impl.impl))
    in
    add_cycle cycle
  in
  Vec.iter scc (fun c -> if has_cycle c then handle_cycle c);
  let value_defs =
    Vec.map_into_list scc (fun c ->
        VI.map_into_list c (fun i -> impl_array.(i).impl))
    |> List.concat
  in
  Output { value_defs; type_defs; trait_defs }
