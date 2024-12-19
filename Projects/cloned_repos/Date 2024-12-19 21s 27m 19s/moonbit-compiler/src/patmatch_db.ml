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


module Path = Pat_path
module StaticInfo = Patmatch_static_info

type t = StaticInfo.t Path.Map.t

type 'a eval_result = 'a Basic_case_set_intf.static_matching_result =
  | For_sure_yes of { ok_db : 'a }
  | For_sure_no of { fail_db : 'a }
  | Uncertain of { ok_db : 'a; fail_db : 'a }

type static_eval_result = t eval_result

let empty : t = Path.Map.empty

let make_result (self : t) (path : Path.t)
    ~info:(matched : StaticInfo.t eval_result) : static_eval_result =
  match matched with
  | For_sure_yes { ok_db } ->
      For_sure_yes { ok_db = Path.Map.add self path ok_db }
  | For_sure_no { fail_db } ->
      For_sure_no { fail_db = Path.Map.add self path fail_db }
  | Uncertain { ok_db; fail_db } ->
      Uncertain
        {
          ok_db = Path.Map.add self path ok_db;
          fail_db = Path.Map.add self path fail_db;
        }

let eval_constant self path constant : static_eval_result =
  make_result self path
    ~info:(StaticInfo.eval_constant (Path.Map.find_opt self path) constant)

let eval_range self path lo hi ~inclusive : static_eval_result =
  make_result self path
    ~info:(StaticInfo.eval_range (Path.Map.find_opt self path) lo hi ~inclusive)

let eval_constructor self path constr_tag ~used_error_subtyping :
    static_eval_result =
  make_result self path
    ~info:
      (StaticInfo.eval_constructor
         (Path.Map.find_opt self path)
         constr_tag ~used_error_subtyping)

let eval_eq_array_len self path len : static_eval_result =
  make_result self path
    ~info:(StaticInfo.eval_eq_array_len (Path.Map.find_opt self path) len)

let eval_geq_array_len self path len : static_eval_result =
  make_result self path
    ~info:(StaticInfo.eval_geq_array_len (Path.Map.find_opt self path) len)

let eval_map_elem self path key ~elem_ty : t =
  Path.Map.add self path
    (StaticInfo.eval_map_elem (Path.Map.find_opt self path) key ~elem_ty)
