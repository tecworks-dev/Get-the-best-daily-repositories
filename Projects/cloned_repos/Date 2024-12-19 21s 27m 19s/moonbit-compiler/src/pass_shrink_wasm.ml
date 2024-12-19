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


module StringSet = Basic_set_string
module Lst = Basic_lst

type program_item =
  | Binding of (string * W.t)
  | Rec_type of (string list * W.t)
  | Raw of W.t

let is_ident name = String.starts_with ~prefix:"$" name

let extract_item keep_types sexp : program_item =
  match sexp with
  | (List (Atom "func" :: Atom name :: _) : W.t)
  | (List (Atom "global" :: Atom name :: _) : W.t)
  | (List (Atom "data" :: Atom name :: _) : W.t)
  | (List [ Atom "import"; _; _; List (Atom "func" :: Atom name :: _) ] : W.t)
    ->
      Binding (name, sexp)
  | (List (Atom "type" :: Atom name :: _) : W.t) ->
      if keep_types then Raw sexp else Binding (name, sexp)
  | (List (Atom "rec" :: types) : W.t) ->
      let names =
        Lst.fold_right types [] (fun t acc ->
            match t with
            | (List (Atom "type" :: Atom name :: _) : W.t) -> name :: acc
            | _ -> acc)
      in
      Rec_type (names, sexp)
  | _ -> Raw sexp

let get_all_ident ~(except : StringSet.t) (sexp : W.t) : StringSet.t =
  let rec go (sexp : W.t) =
    match sexp with
    | Atom name ->
        if is_ident name && not (StringSet.mem except name) then
          StringSet.singleton name
        else StringSet.empty
    | List sexps ->
        List.fold_left
          (fun acc sexp -> StringSet.union (go sexp) acc)
          StringSet.empty sexps
  in
  go sexp

let get_exported (sexp : W.t list) : StringSet.t =
  let exports = ref StringSet.empty in
  let go sexp =
    match sexp with
    | (List (Atom "func" :: Atom name :: List [ Atom "export"; _ ] :: _) : W.t)
    | (List (Atom "func" :: Atom name :: _ :: List [ Atom "export"; _ ] :: _) :
        W.t)
    | (List [ Atom "start"; Atom name ] : W.t)
    | (List [ Atom "export"; _; List [ _; Atom name ] ] : W.t)
    | (List (Atom "memory" :: Atom name :: List (Atom "export" :: _) :: _) :
        W.t) ->
        exports := StringSet.add !exports name
    | (List [ Atom "table"; _; Atom "funcref"; List (Atom "elem" :: elems) ] :
        W.t)
    | (List (Atom "elem" :: Atom "declare" :: Atom "func" :: elems) : W.t) ->
        exports :=
          StringSet.union !exports
            (List.fold_left StringSet.union StringSet.empty
               (Lst.map elems (get_all_ident ~except:StringSet.empty)))
    | _ -> ()
  in
  Basic_lst.iter sexp go;
  !exports

let slice_sexp_prog keep_types (body : W.t list) : program_item list =
  Lst.map body (extract_item keep_types)

let dependency_of_exports (items : program_item list) (exported : StringSet.t) :
    StringSet.t =
  let tbl = Hashtbl.create 50 in
  let collect_item (item : program_item) =
    match item with
    | Binding (x, sexp) -> Hashtbl.add tbl x sexp
    | Rec_type (names, sexp) ->
        Lst.iter names (fun name -> Hashtbl.add tbl name sexp)
    | Raw _ -> ()
  in
  Lst.iter items collect_item;
  let deps_of_exported = ref exported in
  let update_bit = ref true in
  let visited = ref StringSet.empty in
  while !update_bit do
    update_bit := false;
    StringSet.iter !deps_of_exported (fun name ->
        if not (StringSet.mem !visited name) then (
          visited := StringSet.add !visited name;
          match Hashtbl.find_opt tbl name with
          | Some sexp ->
              let new_deps = get_all_ident ~except:!deps_of_exported sexp in
              update_bit := !update_bit || not (StringSet.is_empty new_deps);
              deps_of_exported := StringSet.union !deps_of_exported new_deps
          | None -> ()))
  done;
  !deps_of_exported

let rebuild_program items usage =
  Lst.fold_right items [] (fun item acc ->
      match item with
      | Binding (name, sexp) ->
          if StringSet.mem usage name then sexp :: acc else acc
      | Rec_type (names, sexp) ->
          if Lst.exists names (fun name -> StringSet.mem usage name) then
            sexp :: acc
          else acc
      | Raw sexp -> sexp :: acc)

let shrink ?(keep_types = false) (code : W.t list) : W.t list =
  let items = slice_sexp_prog keep_types code in
  let exported = get_exported code in
  let usage = dependency_of_exports items exported in
  rebuild_program items usage
