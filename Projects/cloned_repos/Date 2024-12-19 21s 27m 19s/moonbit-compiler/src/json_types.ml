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


type json_str = { str : string; loc : Loc.t }

include struct
  let _ = fun (_ : json_str) -> ()

  let sexp_of_json_str =
    (fun { str = str__002_; loc = loc__004_ } ->
       let bnds__001_ = ([] : _ Stdlib.List.t) in
       let bnds__001_ =
         let arg__005_ = Loc.sexp_of_t loc__004_ in
         (S.List [ S.Atom "loc"; arg__005_ ] :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__003_ = Moon_sexp_conv.sexp_of_string str__002_ in
         (S.List [ S.Atom "str"; arg__003_ ] :: bnds__001_ : _ Stdlib.List.t)
       in
       S.List bnds__001_
      : json_str -> S.t)

  let _ = sexp_of_json_str
end

type json_float = { float : string; loc : Loc.t }

include struct
  let _ = fun (_ : json_float) -> ()

  let sexp_of_json_float =
    (fun { float = float__007_; loc = loc__009_ } ->
       let bnds__006_ = ([] : _ Stdlib.List.t) in
       let bnds__006_ =
         let arg__010_ = Loc.sexp_of_t loc__009_ in
         (S.List [ S.Atom "loc"; arg__010_ ] :: bnds__006_ : _ Stdlib.List.t)
       in
       let bnds__006_ =
         let arg__008_ = Moon_sexp_conv.sexp_of_string float__007_ in
         (S.List [ S.Atom "float"; arg__008_ ] :: bnds__006_ : _ Stdlib.List.t)
       in
       S.List bnds__006_
      : json_float -> S.t)

  let _ = sexp_of_json_float
end

type json_array = { content : t array; loc : Loc.t }
and json_map = { map : t Basic_map_string.t; loc : Loc.t }

and t =
  | True of Loc.t
  | False of Loc.t
  | Null of Loc.t
  | Float of json_float
  | Str of json_str
  | Arr of json_array
  | Obj of json_map

include struct
  let _ = fun (_ : json_array) -> ()
  let _ = fun (_ : json_map) -> ()
  let _ = fun (_ : t) -> ()

  let rec sexp_of_json_array =
    (fun { content = content__012_; loc = loc__014_ } ->
       let bnds__011_ = ([] : _ Stdlib.List.t) in
       let bnds__011_ =
         let arg__015_ = Loc.sexp_of_t loc__014_ in
         (S.List [ S.Atom "loc"; arg__015_ ] :: bnds__011_ : _ Stdlib.List.t)
       in
       let bnds__011_ =
         let arg__013_ = Moon_sexp_conv.sexp_of_array sexp_of_t content__012_ in
         (S.List [ S.Atom "content"; arg__013_ ] :: bnds__011_
           : _ Stdlib.List.t)
       in
       S.List bnds__011_
      : json_array -> S.t)

  and sexp_of_json_map =
    (fun { map = map__017_; loc = loc__019_ } ->
       let bnds__016_ = ([] : _ Stdlib.List.t) in
       let bnds__016_ =
         let arg__020_ = Loc.sexp_of_t loc__019_ in
         (S.List [ S.Atom "loc"; arg__020_ ] :: bnds__016_ : _ Stdlib.List.t)
       in
       let bnds__016_ =
         let arg__018_ = Basic_map_string.sexp_of_t sexp_of_t map__017_ in
         (S.List [ S.Atom "map"; arg__018_ ] :: bnds__016_ : _ Stdlib.List.t)
       in
       S.List bnds__016_
      : json_map -> S.t)

  and sexp_of_t =
    (function
     | True arg0__021_ ->
         let res0__022_ = Loc.sexp_of_t arg0__021_ in
         S.List [ S.Atom "True"; res0__022_ ]
     | False arg0__023_ ->
         let res0__024_ = Loc.sexp_of_t arg0__023_ in
         S.List [ S.Atom "False"; res0__024_ ]
     | Null arg0__025_ ->
         let res0__026_ = Loc.sexp_of_t arg0__025_ in
         S.List [ S.Atom "Null"; res0__026_ ]
     | Float arg0__027_ ->
         let res0__028_ = sexp_of_json_float arg0__027_ in
         S.List [ S.Atom "Float"; res0__028_ ]
     | Str arg0__029_ ->
         let res0__030_ = sexp_of_json_str arg0__029_ in
         S.List [ S.Atom "Str"; res0__030_ ]
     | Arr arg0__031_ ->
         let res0__032_ = sexp_of_json_array arg0__031_ in
         S.List [ S.Atom "Arr"; res0__032_ ]
     | Obj arg0__033_ ->
         let res0__034_ = sexp_of_json_map arg0__033_ in
         S.List [ S.Atom "Obj"; res0__034_ ]
      : t -> S.t)

  let _ = sexp_of_json_array
  and _ = sexp_of_json_map
  and _ = sexp_of_t
end
