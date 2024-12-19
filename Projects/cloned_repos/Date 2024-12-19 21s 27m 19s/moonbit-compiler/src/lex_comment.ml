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


type t = {
  content : string;
  mutable kind : kind;
  consumed_by_docstring : bool ref; [@sexp_drop_if fun _ -> true]
}

and kind =
  | Inline_trailing
  | Ownline of { leading_blank_line : bool; trailing_blank_line : bool }

include struct
  let _ = fun (_ : t) -> ()
  let _ = fun (_ : kind) -> ()

  let rec sexp_of_t =
    (let (drop_if__007_ : bool ref -> Stdlib.Bool.t) = fun _ -> true in
     fun {
           content = content__002_;
           kind = kind__004_;
           consumed_by_docstring = consumed_by_docstring__008_;
         } ->
       let bnds__001_ = ([] : _ Stdlib.List.t) in
       let bnds__001_ =
         if drop_if__007_ consumed_by_docstring__008_ then bnds__001_
         else
           let arg__010_ =
             (Moon_sexp_conv.sexp_of_ref Moon_sexp_conv.sexp_of_bool)
               consumed_by_docstring__008_
           in
           let bnd__009_ =
             S.List [ S.Atom "consumed_by_docstring"; arg__010_ ]
           in
           (bnd__009_ :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__005_ = sexp_of_kind kind__004_ in
         (S.List [ S.Atom "kind"; arg__005_ ] :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__003_ = Moon_sexp_conv.sexp_of_string content__002_ in
         (S.List [ S.Atom "content"; arg__003_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       S.List bnds__001_
      : t -> S.t)

  and sexp_of_kind =
    (function
     | Inline_trailing -> S.Atom "Inline_trailing"
     | Ownline
         {
           leading_blank_line = leading_blank_line__012_;
           trailing_blank_line = trailing_blank_line__014_;
         } ->
         let bnds__011_ = ([] : _ Stdlib.List.t) in
         let bnds__011_ =
           let arg__015_ =
             Moon_sexp_conv.sexp_of_bool trailing_blank_line__014_
           in
           (S.List [ S.Atom "trailing_blank_line"; arg__015_ ] :: bnds__011_
             : _ Stdlib.List.t)
         in
         let bnds__011_ =
           let arg__013_ =
             Moon_sexp_conv.sexp_of_bool leading_blank_line__012_
           in
           (S.List [ S.Atom "leading_blank_line"; arg__013_ ] :: bnds__011_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Ownline" :: bnds__011_)
      : kind -> S.t)

  let _ = sexp_of_t
  and _ = sexp_of_kind
end

type with_loc = (Loc.t * t) list

include struct
  let _ = fun (_ : with_loc) -> ()

  let sexp_of_with_loc =
    (fun x__020_ ->
       Moon_sexp_conv.sexp_of_list
         (fun (arg0__016_, arg1__017_) ->
           let res0__018_ = Loc.sexp_of_t arg0__016_
           and res1__019_ = sexp_of_t arg1__017_ in
           S.List [ res0__018_; res1__019_ ])
         x__020_
      : with_loc -> S.t)

  let _ = sexp_of_with_loc
end
