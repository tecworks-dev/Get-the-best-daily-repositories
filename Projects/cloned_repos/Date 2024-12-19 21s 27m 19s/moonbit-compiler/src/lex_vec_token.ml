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


module Vec = Basic_vec

type t = Lex_token_triple.t Vec.t

include struct
  let _ = fun (_ : t) -> ()

  let sexp_of_t =
    (fun x__001_ -> Vec.sexp_of_t Lex_token_triple.sexp_of_t x__001_ : t -> S.t)

  let _ = sexp_of_t
end

let string_of_tokens (tokens : t) =
  match sexp_of_t tokens with
  | Atom _ as s -> S.to_string s
  | List ss -> String.concat "\n" (Basic_lst.map ss (fun s -> S.to_string s))
