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
  mutable base_pos : Lexing.position;
  mutable segment : Parsing_segment.t;
}

let state : t =
  { base_pos = Lexing.dummy_pos; segment = Parsing_segment.dummy_segment }

let get_base_pos () = state.base_pos

let initialize_state (seg : Parsing_segment.t) =
  let _, base_pos, _ = Parsing_segment.peek seg in
  state.base_pos <- base_pos;
  state.segment <- seg

let initialize_base_pos (base_pos : Lexing.position) =
  state.base_pos <- base_pos

let update_state () =
  let _, base_pos, _ = Parsing_segment.peek state.segment in
  state.base_pos <- base_pos
