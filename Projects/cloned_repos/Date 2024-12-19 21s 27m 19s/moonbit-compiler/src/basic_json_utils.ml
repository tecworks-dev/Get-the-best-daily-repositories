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


let buffer_add_json_string ob s =
  Buffer.add_char ob '"';
  let start = ref 0 in
  let end_ = String.length s in
  let add_til_escape i ob esc =
    Buffer.add_substring ob s !start (i - !start);
    Buffer.add_string ob esc;
    start := i + 1
  in
  for i = 0 to end_ - 1 do
    match s.[i] with
    | '"' -> add_til_escape i ob "\\\""
    | '\\' -> add_til_escape i ob "\\\\"
    | '\b' -> add_til_escape i ob "\\b"
    | '\012' -> add_til_escape i ob "\\f"
    | '\n' -> add_til_escape i ob "\\n"
    | '\r' -> add_til_escape i ob "\\r"
    | '\t' -> add_til_escape i ob "\\t"
    | ('\000' .. '\031' | '\127') as c ->
        Buffer.add_substring ob s !start (i - !start);
        Buffer.add_string ob "\\u00";
        let hex n = Char.chr (if n < 10 then n + 48 else n + 87) in
        Buffer.add_char ob (hex (Char.code c lsr 4));
        Buffer.add_char ob (hex (Char.code c land 0xf));
        start := i + 1
    | _ -> ()
  done;
  Buffer.add_substring ob s !start (end_ - !start);
  Buffer.add_char ob '"'
