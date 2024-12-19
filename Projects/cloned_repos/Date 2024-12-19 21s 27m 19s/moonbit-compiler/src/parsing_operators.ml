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


module Type_path = Basic_type_path
module H = Basic_hash_string

type operator_kind =
  | Prefix
  | Infix of { should_have_equal_param_type : bool }
  | Mixfix of { arity : int; snippet : string }

type operator_impl =
  | Regular_function
  | Method
  | Method_with_default of Type_path.t
  | Trait_method of Type_path.t

type operator_info = {
  op_name : string;
  method_name : string;
  kind : operator_kind;
  impl : operator_impl;
}

let tint = Type_path.Builtin.type_path_int

let op_equal_info =
  {
    op_name = "==";
    method_name = "op_equal";
    kind = Infix { should_have_equal_param_type = true };
    impl = Trait_method Type_path.Builtin.trait_eq;
  }

let op_add_info =
  {
    op_name = "+";
    method_name = "op_add";
    kind = Infix { should_have_equal_param_type = true };
    impl = Method_with_default tint;
  }

let op_get_info =
  {
    op_name = "_[_]";
    method_name = "op_get";
    kind = Mixfix { arity = 2; snippet = "[$1]$0" };
    impl = Method;
  }

let op_get_slice_info =
  {
    op_name = "_[_:_]";
    method_name = "op_as_view";
    kind = Mixfix { arity = 3; snippet = "[$1:$2]$0" };
    impl = Method;
  }

let op_set_info =
  {
    op_name = "_[_]=_";
    method_name = "op_set";
    kind = Mixfix { arity = 3; snippet = "[$1] = $0" };
    impl = Method;
  }

let op_bitwise_and =
  {
    op_name = "&";
    method_name = "land";
    kind = Mixfix { arity = 2; snippet = ".land($1)" };
    impl = Method_with_default tint;
  }

let op_bitwise_xor =
  {
    op_name = "^";
    method_name = "lxor";
    kind = Mixfix { arity = 2; snippet = ".lxor($1)" };
    impl = Method_with_default tint;
  }

let op_bitwise_or =
  {
    op_name = "|";
    method_name = "lor";
    kind = Mixfix { arity = 2; snippet = ".lor($1)" };
    impl = Method_with_default tint;
  }

let op_bitwise_leftshift =
  {
    op_name = "<<";
    method_name = "op_shl";
    kind = Infix { should_have_equal_param_type = false };
    impl = Method_with_default tint;
  }

let op_bitwise_rightshift =
  {
    op_name = ">>";
    method_name = "op_shr";
    kind = Infix { should_have_equal_param_type = false };
    impl = Method_with_default tint;
  }

let operators =
  [
    {
      op_name = "~-";
      method_name = "op_neg";
      kind = Prefix;
      impl = Method_with_default tint;
    };
    op_add_info;
    {
      op_name = "-";
      method_name = "op_sub";
      kind = Infix { should_have_equal_param_type = true };
      impl = Method_with_default tint;
    };
    {
      op_name = "*";
      method_name = "op_mul";
      kind = Infix { should_have_equal_param_type = true };
      impl = Method_with_default tint;
    };
    {
      op_name = "/";
      method_name = "op_div";
      kind = Infix { should_have_equal_param_type = true };
      impl = Method_with_default tint;
    };
    {
      op_name = "%";
      method_name = "op_mod";
      kind = Infix { should_have_equal_param_type = false };
      impl = Method_with_default tint;
    };
    op_get_info;
    op_get_slice_info;
    op_set_info;
    op_equal_info;
    {
      op_name = "!=";
      method_name = "op_notequal";
      kind = Infix { should_have_equal_param_type = true };
      impl = Regular_function;
    };
    {
      op_name = "<";
      method_name = "op_lt";
      kind = Infix { should_have_equal_param_type = true };
      impl = Regular_function;
    };
    {
      op_name = "<=";
      method_name = "op_le";
      kind = Infix { should_have_equal_param_type = true };
      impl = Regular_function;
    };
    {
      op_name = ">";
      method_name = "op_gt";
      kind = Infix { should_have_equal_param_type = true };
      impl = Regular_function;
    };
    {
      op_name = ">=";
      method_name = "op_ge";
      kind = Infix { should_have_equal_param_type = true };
      impl = Regular_function;
    };
    op_bitwise_xor;
    op_bitwise_or;
    op_bitwise_and;
    op_bitwise_leftshift;
    op_bitwise_rightshift;
  ]

let display_name_of_op op = if op.op_name = "~-" then "-" else op.op_name
let table_by_name = H.of_list_map operators (fun op -> (op.op_name, op))
let table_by_method = H.of_list_map operators (fun op -> (op.method_name, op))
let find_exn op = H.find_exn table_by_name op
let find_by_method_opt method_name = H.find_opt table_by_method method_name
