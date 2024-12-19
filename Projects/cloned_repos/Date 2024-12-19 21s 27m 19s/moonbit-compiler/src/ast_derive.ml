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


module Lst = Basic_lst
module Vec = Basic_vec
module Type_path = Basic_type_path
module Syntax = Parsing_syntax
module Operators = Parsing_operators

let ghost_loc_ = Rloc.no_location

let fresh_name name =
  Stdlib.String.concat "" [ "*"; name; "_"; Int.to_string (Basic_uuid.next ()) ]

module S = struct
  let typ (ty_name : string) (tys : Syntax.typ list) : Syntax.typ =
    Ptype_name
      {
        constr_id = { lid = Lident ty_name; loc_ = ghost_loc_ };
        tys;
        loc_ = ghost_loc_;
      }

  let typ_any : Syntax.typ = Ptype_any { loc_ = ghost_loc_ }

  let is_option (field : Syntax.field_decl) =
    match field.field_ty with
    | Ptype_option _
    | Ptype_name { constr_id = { lid = Lident "Option"; _ }; _ } ->
        true
    | _ -> false

  let label label_name : Syntax.label = { label_name; loc_ = ghost_loc_ }
  let type_name name : Syntax.type_name = { name; loc_ = ghost_loc_ }
  let hole : Syntax.expr = Pexpr_hole { loc_ = ghost_loc_; kind = Synthesized }

  let const_uint_pat i : Syntax.pattern =
    Ppat_constant { c = Const_uint i; loc_ = ghost_loc_ }

  let const_uint i : Syntax.expr =
    Pexpr_constant { c = Const_uint i; loc_ = ghost_loc_ }

  let const_int i : Syntax.expr =
    Pexpr_constant { c = Const_int i; loc_ = ghost_loc_ }

  let const_bool b : Syntax.expr =
    Pexpr_constant { c = Const_bool b; loc_ = ghost_loc_ }

  let const_string s : Syntax.expr =
    Pexpr_constant
      {
        c = Const_string { string_val = s; string_repr = s };
        loc_ = ghost_loc_;
      }

  let method_ type_ method_name : Syntax.expr =
    Pexpr_method
      {
        type_name = type_name type_;
        method_name = label method_name;
        loc_ = ghost_loc_;
      }

  let apply func args attr : Syntax.expr =
    let args =
      Lst.map args (fun arg : Syntax.argument ->
          { arg_value = arg; arg_kind = Positional })
    in
    Pexpr_apply { func; args; attr; loc_ = ghost_loc_ }

  let annotation expr ty : Syntax.expr =
    Pexpr_constraint { expr; ty; loc_ = ghost_loc_ }

  let apply_trait_method ~assertions ~typ ~loc ~msg trait method_name args attr
      : Syntax.expr =
    let assertion : Syntax.static_assertion =
      {
        assert_type = typ;
        assert_trait = trait;
        assert_loc = loc;
        assert_msg = msg;
      }
    in
    Vec.push assertions assertion;
    apply (method_ trait method_name) args attr

  let dot_apply self name args : Syntax.expr =
    let args =
      Lst.map args (fun arg : Syntax.argument ->
          { arg_value = arg; arg_kind = Positional })
    in
    let method_name : Syntax.label = { label_name = name; loc_ = ghost_loc_ } in
    Pexpr_dot_apply
      {
        self;
        method_name;
        args;
        attr = No_attr;
        loc_ = ghost_loc_;
        return_self = false;
      }

  let dotdot_apply self name args : Syntax.expr =
    let args =
      Lst.map args (fun arg : Syntax.argument ->
          { arg_value = arg; arg_kind = Positional })
    in
    let method_name : Syntax.label = { label_name = name; loc_ = ghost_loc_ } in
    Pexpr_dot_apply
      {
        self;
        method_name;
        args;
        attr = No_attr;
        loc_ = ghost_loc_;
        return_self = true;
      }

  let constr ?(extra_info = Syntax.No_extra_info) name args : Syntax.expr =
    let constr : Syntax.expr =
      Pexpr_constr
        {
          constr =
            {
              extra_info;
              constr_name = { name; loc_ = ghost_loc_ };
              loc_ = ghost_loc_;
            };
          loc_ = ghost_loc_;
        }
    in
    match args with [] -> constr | args -> apply constr args No_attr

  let apply_label func args : Syntax.expr =
    let args : Syntax.argument list =
      Lst.map args
        (fun (arg : Syntax.expr * Syntax.label option) : Syntax.argument ->
          let kind : Syntax.argument_kind =
            match snd arg with
            | None -> Syntax.Positional
            | Some label -> Labelled label
          in
          { arg_value = fst arg; arg_kind = kind })
    in
    Pexpr_apply { func; args; attr = No_attr; loc_ = ghost_loc_ }

  let constr_label name args : Syntax.expr =
    let constr : Syntax.expr =
      Pexpr_constr
        {
          constr =
            {
              extra_info = No_extra_info;
              constr_name = { name; loc_ = ghost_loc_ };
              loc_ = ghost_loc_;
            };
          loc_ = ghost_loc_;
        }
    in
    match args with [] -> constr | args -> apply_label constr args

  let record fields : Syntax.expr =
    Pexpr_record
      {
        type_name = None;
        trailing = Trailing_none;
        loc_ = ghost_loc_;
        fields =
          Lst.map fields (fun (label_name, value) : Syntax.field_def ->
              Field_def
                {
                  label = label label_name;
                  expr = value;
                  is_pun = false;
                  loc_ = ghost_loc_;
                });
      }

  let field record label_name : Syntax.expr =
    Pexpr_field
      {
        record;
        accessor = Label { label_name; loc_ = ghost_loc_ };
        loc_ = ghost_loc_;
      }

  let newtype_field nt : Syntax.expr =
    Pexpr_field { record = nt; accessor = Newtype; loc_ = ghost_loc_ }

  let tuple exprs : Syntax.expr = Pexpr_tuple { exprs; loc_ = ghost_loc_ }
  let unit : Syntax.expr = Pexpr_unit { loc_ = ghost_loc_; faked = false }

  let infix_op op lhs rhs : Syntax.expr =
    Pexpr_infix
      {
        op = { var_name = Lident op; loc_ = ghost_loc_ };
        lhs;
        rhs;
        loc_ = ghost_loc_;
      }

  let array exprs : Syntax.expr = Pexpr_array { exprs; loc_ = ghost_loc_ }

  let map_string_expr entries : Syntax.expr =
    let elems =
      Lst.map entries (fun (k, v) : Syntax.map_expr_elem ->
          Map_expr_elem
            {
              key = Const_string { string_val = k; string_repr = k };
              expr = v;
              key_loc_ = ghost_loc_;
              loc_ = ghost_loc_;
            })
    in
    Pexpr_map { elems; loc_ = ghost_loc_ }

  let pany : Syntax.pattern = Ppat_any { loc_ = ghost_loc_ }

  let pvar name : Syntax.pattern =
    Ppat_var { binder_name = name; loc_ = ghost_loc_ }

  let ptuple pats : Syntax.pattern = Ppat_tuple { pats; loc_ = ghost_loc_ }

  let pconstr constr args : Syntax.pattern =
    let args =
      match args with
      | [] -> None
      | args ->
          Lst.map args (fun (label_opt, pat) : Syntax.constr_pat_arg ->
              match label_opt with
              | None -> Constr_pat_arg { pat; kind = Positional }
              | Some label_name ->
                  Constr_pat_arg
                    { pat; kind = Labelled { label_name; loc_ = ghost_loc_ } })
          |> Option.some
    in
    Ppat_constr
      {
        constr =
          {
            extra_info = No_extra_info;
            constr_name = { name = constr; loc_ = ghost_loc_ };
            loc_ = ghost_loc_;
          };
        args;
        is_open = false;
        loc_ = ghost_loc_;
      }

  let pconstant c : Syntax.pattern = Ppat_constant { c; loc_ = ghost_loc_ }
  let pmap elems : Syntax.pattern = Ppat_map { elems; loc_ = ghost_loc_ }

  let por pats : Syntax.pattern =
    match pats with
    | [] -> assert false
    | pat :: [] -> pat
    | pat :: pats ->
        Lst.fold_left pats pat (fun acc pat ->
            Ppat_or { pat1 = acc; pat2 = pat; loc_ = ghost_loc_ })

  let map_pat_elem_lst entries : Syntax.map_pat_elem list =
    Lst.map entries (fun (k, v) : Syntax.map_pat_elem ->
        Map_pat_elem
          {
            key = Const_string { string_val = k; string_repr = k };
            pat = v;
            match_absent = false;
            key_loc_ = ghost_loc_;
            loc_ = ghost_loc_;
          })

  let var name : Syntax.expr =
    Pexpr_ident
      { id = { var_name = Lident name; loc_ = ghost_loc_ }; loc_ = ghost_loc_ }

  let let_ pattern expr body : Syntax.expr =
    Pexpr_let { pattern; expr; body; loc_ = ghost_loc_ }

  let sequence expr1 expr2 : Syntax.expr =
    Pexpr_sequence { expr1; expr2; loc_ = ghost_loc_ }

  let rec seq exprs : Syntax.expr =
    match exprs with
    | [] -> unit
    | expr :: [] -> expr
    | expr :: exprs -> sequence expr (seq exprs)

  let match_ expr cases : Syntax.expr =
    Pexpr_match { expr; cases; match_loc_ = ghost_loc_; loc_ = ghost_loc_ }

  let rest_raise_json_decode_error_ (path : Syntax.expr) (msg : Syntax.expr) :
      Syntax.pattern * Syntax.expr =
    let err_value =
      let t = tuple [ path; msg ] in
      constr "JsonDecodeError" [ t ] ~extra_info:(Syntax.Package "json")
    in
    (pvar "_", Pexpr_raise { err_value; loc_ = ghost_loc_ })

  let map_insert_json key json_val : Syntax.expr =
    dot_apply (var "$map") "set" [ const_string key; json_val ]
end

type deriver =
  Syntax.deriving_directive ->
  Syntax.type_decl ->
  params:string list ->
  assertions:Syntax.static_assertion Vec.t ->
  diagnostics:Local_diagnostics.t ->
  Syntax.expr

let derive_default (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~params:(_params : string list)
    ~(assertions : Syntax.static_assertion Vec.t)
    ~(diagnostics : Local_diagnostics.t) =
  let rec default_of_typ ~msg (typ : Syntax.typ) : Syntax.expr =
    match typ with
    | Ptype_tuple { tys; _ } -> S.tuple (Lst.map tys (default_of_typ ~msg))
    | typ ->
        S.annotation
          (S.apply_trait_method ~assertions ~typ ~msg
             ~loc:(Syntax.loc_of_type_expression typ)
             trait.name "default" [] No_attr)
          typ
  in
  match decl.components with
  | Ptd_abstract ->
      Local_diagnostics.add_error diagnostics
        (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
           ~reason:"target type is abstract" ~loc:trait.loc_);
      S.hole
  | Ptd_newtype typ ->
      S.constr decl.tycon
        [
          default_of_typ
            ~msg:("derive(Default) for newtype " ^ decl.tycon : Stdlib.String.t)
            typ;
        ]
  | Ptd_error No_payload -> S.constr decl.tycon []
  | Ptd_error (Single_payload typ) ->
      S.constr decl.tycon
        [
          default_of_typ
            ~msg:
              ("derive(Default) for error type " ^ decl.tycon : Stdlib.String.t)
            typ;
        ]
  | Ptd_error (Enum_payload constrs) | Ptd_variant constrs -> (
      match
        Lst.filter constrs (fun constr -> Option.is_none constr.constr_args)
      with
      | [] ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"cannot find a constant constructor as default"
               ~loc:trait.loc_);
          S.hole
      | constr :: [] -> S.constr constr.constr_name.name []
      | { constr_name = constr1; _ } :: { constr_name = constr2; _ } :: rest ->
          let candidate_str =
            match rest with
            | [] -> (constr1.name ^ " and " ^ constr2.name : Stdlib.String.t)
            | _ ->
                Stdlib.String.concat ""
                  [ constr1.name; ", "; constr2.name; ", ..." ]
          in
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:
                 (candidate_str ^ " are both candidates as default constructor"
                   : Stdlib.String.t)
               ~loc:trait.loc_);
          S.hole)
  | Ptd_record fields ->
      Lst.map fields (fun field ->
          ( field.field_name.label,
            default_of_typ
              ~msg:
                (Stdlib.String.concat ""
                   [
                     "derive(Default) for field ";
                     field.field_name.label;
                     " of type ";
                     decl.tycon;
                   ])
              field.field_ty ))
      |> S.record
  | Ptd_alias _ -> S.hole

let derive_eq (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  let rec all_eq xs ys zs ~eq_for_elem =
    match (xs, ys, zs) with
    | [], [], [] -> S.const_bool true
    | x :: [], y :: [], z :: [] -> eq_for_elem x y z
    | x :: xs, y :: ys, z :: zs ->
        S.infix_op "&&" (eq_for_elem x y z) (all_eq xs ys zs ~eq_for_elem)
    | _ -> assert false
  in
  let rec eq_of_typ ~msg (typ : Syntax.typ) (lhs : Syntax.expr)
      (rhs : Syntax.expr) =
    match typ with
    | Ptype_tuple { tys; _ } ->
        let names1 =
          Lst.mapi tys (fun i _ ->
              fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
        in
        let names2 =
          Lst.mapi tys (fun i _ ->
              fresh_name ("y" ^ Int.to_string i : Stdlib.String.t))
        in
        S.let_
          (S.ptuple (Lst.map names1 S.pvar))
          lhs
          (S.let_
             (S.ptuple (Lst.map names2 S.pvar))
             rhs
             (all_eq tys names1 names2 ~eq_for_elem:(fun ty lhs rhs ->
                  eq_of_typ ~msg ty (S.var lhs) (S.var rhs))))
    | typ ->
        S.apply_trait_method ~assertions ~typ
          ~loc:(Syntax.loc_of_type_expression typ)
          ~msg trait.name Operators.op_equal_info.method_name [ lhs; rhs ]
          No_attr
  in
  match[@warning "-fragile-match"] params with
  | [ lhs; rhs ] -> (
      let lhs = S.var lhs in
      let rhs = S.var rhs in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          let lhs = S.newtype_field lhs in
          let rhs = S.newtype_field rhs in
          eq_of_typ
            ~msg:("derive(Eq) for newtype " ^ decl.tycon : Stdlib.String.t)
            typ lhs rhs
      | Ptd_error No_payload -> S.const_bool true
      | Ptd_error (Single_payload typ) ->
          let x = fresh_name "x" in
          let y = fresh_name "y" in
          let msg =
            ("derive(Eq) for error type " ^ decl.tycon : Stdlib.String.t)
          in
          S.let_
            (S.pconstr decl.tycon [ (None, S.pvar x) ])
            lhs
            (S.let_
               (S.pconstr decl.tycon [ (None, S.pvar y) ])
               rhs
               (eq_of_typ ~msg typ (S.var x) (S.var y)))
      | Ptd_variant constrs | Ptd_error (Enum_payload constrs) ->
          let fallback_cases = [ (S.pany, S.const_bool false) ] in
          let cases =
            Lst.map_append constrs fallback_cases
              (fun { constr_name; constr_args; _ } ->
                let constr_name = constr_name.name in
                let make_result pat_args1 pat_args2 action =
                  ( S.ptuple
                      [
                        S.pconstr constr_name pat_args1;
                        S.pconstr constr_name pat_args2;
                      ],
                    action )
                    [@@inline]
                in
                match constr_args with
                | None -> make_result [] [] (S.const_bool true)
                | Some args ->
                    let msg =
                      Stdlib.String.concat ""
                        [
                          "derive(Eq) for constructor ";
                          constr_name;
                          " of type ";
                          decl.tycon;
                        ]
                    in
                    let names1 =
                      Lst.mapi args (fun i _ ->
                          fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
                    in
                    let names2 =
                      Lst.mapi args (fun i _ ->
                          fresh_name ("y" ^ Int.to_string i : Stdlib.String.t))
                    in
                    let action =
                      all_eq args names1 names2
                        ~eq_for_elem:(fun cparam lhs rhs ->
                          eq_of_typ ~msg cparam.cparam_typ (S.var lhs)
                            (S.var rhs))
                    in
                    let pat_args1 =
                      Lst.map2 args names1 (fun cparam name ->
                          match cparam.cparam_label with
                          | None -> (None, S.pvar name)
                          | Some label -> (Some label.label_name, S.pvar name))
                    in
                    let pat_args2 =
                      Lst.map2 args names2 (fun cparam name ->
                          match cparam.cparam_label with
                          | None -> (None, S.pvar name)
                          | Some label -> (Some label.label_name, S.pvar name))
                    in
                    make_result pat_args1 pat_args2 action)
          in
          S.match_ (S.tuple [ lhs; rhs ]) cases
      | Ptd_record [] -> S.const_bool true
      | Ptd_record (field0 :: fields) ->
          let eq_of_field (field : Syntax.field_decl) =
            let field_name = field.field_name.label in
            eq_of_typ
              ~msg:
                (Stdlib.String.concat ""
                   [
                     "derive(Eq) for field ";
                     field_name;
                     " of type ";
                     decl.tycon;
                   ])
              field.field_ty (S.field lhs field_name) (S.field rhs field_name)
          in
          Lst.fold_left fields (eq_of_field field0) (fun acc field ->
              S.infix_op "&&" acc (eq_of_field field))
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_compare (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  let rec compare_all tys values1 values2 ~cmp_for_elem =
    match (tys, values1, values2) with
    | [], [], [] -> S.const_int "0"
    | ty :: [], v1 :: [], v2 :: [] -> cmp_for_elem ty v1 v2
    | ty :: tys, v1 :: values1, v2 :: values2 ->
        let tmp = fresh_name "ord" in
        S.match_ (cmp_for_elem ty v1 v2)
          [
            ( S.pconstant (Const_int "0"),
              compare_all tys values1 values2 ~cmp_for_elem );
            (S.pvar tmp, S.var tmp);
          ]
    | _ -> assert false
  in
  let rec compare_of_typ ~msg (typ : Syntax.typ) (lhs : Syntax.expr)
      (rhs : Syntax.expr) =
    match typ with
    | Ptype_tuple { tys; _ } ->
        let names1 =
          Lst.mapi tys (fun i _ ->
              fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
        in
        let names2 =
          Lst.mapi tys (fun i _ ->
              fresh_name ("y" ^ Int.to_string i : Stdlib.String.t))
        in
        S.let_
          (S.ptuple (Lst.map names1 S.pvar))
          lhs
          (S.let_
             (S.ptuple (Lst.map names2 S.pvar))
             rhs
             (compare_all tys names1 names2
                ~cmp_for_elem:(fun typ name1 name2 ->
                  compare_of_typ ~msg typ (S.var name1) (S.var name2))))
    | typ ->
        S.apply_trait_method ~assertions ~typ
          ~loc:(Syntax.loc_of_type_expression typ)
          ~msg trait.name "compare" [ lhs; rhs ] No_attr
  in
  match[@warning "-fragile-match"] params with
  | [ lhs; rhs ] -> (
      let lhs = S.var lhs in
      let rhs = S.var rhs in
      Vec.push assertions
        {
          assert_type =
            S.typ decl.tycon
              (Lst.map decl.params (fun { tvar_name; _ } ->
                   match tvar_name with
                   | None -> S.typ_any
                   | Some tvar_name -> S.typ tvar_name []));
          assert_trait = Lident "Eq";
          assert_loc = trait.loc_;
          assert_msg =
            ("derive(Compare) of type " ^ decl.tycon : Stdlib.String.t);
        };
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          let lhs = S.newtype_field lhs in
          let rhs = S.newtype_field rhs in
          compare_of_typ
            ~msg:("derive(Compare) for newtype " ^ decl.tycon : Stdlib.String.t)
            typ lhs rhs
      | Ptd_error No_payload -> S.const_int "0"
      | Ptd_error (Single_payload typ) ->
          let x = fresh_name "x" in
          let y = fresh_name "y" in
          let msg =
            ("derive(Compare) for error type " ^ decl.tycon : Stdlib.String.t)
          in
          S.let_
            (S.pconstr decl.tycon [ (None, S.pvar x) ])
            lhs
            (S.let_
               (S.pconstr decl.tycon [ (None, S.pvar y) ])
               rhs
               (compare_of_typ ~msg typ (S.var x) (S.var y)))
      | Ptd_variant constrs | Ptd_error (Enum_payload constrs) ->
          let constr_count = List.length constrs in
          let cases =
            Lst.mapi constrs
              (fun constr_index { constr_name; constr_args; _ } ->
                let constr_name = constr_name.name in
                let constr_args = Option.value constr_args ~default:[] in
                let arg_names =
                  Lst.mapi constr_args (fun i _ ->
                      fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
                in
                let arg_pats =
                  Lst.map2 arg_names constr_args
                    (fun name { cparam_label; _ } ->
                      match cparam_label with
                      | None -> (None, S.pvar name)
                      | Some label -> (Some label.label_name, S.pvar name))
                in
                let eq_case =
                  let arg_names' =
                    Lst.mapi constr_args (fun i _ ->
                        fresh_name ("y" ^ Int.to_string i : Stdlib.String.t))
                  in
                  let arg_pats' =
                    Lst.map2 arg_names' constr_args
                      (fun name { cparam_label; _ } ->
                        match cparam_label with
                        | None -> (None, S.pvar name)
                        | Some label -> (Some label.label_name, S.pvar name))
                  in
                  let msg =
                    Stdlib.String.concat ""
                      [
                        "derive(Compare) for constructor ";
                        constr_name;
                        " of type ";
                        decl.tycon;
                      ]
                  in
                  ( S.pconstr constr_name arg_pats',
                    compare_all constr_args arg_names arg_names'
                      ~cmp_for_elem:(fun { cparam_typ; _ } name1 name2 ->
                        compare_of_typ ~msg cparam_typ (S.var name1)
                          (S.var name2)) )
                in
                let lt_case = (S.pany, S.const_int "-1") in
                let cases =
                  if constr_index = 0 then [ eq_case; lt_case ]
                  else
                    let pat =
                      Lst.map (Lst.take constr_index constrs)
                        (fun { constr_name; _ } ->
                          S.pconstr constr_name.name [ (None, S.pany) ])
                      |> S.por
                    in
                    let gt_case = (pat, S.const_int "1") in
                    if constr_index = constr_count - 1 then [ gt_case; eq_case ]
                    else [ gt_case; eq_case; lt_case ]
                in
                (S.pconstr constr_name arg_pats, S.match_ rhs cases))
          in
          S.match_ lhs cases
      | Ptd_record fields ->
          let lhs_fields =
            Lst.map fields (fun field -> S.field lhs field.field_name.label)
          in
          let rhs_fields =
            Lst.map fields (fun field -> S.field rhs field.field_name.label)
          in
          let cmp_of_field (field : Syntax.field_decl) lhs rhs =
            let field_name = field.field_name.label in
            compare_of_typ
              ~msg:
                (Stdlib.String.concat ""
                   [
                     "derive(Compare) for field ";
                     field_name;
                     " of type ";
                     decl.tycon;
                   ])
              field.field_ty lhs rhs
          in
          compare_all fields lhs_fields rhs_fields ~cmp_for_elem:cmp_of_field
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_show (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  match[@warning "-fragile-match"] params with
  | [ obj; logger ] -> (
      let obj = S.var obj in
      let logger = S.var logger in
      let write_string ?(is_last = false) str self =
        if is_last then S.dot_apply self "write_string" [ S.const_string str ]
        else S.dotdot_apply self "write_string" [ S.const_string str ]
      in
      let rec write_list tys objs ~write_elem self =
        match (tys, objs) with
        | [], [] -> self
        | ty :: [], obj :: [] -> write_elem ty obj self
        | ty :: tys, obj :: objs ->
            self |> write_elem ty obj |> write_string ", "
            |> write_list tys objs ~write_elem
        | _ -> assert false
      in
      let rec write_object ~msg (typ : Syntax.typ) obj self =
        match typ with
        | Ptype_arrow _ -> self |> write_string "<function>"
        | Ptype_tuple { tys; _ } ->
            let names =
              Lst.mapi tys (fun i _ ->
                  fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
            in
            self |> write_string "("
            |> write_list tys names ~write_elem:(fun typ name self ->
                   self |> write_object ~msg typ (S.var name))
            |> write_string ")"
            |> S.let_ (S.ptuple (Lst.map names S.pvar)) obj
        | _ ->
            let assertion : Syntax.static_assertion =
              {
                assert_type = typ;
                assert_trait = trait.name;
                assert_loc = Syntax.loc_of_type_expression typ;
                assert_msg = msg;
              }
            in
            Vec.push assertions assertion;
            S.dotdot_apply self "write_object" [ obj ]
      in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          let msg =
            ("derive(Show) for newtype " ^ decl.tycon : Stdlib.String.t)
          in
          logger
          |> write_string (decl.tycon ^ "(")
          |> write_object ~msg typ (S.newtype_field obj)
          |> write_string ~is_last:true ")"
      | Ptd_error No_payload -> logger |> write_string ~is_last:true decl.tycon
      | Ptd_error (Single_payload typ) ->
          let var = fresh_name "err_payload" in
          let msg =
            ("derive(Show) for error type " ^ decl.tycon : Stdlib.String.t)
          in
          S.let_
            (S.pconstr decl.tycon [ (None, S.pvar var) ])
            obj
            (logger
            |> write_string (decl.tycon ^ "(")
            |> write_object ~msg typ (S.var var)
            |> write_string ~is_last:true ")")
      | Ptd_error (Enum_payload constrs) | Ptd_variant constrs ->
          Lst.map constrs (fun { constr_name; constr_args; _ } ->
              let constr_name = constr_name.name in
              match constr_args with
              | None ->
                  ( S.pconstr constr_name [],
                    logger |> write_string ~is_last:true constr_name )
              | Some args ->
                  let vars = Lst.map args (fun _ -> fresh_name "arg") in
                  let action =
                    let msg =
                      Stdlib.String.concat ""
                        [
                          "derive(Show) for constructor ";
                          constr_name;
                          " of type ";
                          decl.tycon;
                        ]
                    in
                    logger
                    |> write_string (constr_name ^ "(")
                    |> write_list args vars
                         ~write_elem:(fun constr_param var self ->
                           match constr_param.cparam_label with
                           | None ->
                               self
                               |> write_object ~msg constr_param.cparam_typ
                                    (S.var var)
                           | Some label ->
                               self
                               |> write_string (label.label_name ^ "=")
                               |> write_object ~msg constr_param.cparam_typ
                                    (S.var var))
                    |> write_string ~is_last:true ")"
                  in
                  ( S.pconstr constr_name
                      (Lst.map2 args vars (fun cparam var ->
                           let label =
                             Option.map
                               (fun (l : Syntax.label) -> l.label_name)
                               cparam.cparam_label
                           in
                           (label, S.pvar var))),
                    action ))
          |> S.match_ obj
      | Ptd_record fields ->
          let write_field (field : Syntax.field_decl) _ self =
            let field_name = field.field_name.label in
            let msg =
              Stdlib.String.concat ""
                [
                  "derive(Show) for field "; field_name; " of type "; decl.tycon;
                ]
            in
            self
            |> write_string (field_name ^ ": ")
            |> write_object ~msg field.field_ty (S.field obj field_name)
          in
          logger |> write_string "{"
          |> write_list fields fields ~write_elem:write_field
          |> write_string ~is_last:true "}"
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_hash (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  match[@warning "-fragile-match"] params with
  | [ obj; hasher ] -> (
      let obj = S.var obj in
      let hasher = S.var hasher in
      let rec hash_of_typ ~msg (typ : Syntax.typ) (obj : Syntax.expr) =
        match typ with
        | Ptype_tuple { tys; _ } ->
            let names =
              Lst.mapi tys (fun i _ ->
                  fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
            in
            Lst.map2 tys names (fun typ name ->
                hash_of_typ ~msg typ (S.var name))
            |> S.seq
            |> S.let_ (S.ptuple (Lst.map names S.pvar)) obj
        | typ ->
            S.apply_trait_method ~assertions ~typ
              ~loc:(Syntax.loc_of_type_expression typ)
              ~msg trait.name "hash_combine" [ obj; hasher ] No_attr
      in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          let msg =
            ("derive(Hash) for newtype " ^ decl.tycon : Stdlib.String.t)
          in
          hash_of_typ ~msg typ (S.newtype_field obj)
      | Ptd_error No_payload ->
          S.dot_apply hasher "combine_int" [ S.const_int "0" ]
      | Ptd_error (Single_payload typ) ->
          let var = fresh_name "err_payload" in
          let msg =
            ("derive(Hash) for error type " ^ decl.tycon : Stdlib.String.t)
          in
          S.let_
            (S.pconstr decl.tycon [ (None, S.pvar var) ])
            obj
            (S.sequence
               (S.dot_apply hasher "combine_int" [ S.const_int "0" ])
               (hash_of_typ ~msg typ (S.var var)))
      | Ptd_error (Enum_payload constrs) | Ptd_variant constrs ->
          Lst.mapi constrs (fun constr_index { constr_name; constr_args; _ } ->
              let constr_name = constr_name.name in
              let args = Option.value ~default:[] constr_args in
              let vars = Lst.map args (fun _ -> fresh_name "arg") in
              let action =
                let msg =
                  Stdlib.String.concat ""
                    [
                      "derive(Hash) for constructor ";
                      constr_name;
                      " of type ";
                      decl.tycon;
                    ]
                in
                S.dot_apply hasher "combine_int"
                  [ S.const_int (string_of_int constr_index) ]
                :: Lst.map2 args vars (fun cparam var ->
                       hash_of_typ ~msg cparam.cparam_typ (S.var var))
                |> S.seq
              in
              ( S.pconstr constr_name
                  (Lst.map2 args vars (fun cparam var ->
                       let label =
                         Option.map
                           (fun (l : Syntax.label) -> l.label_name)
                           cparam.cparam_label
                       in
                       (label, S.pvar var))),
                action ))
          |> S.match_ obj
      | Ptd_record fields ->
          Lst.map fields (fun field ->
              let field_name = field.field_name.label in
              hash_of_typ
                ~msg:
                  (Stdlib.String.concat ""
                     [
                       "derive(Hash) for field ";
                       field_name;
                       " of type ";
                       decl.tycon;
                     ])
                field.field_ty (S.field obj field_name))
          |> S.seq
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_to_json (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  let rec to_json_of_typ ~msg (typ : Syntax.typ) (obj : Syntax.expr) =
    match typ with
    | Ptype_tuple { tys; _ } ->
        let names =
          Lst.mapi tys (fun i _ ->
              fresh_name ("x" ^ Int.to_string i : Stdlib.String.t))
        in
        S.let_
          (S.ptuple (Lst.map names S.pvar))
          obj
          (S.array
             (Lst.map2 tys names (fun ty v -> to_json_of_typ ~msg ty (S.var v))))
    | typ ->
        S.apply_trait_method ~assertions ~typ
          ~loc:(Syntax.loc_of_type_expression typ)
          ~msg trait.name "to_json" [ obj ] No_attr
  in
  match[@warning "-fragile-match"] params with
  | self :: [] -> (
      let self = S.var self in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          to_json_of_typ
            ~msg:("derive(ToJson) for newtype " ^ decl.tycon : Stdlib.String.t)
            typ (S.newtype_field self)
      | Ptd_error No_payload ->
          S.map_string_expr [ ("$tag", S.const_string decl.tycon) ]
      | Ptd_error (Single_payload typ) ->
          let var = fresh_name "err_payload" in
          let to_json_for_payload =
            to_json_of_typ
              ~msg:
                ("derive(ToJson) for error type " ^ decl.tycon
                  : Stdlib.String.t)
              typ (S.var var)
          in
          S.let_
            (S.pconstr decl.tycon [ (None, S.pvar var) ])
            self
            (S.map_string_expr
               [
                 ("$tag", S.const_string decl.tycon); ("0", to_json_for_payload);
               ])
      | Ptd_error (Enum_payload constrs) | Ptd_variant constrs ->
          Lst.map constrs (fun { constr_name; constr_args; _ } ->
              let constr_name = constr_name.name in
              let args = Option.value ~default:[] constr_args in
              let vars = Lst.map args (fun _ -> fresh_name "arg") in
              let action =
                let positional_index = ref (-1) in
                Lst.map2 args vars (fun cparam var ->
                    let arg =
                      to_json_of_typ
                        ~msg:
                          (Stdlib.String.concat ""
                             [
                               "derive(ToJson) for constructor ";
                               constr_name;
                               " of type ";
                               decl.tycon;
                             ])
                        cparam.cparam_typ (S.var var)
                    in
                    match cparam.cparam_label with
                    | None ->
                        incr positional_index;
                        (string_of_int !positional_index, arg)
                    | Some label -> (label.label_name, arg))
                |> List.cons ("$tag", S.const_string constr_name)
                |> S.map_string_expr
              in
              ( S.pconstr constr_name
                  (Lst.map2 args vars (fun cparam var ->
                       let label =
                         Option.map
                           (fun (l : Syntax.label) -> l.label_name)
                           cparam.cparam_label
                       in
                       (label, S.pvar var))),
                action ))
          |> S.match_ self
      | Ptd_record fields ->
          let field_is_option_match_cases (field : Syntax.field_decl) =
            let field_name = field.field_name.label in
            let expr = S.field self field_name in
            [
              (S.pconstr "None" [], S.unit);
              ( S.pconstr "Some" [ (None, S.pvar "v") ],
                S.map_insert_json field_name
                  (to_json_of_typ
                     ~msg:
                       (Stdlib.String.concat ""
                          [
                            "derive(ToJson) for field ";
                            field_name;
                            " of type ";
                            decl.tycon;
                          ])
                     field.field_ty (S.var "v")) );
            ]
            |> S.match_ expr
          in
          let map_var = S.var "$map" in
          let expr_lst =
            Lst.map fields (fun field ->
                if S.is_option field then field_is_option_match_cases field
                else
                  let field_name = field.field_name.label in
                  S.map_insert_json field_name
                    (to_json_of_typ
                       ~msg:
                         (Stdlib.String.concat ""
                            [
                              "derive(ToJson) for field ";
                              field.field_name.label;
                              " of type ";
                              decl.tycon;
                            ])
                       field.field_ty (S.field self field_name)))
          in
          let obj_constr = S.constr "Object" [ map_var ] in
          let string_to_json_map_typ =
            S.typ "Map" [ S.typ "String" []; S.typ "Json" [] ]
          in
          S.let_ (S.pvar "$map")
            (S.annotation (S.map_string_expr []) string_to_json_map_typ)
            (Lst.fold_right expr_lst obj_constr S.sequence)
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_arbitrary (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  match[@warning "-fragile-match"] params with
  | [ size; rng ] -> (
      let size, rng = (S.var size, S.var rng) in
      let rec arbitrary_of_typ ~msg (typ : Syntax.typ) =
        match typ with
        | Ptype_tuple { tys; _ } ->
            S.tuple (Lst.map tys (arbitrary_of_typ ~msg))
        | typ ->
            S.apply_trait_method ~assertions ~typ ~msg
              ~loc:(Syntax.loc_of_type_expression typ)
              trait.name "arbitrary" [ size; rng ] No_attr
      in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          arbitrary_of_typ
            ~msg:
              ("derive(Arbitrary) for newtype " ^ decl.tycon : Stdlib.String.t)
            typ
      | Ptd_error No_payload -> S.constr decl.tycon []
      | Ptd_error (Single_payload typ) ->
          S.constr decl.tycon
            [
              arbitrary_of_typ
                ~msg:
                  ("derive(Arbitrary) for error type " ^ decl.tycon
                    : Stdlib.String.t)
                typ;
            ]
      | Ptd_error (Enum_payload constrs) | Ptd_variant constrs -> (
          match constrs with
          | [] ->
              Local_diagnostics.add_error diagnostics
                (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
                   ~reason:"cannot find a constant constructor" ~loc:trait.loc_);
              S.hole
          | constr :: [] ->
              let constr_name = constr.constr_name.name in
              let args = Option.value ~default:[] constr.constr_args in
              S.constr_label constr_name
                (Lst.map args (fun arg ->
                     ( arbitrary_of_typ
                         ~msg:
                           (Stdlib.String.concat ""
                              [
                                "derive(Arbitrary) for constructor ";
                                constr_name;
                                " of type ";
                                decl.tycon;
                              ])
                         arg.cparam_typ,
                       arg.cparam_label )))
          | multi ->
              let exp =
                S.infix_op "%"
                  (S.infix_op "+"
                     (S.dot_apply rng "next_uint" [])
                     (S.const_uint "1"))
                  (S.const_uint (string_of_int (List.length multi)))
              in
              Lst.mapi multi (fun i { constr_name; constr_args; _ } ->
                  let constr_name = constr_name.name in
                  let args = Option.value ~default:[] constr_args in
                  let action =
                    S.constr_label constr_name
                      (Lst.map args (fun arg ->
                           ( arbitrary_of_typ
                               ~msg:
                                 (Stdlib.String.concat ""
                                    [
                                      "derive(Arbitrary) for constructor ";
                                      constr_name;
                                      " of type ";
                                      decl.tycon;
                                    ])
                               arg.cparam_typ,
                             arg.cparam_label )))
                  in
                  (S.const_uint_pat (string_of_int i), action))
              |> S.match_ exp)
      | Ptd_record fields ->
          Lst.map fields (fun field ->
              let field_name = field.field_name.label in
              ( field_name,
                arbitrary_of_typ
                  ~msg:
                    (Stdlib.String.concat ""
                       [
                         "derive(Arbitrary) for field ";
                         field_name;
                         " of type ";
                         decl.tycon;
                       ])
                  field.field_ty ))
          |> S.record
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let derive_from_json (trait : Syntax.type_name) (decl : Syntax.type_decl)
    ~(params : string list) ~assertions ~diagnostics =
  match[@warning "-fragile-match"] params with
  | [ json; path ] -> (
      let json, path = (S.var json, S.var path) in
      let from_json_of_typ ~msg (typ : Syntax.typ) (json : Syntax.expr)
          (path : Syntax.expr) : Syntax.expr =
        S.apply_trait_method ~assertions ~typ ~msg
          ~loc:(Syntax.loc_of_type_expression typ)
          trait.name "from_json" [ json; path ] Exclamation
      in
      let err_case =
        S.rest_raise_json_decode_error_ path
          (S.const_string ("invalid JSON for " ^ decl.tycon : Stdlib.String.t))
      in
      match decl.components with
      | Ptd_abstract ->
          Local_diagnostics.add_error diagnostics
            (Errors.cannot_derive ~tycon:decl.tycon ~trait:trait.name
               ~reason:"target type is abstract" ~loc:trait.loc_);
          S.hole
      | Ptd_newtype typ ->
          S.constr decl.tycon
            [
              S.annotation
                (from_json_of_typ
                   ~msg:
                     ("derive(FromJson) for newtype " ^ decl.tycon
                       : Stdlib.String.t)
                   typ json path)
                typ;
            ]
      | Ptd_error (Single_payload typ) ->
          let vars = fresh_name "err_payload" in
          let pattern =
            [
              ( "$tag",
                S.pconstant
                  (Const_string
                     { string_val = decl.tycon; string_repr = decl.tycon }) );
              ("0", S.pvar vars);
            ]
            |> S.map_pat_elem_lst
          in
          let expr =
            let updated_path =
              S.apply
                (S.method_ (Ldot { pkg = "json"; id = "JsonPath" }) "add_index")
                [ path; S.const_int "0" ]
                No_attr
            in
            S.constr decl.tycon
              [
                S.annotation
                  (from_json_of_typ
                     ~msg:
                       ("derive(FromJson) for error type " ^ decl.tycon
                         : Stdlib.String.t)
                     typ (S.var vars) updated_path)
                  typ;
              ]
          in
          [ (S.pmap pattern, expr); err_case ] |> S.match_ json
      | Ptd_error No_payload ->
          let pattern =
            [
              ( "$tag",
                S.pconstant
                  (Const_string
                     { string_val = decl.tycon; string_repr = decl.tycon }) );
            ]
            |> S.map_pat_elem_lst
          in
          let expr = S.constr decl.tycon [] in
          [ (S.pmap pattern, expr); err_case ] |> S.match_ json
      | Ptd_error (Enum_payload constrs) | Ptd_variant constrs ->
          let map_cases =
            Lst.map constrs (fun { constr_name; constr_args; _ } ->
                let constr_name = constr_name.name in
                let args = Option.value ~default:[] constr_args in
                let vars = Lst.map args (fun _ -> fresh_name "arg") in
                let pattern =
                  let positional_index = ref (-1) in
                  Lst.map2 args vars (fun cparam var ->
                      let p = S.pvar var in
                      match cparam.cparam_label with
                      | None ->
                          incr positional_index;
                          (string_of_int !positional_index, p)
                      | Some label -> (label.label_name, p))
                  |> List.cons
                       ( "$tag",
                         S.pconstant
                           (Const_string
                              {
                                string_val = constr_name;
                                string_repr = constr_name;
                              }) )
                  |> S.map_pat_elem_lst
                in
                let expr =
                  let positional_index = ref (-1) in
                  Lst.map2 args vars (fun cparam var ->
                      let updated_path, label_opt =
                        match cparam.cparam_label with
                        | None ->
                            incr positional_index;
                            let path =
                              S.apply
                                (S.method_
                                   (Ldot { pkg = "json"; id = "JsonPath" })
                                   "add_index")
                                [
                                  path;
                                  S.const_int (string_of_int !positional_index);
                                ]
                                No_attr
                            in
                            (path, None)
                        | Some label ->
                            let path =
                              S.apply
                                (S.method_
                                   (Ldot { pkg = "json"; id = "JsonPath" })
                                   "add_key")
                                [ path; S.const_string label.label_name ]
                                No_attr
                            in
                            (path, Some label)
                      in
                      let typ = cparam.cparam_typ in
                      ( S.annotation
                          (from_json_of_typ
                             ~msg:
                               (Stdlib.String.concat ""
                                  [
                                    "derive(FromJson) for constructor ";
                                    constr_name;
                                    " of type ";
                                    decl.tycon;
                                  ])
                             typ (S.var var) updated_path)
                          typ,
                        label_opt ))
                  |> S.constr_label constr_name
                in
                (S.pmap pattern, expr))
          in
          Lst.append_one map_cases err_case |> S.match_ json
      | Ptd_record fields ->
          let vars = Lst.map fields (fun _ -> fresh_name "field") in
          let pattern =
            Lst.map2 fields vars (fun field var ->
                let field_name = field.field_name.label in
                let match_absent = S.is_option field in
                Syntax.Map_pat_elem
                  {
                    key =
                      Const_string
                        { string_val = field_name; string_repr = field_name };
                    pat = S.pvar var;
                    match_absent;
                    key_loc_ = ghost_loc_;
                    loc_ = ghost_loc_;
                  })
          in
          let expr =
            S.record
              (Lst.map2 fields vars (fun field var ->
                   let field_name = field.field_name.label in
                   let updated_path =
                     S.apply
                       (S.method_
                          (Ldot { pkg = "json"; id = "JsonPath" })
                          "add_key")
                       [ path; S.const_string field_name ]
                       No_attr
                   in
                   let typ = field.field_ty in
                   let field_expr =
                     if S.is_option field then
                       S.match_ (S.var var)
                         [
                           ( S.pconstr "Some" [ (None, S.pvar "v") ],
                             S.annotation
                               (S.constr "Some"
                                  [
                                    from_json_of_typ
                                      ~msg:
                                        (Stdlib.String.concat ""
                                           [
                                             "derive(FromJson) for field ";
                                             field_name;
                                             " of type ";
                                             decl.tycon;
                                           ])
                                      typ (S.var "v") updated_path;
                                  ])
                               typ );
                           (S.pconstr "None" [], S.constr "None" []);
                         ]
                     else
                       S.annotation
                         (from_json_of_typ
                            ~msg:
                              (Stdlib.String.concat ""
                                 [
                                   "derive(FromJson) for field ";
                                   field_name;
                                   " of type ";
                                   decl.tycon;
                                 ])
                            typ (S.var var) updated_path)
                         typ
                   in
                   (field_name, field_expr)))
          in
          let map_cases = (S.pmap pattern, expr) in
          [ map_cases; err_case ] |> S.match_ json
      | Ptd_alias _ -> S.hole)
  | _ -> assert false

let deny_all_derive_args_then f (directive : Syntax.deriving_directive)
    (decl : Syntax.type_decl) ~(params : string list) ~assertions ~diagnostics =
  let trait = directive.type_name_ in
  if
    Derive_args.deny_all_args ~host_type:decl.tycon ~trait_name:trait.name
      diagnostics directive
  then S.hole
  else f directive.type_name_ decl ~params ~assertions ~diagnostics

let derivers : (string * deriver) list Type_path.Hash.t =
  Type_path.Hash.of_list
    [
      ( Type_path.Builtin.trait_default,
        [ ("default", deny_all_derive_args_then derive_default) ] );
      ( Type_path.Builtin.trait_eq,
        [
          ( Operators.op_equal_info.method_name,
            deny_all_derive_args_then derive_eq );
        ] );
      ( Type_path.Builtin.trait_compare,
        [ ("compare", deny_all_derive_args_then derive_compare) ] );
      ( Type_path.Builtin.trait_show,
        [ ("output", deny_all_derive_args_then derive_show) ] );
      ( Type_path.Builtin.trait_hash,
        [ ("hash_combine", deny_all_derive_args_then derive_hash) ] );
      ( Type_path.toplevel_type ~pkg:Basic_config.builtin_package "ToJson",
        [ ("to_json", deny_all_derive_args_then derive_to_json) ] );
      ( Type_path.toplevel_type ~pkg:"moonbitlang/core/quickcheck" "Arbitrary",
        [ ("arbitrary", deny_all_derive_args_then derive_arbitrary) ] );
      ( Type_path.toplevel_type ~pkg:"moonbitlang/core/json" "FromJson",
        [ ("from_json", deny_all_derive_args_then derive_from_json) ] );
    ]
