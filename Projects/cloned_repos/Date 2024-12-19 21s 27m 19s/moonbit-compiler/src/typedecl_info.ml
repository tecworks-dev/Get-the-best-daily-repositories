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
module Constr_info = Basic_constr_info
module Lst = Basic_lst

type typ = Stype.t

include struct
  let _ = fun (_ : typ) -> ()
  let sexp_of_typ = (Stype.sexp_of_t : typ -> S.t)
  let _ = sexp_of_typ
end

type constr_tag = Constr_info.constr_tag

include struct
  let _ = fun (_ : constr_tag) -> ()
  let sexp_of_constr_tag = (Constr_info.sexp_of_constr_tag : constr_tag -> S.t)
  let _ = sexp_of_constr_tag
end

type path = Type_path.t

include struct
  let _ = fun (_ : path) -> ()
  let sexp_of_path = (Type_path.sexp_of_t : path -> S.t)
  let _ = sexp_of_path
end

type location = Loc.t

include struct
  let _ = fun (_ : location) -> ()
  let sexp_of_location = (Loc.sexp_of_t : location -> S.t)
  let _ = sexp_of_location
end

type fn_arity = Fn_arity.t

include struct
  let _ = fun (_ : fn_arity) -> ()
  let sexp_of_fn_arity = (Fn_arity.sexp_of_t : fn_arity -> S.t)
  let _ = sexp_of_fn_arity
end

type tvar_env = Tvar_env.t

include struct
  let _ = fun (_ : tvar_env) -> ()
  let sexp_of_tvar_env = (Tvar_env.sexp_of_t : tvar_env -> S.t)
  let _ = sexp_of_tvar_env
end

type docstring = Docstring.t

include struct
  let _ = fun (_ : docstring) -> ()
  let sexp_of_docstring = (Docstring.sexp_of_t : docstring -> S.t)
  let _ = sexp_of_docstring
end

type visibility = Vis_priv | Vis_default | Vis_readonly | Vis_fully_pub

include struct
  let _ = fun (_ : visibility) -> ()

  let sexp_of_visibility =
    (function
     | Vis_priv -> S.Atom "Vis_priv"
     | Vis_default -> S.Atom "Vis_default"
     | Vis_readonly -> S.Atom "Vis_readonly"
     | Vis_fully_pub -> S.Atom "Vis_fully_pub"
      : visibility -> S.t)

  let _ = sexp_of_visibility
end

let vis_is_pub vis =
  match vis with
  | Vis_priv | Vis_default -> false
  | Vis_readonly | Vis_fully_pub -> true

let hide_loc _ = not !Basic_config.show_loc

class ['a] iterbase =
  object (self)
    method visit_typ : 'a -> typ -> unit = fun _ _ -> ()

    method private visit_type_constraint
        : 'a -> Tvar_env.type_constraint -> unit =
      fun _ _ -> ()

    method private visit_constr_tag : 'a -> Constr_info.constr_tag -> unit =
      fun _ _ -> ()

    method private visit_tvar_env : 'a -> tvar_env -> unit =
      fun ctx env ->
        Tvar_env.iter env (fun tvar_info ->
            self#visit_typ ctx tvar_info.typ;
            Lst.iter tvar_info.constraints (self#visit_type_constraint ctx))
  end

type t = {
  ty_constr : path;
  ty_arity : int;
  ty_desc : type_components;
  ty_vis : visibility; [@sexp_drop_if fun vis -> vis = Vis_default]
  ty_params_ : tvar_env;
  ty_loc_ : location;
  ty_doc_ : docstring; [@sexp_drop_if Docstring.is_empty]
  ty_is_only_tag_enum_ : bool; [@sexp_drop_if fun x -> x = false]
  ty_is_suberror_ : bool; [@sexp_drop_if fun x -> x = false]
}

and constructors = constructor list
and fields = field list

and newtype_info = {
  newtype_constr : constructor;
  underlying_typ : typ;
  recursive : bool; [@sexp_drop_if fun recur -> not recur]
}

and type_components =
  | Extern_type
  | Abstract_type
  | Error_type of constructor
  | ErrorEnum_type of constructor list
  | New_type of newtype_info
  | Variant_type of constructor list
  | Record_type of { fields : field list; has_private_field_ : bool }

and type_component_visibility = Invisible | Readable | Read_write

and constructor = {
  constr_name : string;
  cs_args : typ list;
  cs_res : typ;
  cs_tag : constr_tag;
  cs_vis : type_component_visibility; [@sexp_drop_if fun vis -> vis = Invisible]
  cs_ty_params_ : tvar_env;
  cs_arity_ : fn_arity; [@sexp_drop_if Fn_arity.is_simple]
  cs_constr_loc_ : location; [@sexp_drop_if hide_loc]
  cs_loc_ : location; [@sexp_drop_if hide_loc]
}

and field = {
  field_name : string;
  pos : int;
  ty_field : typ;
  ty_record : typ;
  mut : bool;
  vis : type_component_visibility; [@sexp_drop_if fun vis -> vis = Invisible]
  all_labels : string list;
  ty_params_ : tvar_env;
  label_loc_ : location; [@sexp_drop_if hide_loc]
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  [@@@ocaml.warning "-4-26-27"]
  [@@@VISITORS.BEGIN]

  class virtual ['self] iter =
    object (self : 'self)
      inherit [_] iterbase

      method visit_t : _ -> t -> unit =
        fun env _visitors_this ->
          let _visitors_r0 =
            (fun _visitors_this -> ()) _visitors_this.ty_constr
          in
          let _visitors_r1 =
            (fun _visitors_this -> ()) _visitors_this.ty_arity
          in
          let _visitors_r2 =
            self#visit_type_components env _visitors_this.ty_desc
          in
          let _visitors_r3 = (fun _visitors_this -> ()) _visitors_this.ty_vis in
          let _visitors_r4 =
            self#visit_tvar_env env _visitors_this.ty_params_
          in
          let _visitors_r5 =
            (fun _visitors_this -> ()) _visitors_this.ty_loc_
          in
          let _visitors_r6 =
            (fun _visitors_this -> ()) _visitors_this.ty_doc_
          in
          let _visitors_r7 =
            (fun _visitors_this -> ()) _visitors_this.ty_is_only_tag_enum_
          in
          let _visitors_r8 =
            (fun _visitors_this -> ()) _visitors_this.ty_is_suberror_
          in
          ()

      method private visit_constructors : _ -> constructors -> unit =
        fun env _visitors_this ->
          Basic_lst.iter _visitors_this (self#visit_constructor env)

      method private visit_fields : _ -> fields -> unit =
        fun env _visitors_this ->
          Basic_lst.iter _visitors_this (self#visit_field env)

      method private visit_newtype_info : _ -> newtype_info -> unit =
        fun env _visitors_this ->
          let _visitors_r0 =
            self#visit_constructor env _visitors_this.newtype_constr
          in
          let _visitors_r1 = self#visit_typ env _visitors_this.underlying_typ in
          let _visitors_r2 =
            (fun _visitors_this -> ()) _visitors_this.recursive
          in
          ()

      method private visit_Extern_type : _ -> unit = fun env -> ()
      method private visit_Abstract_type : _ -> unit = fun env -> ()

      method private visit_Error_type : _ -> constructor -> unit =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_constructor env _visitors_c0 in
          ()

      method private visit_ErrorEnum_type : _ -> constructor list -> unit =
        fun env _visitors_c0 ->
          let _visitors_r0 =
            (fun _visitors_this ->
              Basic_lst.iter _visitors_this (self#visit_constructor env))
              _visitors_c0
          in
          ()

      method private visit_New_type : _ -> newtype_info -> unit =
        fun env _visitors_c0 ->
          let _visitors_r0 = self#visit_newtype_info env _visitors_c0 in
          ()

      method private visit_Variant_type : _ -> constructor list -> unit =
        fun env _visitors_c0 ->
          let _visitors_r0 =
            (fun _visitors_this ->
              Basic_lst.iter _visitors_this (self#visit_constructor env))
              _visitors_c0
          in
          ()

      method private visit_Record_type : _ -> field list -> bool -> unit =
        fun env _visitors_ffields _visitors_fhas_private_field_ ->
          let _visitors_r0 =
            (fun _visitors_this ->
              Basic_lst.iter _visitors_this (self#visit_field env))
              _visitors_ffields
          in
          let _visitors_r1 =
            (fun _visitors_this -> ()) _visitors_fhas_private_field_
          in
          ()

      method private visit_type_components : _ -> type_components -> unit =
        fun env _visitors_this ->
          match _visitors_this with
          | Extern_type -> self#visit_Extern_type env
          | Abstract_type -> self#visit_Abstract_type env
          | Error_type _visitors_c0 -> self#visit_Error_type env _visitors_c0
          | ErrorEnum_type _visitors_c0 ->
              self#visit_ErrorEnum_type env _visitors_c0
          | New_type _visitors_c0 -> self#visit_New_type env _visitors_c0
          | Variant_type _visitors_c0 ->
              self#visit_Variant_type env _visitors_c0
          | Record_type
              {
                fields = _visitors_ffields;
                has_private_field_ = _visitors_fhas_private_field_;
              } ->
              self#visit_Record_type env _visitors_ffields
                _visitors_fhas_private_field_

      method private visit_Invisible : _ -> unit = fun env -> ()
      method private visit_Readable : _ -> unit = fun env -> ()
      method private visit_Read_write : _ -> unit = fun env -> ()

      method private visit_type_component_visibility
          : _ -> type_component_visibility -> unit =
        fun env _visitors_this ->
          match _visitors_this with
          | Invisible -> self#visit_Invisible env
          | Readable -> self#visit_Readable env
          | Read_write -> self#visit_Read_write env

      method private visit_constructor : _ -> constructor -> unit =
        fun env _visitors_this ->
          let _visitors_r0 =
            (fun _visitors_this -> ()) _visitors_this.constr_name
          in
          let _visitors_r1 =
            (fun _visitors_this ->
              Basic_lst.iter _visitors_this (self#visit_typ env))
              _visitors_this.cs_args
          in
          let _visitors_r2 = self#visit_typ env _visitors_this.cs_res in
          let _visitors_r3 = self#visit_constr_tag env _visitors_this.cs_tag in
          let _visitors_r4 =
            self#visit_type_component_visibility env _visitors_this.cs_vis
          in
          let _visitors_r5 =
            self#visit_tvar_env env _visitors_this.cs_ty_params_
          in
          let _visitors_r6 =
            (fun _visitors_this -> ()) _visitors_this.cs_arity_
          in
          let _visitors_r7 =
            (fun _visitors_this -> ()) _visitors_this.cs_constr_loc_
          in
          let _visitors_r8 =
            (fun _visitors_this -> ()) _visitors_this.cs_loc_
          in
          ()

      method private visit_field : _ -> field -> unit =
        fun env _visitors_this ->
          let _visitors_r0 =
            (fun _visitors_this -> ()) _visitors_this.field_name
          in
          let _visitors_r1 = (fun _visitors_this -> ()) _visitors_this.pos in
          let _visitors_r2 = self#visit_typ env _visitors_this.ty_field in
          let _visitors_r3 = self#visit_typ env _visitors_this.ty_record in
          let _visitors_r4 = (fun _visitors_this -> ()) _visitors_this.mut in
          let _visitors_r5 =
            self#visit_type_component_visibility env _visitors_this.vis
          in
          let _visitors_r6 =
            (fun _visitors_this ->
              Basic_lst.iter _visitors_this (fun _visitors_this -> ()))
              _visitors_this.all_labels
          in
          let _visitors_r7 =
            self#visit_tvar_env env _visitors_this.ty_params_
          in
          let _visitors_r8 =
            (fun _visitors_this -> ()) _visitors_this.label_loc_
          in
          let _visitors_r9 = (fun _visitors_this -> ()) _visitors_this.loc_ in
          ()
    end

  [@@@VISITORS.END]
end

include struct
  let _ = fun (_ : t) -> ()
  let _ = fun (_ : constructors) -> ()
  let _ = fun (_ : fields) -> ()
  let _ = fun (_ : newtype_info) -> ()
  let _ = fun (_ : type_components) -> ()
  let _ = fun (_ : type_component_visibility) -> ()
  let _ = fun (_ : constructor) -> ()
  let _ = fun (_ : field) -> ()

  let rec sexp_of_t =
    (let (drop_if__009_ : visibility -> Stdlib.Bool.t) =
      fun vis -> vis = Vis_default
     and (drop_if__018_ : docstring -> Stdlib.Bool.t) = Docstring.is_empty
     and (drop_if__023_ : bool -> Stdlib.Bool.t) = fun x -> x = false
     and (drop_if__028_ : bool -> Stdlib.Bool.t) = fun x -> x = false in
     fun {
           ty_constr = ty_constr__002_;
           ty_arity = ty_arity__004_;
           ty_desc = ty_desc__006_;
           ty_vis = ty_vis__010_;
           ty_params_ = ty_params___013_;
           ty_loc_ = ty_loc___015_;
           ty_doc_ = ty_doc___019_;
           ty_is_only_tag_enum_ = ty_is_only_tag_enum___024_;
           ty_is_suberror_ = ty_is_suberror___029_;
         } ->
       let bnds__001_ = ([] : _ Stdlib.List.t) in
       let bnds__001_ =
         if drop_if__028_ ty_is_suberror___029_ then bnds__001_
         else
           let arg__031_ = Moon_sexp_conv.sexp_of_bool ty_is_suberror___029_ in
           let bnd__030_ = S.List [ S.Atom "ty_is_suberror_"; arg__031_ ] in
           (bnd__030_ :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         if drop_if__023_ ty_is_only_tag_enum___024_ then bnds__001_
         else
           let arg__026_ =
             Moon_sexp_conv.sexp_of_bool ty_is_only_tag_enum___024_
           in
           let bnd__025_ =
             S.List [ S.Atom "ty_is_only_tag_enum_"; arg__026_ ]
           in
           (bnd__025_ :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         if drop_if__018_ ty_doc___019_ then bnds__001_
         else
           let arg__021_ = sexp_of_docstring ty_doc___019_ in
           let bnd__020_ = S.List [ S.Atom "ty_doc_"; arg__021_ ] in
           (bnd__020_ :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__016_ = sexp_of_location ty_loc___015_ in
         (S.List [ S.Atom "ty_loc_"; arg__016_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__014_ = sexp_of_tvar_env ty_params___013_ in
         (S.List [ S.Atom "ty_params_"; arg__014_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         if drop_if__009_ ty_vis__010_ then bnds__001_
         else
           let arg__012_ = sexp_of_visibility ty_vis__010_ in
           let bnd__011_ = S.List [ S.Atom "ty_vis"; arg__012_ ] in
           (bnd__011_ :: bnds__001_ : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__007_ = sexp_of_type_components ty_desc__006_ in
         (S.List [ S.Atom "ty_desc"; arg__007_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__005_ = Moon_sexp_conv.sexp_of_int ty_arity__004_ in
         (S.List [ S.Atom "ty_arity"; arg__005_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       let bnds__001_ =
         let arg__003_ = sexp_of_path ty_constr__002_ in
         (S.List [ S.Atom "ty_constr"; arg__003_ ] :: bnds__001_
           : _ Stdlib.List.t)
       in
       S.List bnds__001_
      : t -> S.t)

  and sexp_of_constructors =
    (fun x__032_ -> Moon_sexp_conv.sexp_of_list sexp_of_constructor x__032_
      : constructors -> S.t)

  and sexp_of_fields =
    (fun x__033_ -> Moon_sexp_conv.sexp_of_list sexp_of_field x__033_
      : fields -> S.t)

  and sexp_of_newtype_info =
    (let (drop_if__040_ : bool -> Stdlib.Bool.t) = fun recur -> not recur in
     fun {
           newtype_constr = newtype_constr__035_;
           underlying_typ = underlying_typ__037_;
           recursive = recursive__041_;
         } ->
       let bnds__034_ = ([] : _ Stdlib.List.t) in
       let bnds__034_ =
         if drop_if__040_ recursive__041_ then bnds__034_
         else
           let arg__043_ = Moon_sexp_conv.sexp_of_bool recursive__041_ in
           let bnd__042_ = S.List [ S.Atom "recursive"; arg__043_ ] in
           (bnd__042_ :: bnds__034_ : _ Stdlib.List.t)
       in
       let bnds__034_ =
         let arg__038_ = sexp_of_typ underlying_typ__037_ in
         (S.List [ S.Atom "underlying_typ"; arg__038_ ] :: bnds__034_
           : _ Stdlib.List.t)
       in
       let bnds__034_ =
         let arg__036_ = sexp_of_constructor newtype_constr__035_ in
         (S.List [ S.Atom "newtype_constr"; arg__036_ ] :: bnds__034_
           : _ Stdlib.List.t)
       in
       S.List bnds__034_
      : newtype_info -> S.t)

  and sexp_of_type_components =
    (function
     | Extern_type -> S.Atom "Extern_type"
     | Abstract_type -> S.Atom "Abstract_type"
     | Error_type arg0__044_ ->
         let res0__045_ = sexp_of_constructor arg0__044_ in
         S.List [ S.Atom "Error_type"; res0__045_ ]
     | ErrorEnum_type arg0__046_ ->
         let res0__047_ =
           Moon_sexp_conv.sexp_of_list sexp_of_constructor arg0__046_
         in
         S.List [ S.Atom "ErrorEnum_type"; res0__047_ ]
     | New_type arg0__048_ ->
         let res0__049_ = sexp_of_newtype_info arg0__048_ in
         S.List [ S.Atom "New_type"; res0__049_ ]
     | Variant_type arg0__050_ ->
         let res0__051_ =
           Moon_sexp_conv.sexp_of_list sexp_of_constructor arg0__050_
         in
         S.List [ S.Atom "Variant_type"; res0__051_ ]
     | Record_type
         {
           fields = fields__053_;
           has_private_field_ = has_private_field___055_;
         } ->
         let bnds__052_ = ([] : _ Stdlib.List.t) in
         let bnds__052_ =
           let arg__056_ =
             Moon_sexp_conv.sexp_of_bool has_private_field___055_
           in
           (S.List [ S.Atom "has_private_field_"; arg__056_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         let bnds__052_ =
           let arg__054_ =
             Moon_sexp_conv.sexp_of_list sexp_of_field fields__053_
           in
           (S.List [ S.Atom "fields"; arg__054_ ] :: bnds__052_
             : _ Stdlib.List.t)
         in
         S.List (S.Atom "Record_type" :: bnds__052_)
      : type_components -> S.t)

  and sexp_of_type_component_visibility =
    (function
     | Invisible -> S.Atom "Invisible"
     | Readable -> S.Atom "Readable"
     | Read_write -> S.Atom "Read_write"
      : type_component_visibility -> S.t)

  and sexp_of_constructor =
    (let (drop_if__067_ : type_component_visibility -> Stdlib.Bool.t) =
      fun vis -> vis = Invisible
     and (drop_if__074_ : fn_arity -> Stdlib.Bool.t) = Fn_arity.is_simple
     and (drop_if__079_ : location -> Stdlib.Bool.t) = hide_loc
     and (drop_if__084_ : location -> Stdlib.Bool.t) = hide_loc in
     fun {
           constr_name = constr_name__058_;
           cs_args = cs_args__060_;
           cs_res = cs_res__062_;
           cs_tag = cs_tag__064_;
           cs_vis = cs_vis__068_;
           cs_ty_params_ = cs_ty_params___071_;
           cs_arity_ = cs_arity___075_;
           cs_constr_loc_ = cs_constr_loc___080_;
           cs_loc_ = cs_loc___085_;
         } ->
       let bnds__057_ = ([] : _ Stdlib.List.t) in
       let bnds__057_ =
         if drop_if__084_ cs_loc___085_ then bnds__057_
         else
           let arg__087_ = sexp_of_location cs_loc___085_ in
           let bnd__086_ = S.List [ S.Atom "cs_loc_"; arg__087_ ] in
           (bnd__086_ :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         if drop_if__079_ cs_constr_loc___080_ then bnds__057_
         else
           let arg__082_ = sexp_of_location cs_constr_loc___080_ in
           let bnd__081_ = S.List [ S.Atom "cs_constr_loc_"; arg__082_ ] in
           (bnd__081_ :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         if drop_if__074_ cs_arity___075_ then bnds__057_
         else
           let arg__077_ = sexp_of_fn_arity cs_arity___075_ in
           let bnd__076_ = S.List [ S.Atom "cs_arity_"; arg__077_ ] in
           (bnd__076_ :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         let arg__072_ = sexp_of_tvar_env cs_ty_params___071_ in
         (S.List [ S.Atom "cs_ty_params_"; arg__072_ ] :: bnds__057_
           : _ Stdlib.List.t)
       in
       let bnds__057_ =
         if drop_if__067_ cs_vis__068_ then bnds__057_
         else
           let arg__070_ = sexp_of_type_component_visibility cs_vis__068_ in
           let bnd__069_ = S.List [ S.Atom "cs_vis"; arg__070_ ] in
           (bnd__069_ :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         let arg__065_ = sexp_of_constr_tag cs_tag__064_ in
         (S.List [ S.Atom "cs_tag"; arg__065_ ] :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         let arg__063_ = sexp_of_typ cs_res__062_ in
         (S.List [ S.Atom "cs_res"; arg__063_ ] :: bnds__057_ : _ Stdlib.List.t)
       in
       let bnds__057_ =
         let arg__061_ =
           Moon_sexp_conv.sexp_of_list sexp_of_typ cs_args__060_
         in
         (S.List [ S.Atom "cs_args"; arg__061_ ] :: bnds__057_
           : _ Stdlib.List.t)
       in
       let bnds__057_ =
         let arg__059_ = Moon_sexp_conv.sexp_of_string constr_name__058_ in
         (S.List [ S.Atom "constr_name"; arg__059_ ] :: bnds__057_
           : _ Stdlib.List.t)
       in
       S.List bnds__057_
      : constructor -> S.t)

  and sexp_of_field =
    (let (drop_if__100_ : type_component_visibility -> Stdlib.Bool.t) =
      fun vis -> vis = Invisible
     and (drop_if__109_ : location -> Stdlib.Bool.t) = hide_loc
     and (drop_if__114_ : location -> Stdlib.Bool.t) = hide_loc in
     fun {
           field_name = field_name__089_;
           pos = pos__091_;
           ty_field = ty_field__093_;
           ty_record = ty_record__095_;
           mut = mut__097_;
           vis = vis__101_;
           all_labels = all_labels__104_;
           ty_params_ = ty_params___106_;
           label_loc_ = label_loc___110_;
           loc_ = loc___115_;
         } ->
       let bnds__088_ = ([] : _ Stdlib.List.t) in
       let bnds__088_ =
         if drop_if__114_ loc___115_ then bnds__088_
         else
           let arg__117_ = sexp_of_location loc___115_ in
           let bnd__116_ = S.List [ S.Atom "loc_"; arg__117_ ] in
           (bnd__116_ :: bnds__088_ : _ Stdlib.List.t)
       in
       let bnds__088_ =
         if drop_if__109_ label_loc___110_ then bnds__088_
         else
           let arg__112_ = sexp_of_location label_loc___110_ in
           let bnd__111_ = S.List [ S.Atom "label_loc_"; arg__112_ ] in
           (bnd__111_ :: bnds__088_ : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__107_ = sexp_of_tvar_env ty_params___106_ in
         (S.List [ S.Atom "ty_params_"; arg__107_ ] :: bnds__088_
           : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__105_ =
           Moon_sexp_conv.sexp_of_list Moon_sexp_conv.sexp_of_string
             all_labels__104_
         in
         (S.List [ S.Atom "all_labels"; arg__105_ ] :: bnds__088_
           : _ Stdlib.List.t)
       in
       let bnds__088_ =
         if drop_if__100_ vis__101_ then bnds__088_
         else
           let arg__103_ = sexp_of_type_component_visibility vis__101_ in
           let bnd__102_ = S.List [ S.Atom "vis"; arg__103_ ] in
           (bnd__102_ :: bnds__088_ : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__098_ = Moon_sexp_conv.sexp_of_bool mut__097_ in
         (S.List [ S.Atom "mut"; arg__098_ ] :: bnds__088_ : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__096_ = sexp_of_typ ty_record__095_ in
         (S.List [ S.Atom "ty_record"; arg__096_ ] :: bnds__088_
           : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__094_ = sexp_of_typ ty_field__093_ in
         (S.List [ S.Atom "ty_field"; arg__094_ ] :: bnds__088_
           : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__092_ = Moon_sexp_conv.sexp_of_int pos__091_ in
         (S.List [ S.Atom "pos"; arg__092_ ] :: bnds__088_ : _ Stdlib.List.t)
       in
       let bnds__088_ =
         let arg__090_ = Moon_sexp_conv.sexp_of_string field_name__089_ in
         (S.List [ S.Atom "field_name"; arg__090_ ] :: bnds__088_
           : _ Stdlib.List.t)
       in
       S.List bnds__088_
      : field -> S.t)

  let _ = sexp_of_t
  and _ = sexp_of_constructors
  and _ = sexp_of_fields
  and _ = sexp_of_newtype_info
  and _ = sexp_of_type_components
  and _ = sexp_of_type_component_visibility
  and _ = sexp_of_constructor
  and _ = sexp_of_field
end

type alias = {
  name : string;
  arity : int;
  ty_params : tvar_env;
  alias : Stype.t;
  is_pub : bool;
  doc_ : docstring;
  loc_ : location; [@sexp_drop_if hide_loc]
}

include struct
  let _ = fun (_ : alias) -> ()

  let sexp_of_alias =
    (let (drop_if__132_ : location -> Stdlib.Bool.t) = hide_loc in
     fun {
           name = name__119_;
           arity = arity__121_;
           ty_params = ty_params__123_;
           alias = alias__125_;
           is_pub = is_pub__127_;
           doc_ = doc___129_;
           loc_ = loc___133_;
         } ->
       let bnds__118_ = ([] : _ Stdlib.List.t) in
       let bnds__118_ =
         if drop_if__132_ loc___133_ then bnds__118_
         else
           let arg__135_ = sexp_of_location loc___133_ in
           let bnd__134_ = S.List [ S.Atom "loc_"; arg__135_ ] in
           (bnd__134_ :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__130_ = sexp_of_docstring doc___129_ in
         (S.List [ S.Atom "doc_"; arg__130_ ] :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__128_ = Moon_sexp_conv.sexp_of_bool is_pub__127_ in
         (S.List [ S.Atom "is_pub"; arg__128_ ] :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__126_ = Stype.sexp_of_t alias__125_ in
         (S.List [ S.Atom "alias"; arg__126_ ] :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__124_ = sexp_of_tvar_env ty_params__123_ in
         (S.List [ S.Atom "ty_params"; arg__124_ ] :: bnds__118_
           : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__122_ = Moon_sexp_conv.sexp_of_int arity__121_ in
         (S.List [ S.Atom "arity"; arg__122_ ] :: bnds__118_ : _ Stdlib.List.t)
       in
       let bnds__118_ =
         let arg__120_ = Moon_sexp_conv.sexp_of_string name__119_ in
         (S.List [ S.Atom "name"; arg__120_ ] :: bnds__118_ : _ Stdlib.List.t)
       in
       S.List bnds__118_
      : alias -> S.t)

  let _ = sexp_of_alias
end
